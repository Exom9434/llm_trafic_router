"""
api_benchmark.py
────────────────
피크 / 논-피크 시간대별 LLM API 벤치마크 스케줄러

■ 수집 스케줄 (미국 동부시간 ET — EST/EDT 자동 처리)
  · 피크:    10:00 ET  (Claude 기준 피크: 8AM-2PM ET)
  · 논-피크: 22:00 ET

■ 실행 기간: 4주 (28일)

■ 연속성 보장
  · progress.json 에 슬롯별 상태("done" | "running") 및 task 단위 완료 기록
  · 재시작 시 "done" 슬롯은 스킵, "running" 슬롯은 중단된 task부터 재실행
  · 4주가 채워질 때까지 자동 반복

■ progress.json 구조
  {
      "start_date": "YYYY-MM-DD",          ← ET 기준 시작일
      "slots": {
          "20240315_peak_1000":    "done",
          "20240315_nonpeak_2200": "running"
      },
      "tasks": {
          "20240315_peak_1000": [
              "q0_openai_r1",
              "q0_anthropic_r1",
              ...
          ]
      }
  }

■ 실행 방법
  1. 의존성 설치
     pip install apscheduler pytz

  2. .env 파일에 API 키 설정
     OPENAI_API_KEY=...
     ANTHROPIC_API_KEY=...
     GOOGLE_API_KEY=...
     DASHSCOPE_API_KEY=...

  3. questions.json 준비
     python prepare_questions.py

  4. 스케줄러 실행
     python api_benchmark.py
"""

import os
import time
import uuid
import json
import sys
import pandas as pd
from datetime import datetime, date, timedelta

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from dotenv import load_dotenv
import openai
from anthropic import Anthropic
from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# 1. 초기 설정
# ──────────────────────────────────────────────
load_dotenv()

REPEATS            = 3
QUESTIONS_FILE     = "questions.json"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
PROGRESS_FILE      = "progress.json"
DURATION_DAYS      = 28
ET                 = pytz.timezone("America/New_York")

# 슬롯 정의 — (slot_type, hour, minute)
SLOTS = [
    ("peak",    10, 0),   # 10:00 ET — 피크 구간(8AM-2PM ET) 한가운데
    ("nonpeak", 22, 0),   # 22:00 ET — 논-피크 구간
]

def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_system_prompt()

oa_client     = openai.OpenAI()
ant_client    = Anthropic()
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
qwen_client   = openai.OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1"
)

TARGETS = [
    {"provider": "openai",    "model": "gpt-4o-mini-2024-07-18"},
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    {"provider": "google",    "model": "gemini-3.1-flash-lite-preview"},
    {"provider": "qwen",      "model": "qwen3.5-flash"},
]


# ──────────────────────────────────────────────
# 2. progress.json 관리
# ──────────────────────────────────────────────
def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    today_et = datetime.now(ET).date()
    progress = {"start_date": str(today_et), "slots": {}, "tasks": {}}
    save_progress(progress)
    return progress


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_start_date(progress: dict) -> date:
    return date.fromisoformat(progress["start_date"])


def elapsed_days(progress: dict) -> int:
    return (datetime.now(ET).date() - get_start_date(progress)).days


def get_slot_status(progress: dict, slot_key: str) -> str | None:
    return progress["slots"].get(slot_key)


def mark_slot_running(progress: dict, slot_key: str):
    progress["slots"][slot_key] = "running"
    save_progress(progress)


def mark_slot_done(progress: dict, slot_key: str):
    progress["slots"][slot_key] = "done"
    save_progress(progress)


def get_done_tasks(progress: dict, slot_key: str) -> set:
    return set(progress.get("tasks", {}).get(slot_key, []))


def mark_task_done(progress: dict, slot_key: str, task_key: str):
    if "tasks" not in progress:
        progress["tasks"] = {}
    if slot_key not in progress["tasks"]:
        progress["tasks"][slot_key] = []
    if task_key not in progress["tasks"][slot_key]:
        progress["tasks"][slot_key].append(task_key)
    save_progress(progress)


def make_task_key(q_idx: int, provider: str, repeat_idx: int) -> str:
    """task 고유 키: 'q{인덱스}_{provider}_r{반복번호}'"""
    return f"q{q_idx}_{provider}_r{repeat_idx}"


# ──────────────────────────────────────────────
# 3. 유틸 함수
# ──────────────────────────────────────────────
def load_questions() -> list:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def append_to_csv(result: dict, period: str, filename_prefix="benchmark_results"):
    date_str = datetime.now(ET).strftime("%Y%m%d")
    filename = f"{filename_prefix}_{date_str}_{period}.csv"
    df       = pd.DataFrame([result])
    header   = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=header, encoding='utf-8-sig')
    return filename


# ──────────────────────────────────────────────
# 4. 핵심 벤치마크 함수
# ──────────────────────────────────────────────
def record_benchmark(provider: str, model_name: str, prompt: str, repeat_index: int = 1) -> dict | None:
    request_id   = str(uuid.uuid4())
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{provider}_{timestamp}_{request_id}.json"

    metrics = {
        "request_id":       request_id,
        "timestamp":        timestamp,
        "provider":         provider,
        "model_requested":  model_name,
        "repeat_index":     repeat_index,
        "ttft":             0,
        "generation_time":  0,
        "output_tokens":    0,
        "tps":              0,
        "full_content":     "",
        "system_fingerprint": None,
        "raw_response":     None,
    }

    start_time       = time.time()
    first_token_time = None

    try:
        if provider == "openai":
            response = oa_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = []
            for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    metrics["full_content"] += chunk.choices[0].delta.content
                if hasattr(chunk, "system_fingerprint"):
                    metrics["system_fingerprint"] = chunk.system_fingerprint
                if chunk.usage and chunk.usage.completion_tokens:
                    metrics["output_tokens"] = chunk.usage.completion_tokens
                chunks.append(chunk.model_dump())
            metrics["raw_response"] = chunks

        elif provider == "anthropic":
            with ant_client.messages.stream(
                model=model_name,
                system=SYSTEM_PROMPT,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    metrics["full_content"] += text
                final_msg = stream.get_final_message()
                metrics["raw_response"]  = final_msg.model_dump()
                metrics["output_tokens"] = final_msg.usage.output_tokens

        elif provider == "google":
            response  = google_client.models.generate_content_stream(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=1000,
                ),
            )
            last_chunk = None
            for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()
                if chunk.text:
                    metrics["full_content"] += chunk.text
                last_chunk = chunk
            if last_chunk and last_chunk.usage_metadata:
                metrics["output_tokens"] = last_chunk.usage_metadata.candidates_token_count or 0
            metrics["raw_response"] = {"model_version": model_name}

        elif provider == "qwen":
            response = qwen_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = []
            for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    metrics["full_content"] += chunk.choices[0].delta.content
                if chunk.usage and chunk.usage.completion_tokens:
                    metrics["output_tokens"] = chunk.usage.completion_tokens
                chunks.append(chunk.model_dump())
            metrics["raw_response"] = chunks

        # ── 지표 계산 (공통) ──────────────────────
        end_time                   = time.time()
        metrics["ttft"]            = first_token_time - start_time if first_token_time else 0
        metrics["generation_time"] = end_time - first_token_time   if first_token_time else 0
        token_count = (
            metrics["output_tokens"]
            if metrics["output_tokens"] > 0
            else len(metrics["full_content"]) / 2
        )
        metrics["tps"] = token_count / metrics["generation_time"] if metrics["generation_time"] > 0 else 0

        os.makedirs("logs", exist_ok=True)
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics

    except Exception as e:
        print(f"   ❌ {provider} 오류: {e}")
        return None


# ──────────────────────────────────────────────
# 5. 실험 사이클 (재개 지원)
# ──────────────────────────────────────────────
def run_experiment(
    period: str = "unknown",
    slot_key: str = "",
    progress: dict = None,
):
    """
    Parameters
    ----------
    period   : "peak" | "nonpeak"  — CSV 파일명 구분용
    slot_key : progress.json 기록용 슬롯 키 (비어있으면 재개 기능 비활성)
    progress : 공유 progress dict  (None 이면 재개 기능 비활성)
    """
    print(f"=== 실험 사이클 시작: {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET "
          f"| 슬롯: {slot_key or '(수동)'} ===")

    problems   = load_questions()
    done_tasks = get_done_tasks(progress, slot_key) if (progress and slot_key) else set()

    total      = len(problems) * REPEATS * len(TARGETS)
    skip_count = sum(
        1 for q_idx in range(len(problems))
        for t in TARGETS
        for r in range(1, REPEATS + 1)
        if make_task_key(q_idx, t["provider"], r) in done_tasks
    )

    print(f"📋 문제 {len(problems)}개 × {REPEATS}회 × {len(TARGETS)}개 모델 = 총 {total}건")
    if skip_count:
        print(f"⏭  이미 완료된 task {skip_count}건 → 스킵 (남은 작업: {total - skip_count}건)")

    for q_idx, prob in enumerate(problems):
        question = (
            f"Question: {prob['question']}\n"
            f"Options: {prob['options']}\n"
            f"Answer with only the option letter."
        )

        for repeat_idx in range(1, REPEATS + 1):

            # 이 (문제, repeat) 조합에서 아직 완료 안 된 provider만 추림
            pending = [
                t for t in TARGETS
                if make_task_key(q_idx, t["provider"], repeat_idx) not in done_tasks
            ]
            if not pending:
                continue

            print(f"\n[{prob.get('subject', '?')} / {prob.get('difficulty', '?')}] "
                  f"q{q_idx} {repeat_idx}/{REPEATS}회"
                  f"{'  (일부 재개)' if len(pending) < len(TARGETS) else ''}")

            for target in pending:
                t_key  = make_task_key(q_idx, target["provider"], repeat_idx)
                result = record_benchmark(
                    target["provider"], target["model"], question,
                    repeat_index=repeat_idx,
                )

                if result:
                    result.update({
                        "difficulty":     prob.get("difficulty", "unknown"),
                        "subject":        prob.get("subject", "unknown"),
                        "correct_answer": prob["answer"],
                        "repeat_index":   repeat_idx,
                        "is_correct":     (
                            1 if result["full_content"].strip().upper()[:1] == prob["answer"] else 0
                        ),
                    })
                    saved_to = append_to_csv(result, period=period)
                    print(f"   ✅ {target['model']} | repeat {repeat_idx} "
                          f"| TTFT {result['ttft']:.2f}s "
                          f"| TPS {result['tps']:.1f} → {saved_to}")

                    # ── task 완료 즉시 기록 ──────────
                    if progress and slot_key:
                        mark_task_done(progress, slot_key, t_key)
                        done_tasks.add(t_key)

                time.sleep(2)

    print(f"=== 실험 사이클 완료: {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET ===")


# ──────────────────────────────────────────────
# 6. 잡 팩토리
# ──────────────────────────────────────────────
def make_job(slot_type: str, hour: int, minute: int, progress: dict, scheduler: BlockingScheduler):
    slot_time = f"{hour:02d}{minute:02d}"
    label     = f"{slot_type}_{slot_time}"   # e.g. "peak_1000"

    def job():
        elapsed = elapsed_days(progress)

        # ── 4주 완료 체크 ──────────────────────────
        if elapsed >= DURATION_DAYS:
            print(f"\n{'='*60}")
            print(f"✅ 4주({DURATION_DAYS}일) 수집 완료! 스케줄러를 종료합니다.")
            scheduler.shutdown(wait=False)
            return

        # ── 슬롯 키 생성 (ET 기준 날짜) ───────────
        today_str = datetime.now(ET).strftime("%Y%m%d")
        slot_key  = f"{today_str}_{label}"   # e.g. "20240315_peak_1000"
        status    = get_slot_status(progress, slot_key)

        if status == "done":
            print(f"[{datetime.now(ET).strftime('%H:%M:%S')} ET] ⏭  {slot_key} 이미 완료 → 스킵")
            return

        if status == "running":
            print(f"[{datetime.now(ET).strftime('%H:%M:%S')} ET] 🔄 {slot_key} 이전에 중단됨 → 재실행")
        else:
            print(f"\n{'='*60}")
            print(f"[{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET] "
                  f"{slot_type.upper()} 세션 시작 — {hour:02d}:{minute:02d} ET "
                  f"| {elapsed+1}/{DURATION_DAYS}일차")
            print(f"{'='*60}")

        mark_slot_running(progress, slot_key)

        try:
            run_experiment(
                period=slot_type,
                slot_key=slot_key,
                progress=progress,
            )
            mark_slot_done(progress, slot_key)
            print(f"✅ {slot_key} 완료 기록됨")
        except Exception as e:
            print(f"❌ 세션 오류 ({slot_key}): {e}")
            print(f"   progress.json에 'running' 상태로 남겨둠 → 재시작 시 자동 재실행")

    return job


# ──────────────────────────────────────────────
# 7. 재시작 시 "running" 슬롯 즉시 재실행
# ──────────────────────────────────────────────
def retry_interrupted_slots(progress: dict, scheduler: BlockingScheduler):
    running_slots = [k for k, v in progress["slots"].items() if v == "running"]
    if not running_slots:
        return

    print(f"\n⚠️  중단된 슬롯 {len(running_slots)}개 감지 → 즉시 재실행 예정:")
    for slot_key in running_slots:
        print(f"   · {slot_key}")

    for slot_key in running_slots:
        parts     = slot_key.split("_")   # ["20240315", "peak", "1000"]
        slot_type = parts[1]
        slot_time = parts[2]
        hour      = int(slot_time[:2])
        minute    = int(slot_time[2:])

        job_fn = make_job(slot_type, hour, minute, progress, scheduler)
        run_at = datetime.now(ET) + timedelta(seconds=3)
        scheduler.add_job(
            job_fn,
            trigger=DateTrigger(run_date=run_at, timezone=ET),
            id=f"retry_{slot_key}",
            replace_existing=True,
        )
        print(f"   → {slot_key} 재실행 잡 등록 ({run_at.strftime('%H:%M:%S')} ET)")


# ──────────────────────────────────────────────
# 8. 스케줄 등록
# ──────────────────────────────────────────────
def register_schedules(progress: dict, scheduler: BlockingScheduler):
    for slot_type, hour, minute in SLOTS:
        job_fn = make_job(slot_type, hour, minute, progress, scheduler)
        scheduler.add_job(
            job_fn,
            trigger=CronTrigger(hour=hour, minute=minute, timezone=ET),
            id=f"cron_{slot_type}_{hour:02d}{minute:02d}",
            replace_existing=True,
        )
        print(f"  ✓ {slot_type:<9} {hour:02d}:{minute:02d} ET 등록")


# ──────────────────────────────────────────────
# 9. 메인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(QUESTIONS_FILE):
        print(f"❌ {QUESTIONS_FILE}가 없습니다. 먼저 prepare_questions.py를 실행하세요.")
        sys.exit(1)

    progress = load_progress()
    elapsed  = elapsed_days(progress)

    if elapsed >= DURATION_DAYS:
        print(f"✅ 이미 4주({DURATION_DAYS}일) 수집 완료. 종료합니다.")
        sys.exit(0)

    done_count    = sum(1 for v in progress["slots"].values() if v == "done")
    running_count = sum(1 for v in progress["slots"].values() if v == "running")
    total_slots   = DURATION_DAYS * len(SLOTS)

    print(f"\n{'='*60}")
    print(f"LLM API 벤치마크 스케줄러 시작  (타임존: America/New_York)")
    print(f"시작일: {get_start_date(progress)} | {elapsed+1}/{DURATION_DAYS}일차")
    print(f"완료 슬롯: {done_count} / {total_slots}  |  중단 슬롯: {running_count}")
    print(f"{'='*60}")

    scheduler = BlockingScheduler(timezone=ET)

    print("\n정기 스케줄 등록 중...")
    register_schedules(progress, scheduler)
    retry_interrupted_slots(progress, scheduler)

    print(f"\n총 {len(scheduler.get_jobs())}개 잡 등록 완료.")
    print("스케줄러 대기 중... (종료: Ctrl+C)\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        done_count = sum(1 for v in progress["slots"].values() if v == "done")
        print(f"\n스케줄러가 수동으로 종료되었습니다.")
        print(f"현재까지 완료된 슬롯: {done_count} / {total_slots}")