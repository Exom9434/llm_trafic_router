import os
import time
import uuid
import json
import random
import schedule
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# 1. 초기 설정
# ──────────────────────────────────────────────
load_dotenv()

REPEATS             = 3          # 문제당 반복 횟수
QUESTIONS_FILE      = "questions.json"
SYSTEM_PROMPT_FILE  = "system_prompt.txt"

def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_system_prompt()

# 각 API 키가 .env에 잘 있는지 확인용
oa_client     = openai.OpenAI()
ant_client    = Anthropic()
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ──────────────────────────────────────────────
# 2. 유틸 함수
# ──────────────────────────────────────────────
def load_questions() -> list:
    """questions.json에서 문제 목록 로드"""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def append_to_csv(result, filename="benchmark_results.csv"):
    """결과를 CSV에 한 줄씩 누적 저장"""
    df = pd.DataFrame([result])
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=header, encoding='utf-8-sig')


# ──────────────────────────────────────────────
# 3. 핵심 벤치마크 함수
# ──────────────────────────────────────────────
def record_benchmark(provider, model_name, prompt, repeat_index=1):
    request_id = str(uuid.uuid4())
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        "raw_response":     None
    }

    start_time       = time.time()
    first_token_time = None

    try:
        if provider == "openai":
            response = oa_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0,
                stream=True,
                stream_options={"include_usage": True}  # 마지막 청크에 usage 포함
            )
            chunks = []
            for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    metrics["full_content"] += chunk.choices[0].delta.content
                if hasattr(chunk, 'system_fingerprint'):
                    metrics["system_fingerprint"] = chunk.system_fingerprint
                # 마지막 청크의 usage에서 실제 토큰 수 추출
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
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    metrics["full_content"] += text
                final_msg = stream.get_final_message()
                metrics["raw_response"]   = final_msg.model_dump()
                metrics["output_tokens"]  = final_msg.usage.output_tokens  # 정확한 토큰 수

        elif provider == "google":
            response = google_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=1000
                )
            )
            first_token_time         = time.time()  # 비스트리밍이므로 응답 수신 시점
            metrics["full_content"]  = response.text
            metrics["system_fingerprint"] = response.model_version
            metrics["output_tokens"] = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            metrics["raw_response"]  = {
                "model_version": response.model_version,
                "output_tokens": metrics["output_tokens"]
            }

        # 지표 계산
        end_time = time.time()
        metrics["ttft"]            = first_token_time - start_time if first_token_time else 0
        metrics["generation_time"] = end_time - first_token_time  if first_token_time else 0

        # TPS: API가 반환한 실제 토큰 수 우선 사용, 없을 경우 글자 수 기반 추정으로 폴백
        token_count = (
            metrics["output_tokens"]
            if metrics["output_tokens"] > 0
            else len(metrics["full_content"]) / 4
        )
        metrics["tps"] = token_count / metrics["generation_time"] if metrics["generation_time"] > 0 else 0

        os.makedirs("logs", exist_ok=True)
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics

    except Exception as e:
        print(f"Error with {provider}: {e}")
        return None


# ──────────────────────────────────────────────
# 4. 실험 사이클
# ──────────────────────────────────────────────
def run_experiment():
    targets = [
        {"provider": "openai",    "model": "gpt-5-nano-2025-08-07"},
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        {"provider": "google",    "model": "gemini-3.1-flash-lite-preview"}
    ]

    problems = load_questions()
    print(f"📋 문제 {len(problems)}개 × {REPEATS}회 × {len(targets)}개 모델 "
          f"= 총 {len(problems) * REPEATS * len(targets)}건")

    for prob in problems:
        question = (
            f"Question: {prob['question']}\n"
            f"Options: {prob['options']}\n"
            f"Answer with only the option letter."
        )

        for repeat_idx in range(1, REPEATS + 1):
            print(f"\n[{prob.get('subject', 'unknown')} / {prob.get('difficulty', '?')}] "
                  f"{repeat_idx}/{REPEATS}회")

            for target in targets:
                result = record_benchmark(
                    target['provider'], target['model'], question,
                    repeat_index=repeat_idx
                )

                if result:
                    result.update({
                        "difficulty":     prob.get("difficulty", "unknown"),
                        "subject":        prob.get("subject", "unknown"),
                        "correct_answer": prob["answer"],
                        "repeat_index":   repeat_idx,
                        "is_correct":     1 if result["full_content"].strip().upper()[:1] == prob["answer"] else 0
                    })

                    append_to_csv(result)
                    print(f"   저장 완료: {target['model']} | repeat {repeat_idx}")

                time.sleep(2)

    print(f"실험 사이클 완료: {datetime.now()}")


# ──────────────────────────────────────────────
# 5. 스케줄러
# ──────────────────────────────────────────────
def job():
    print(f"[{datetime.now()}] 벤치마크 테스트 시작...")
    run_experiment()


# 논-피크 시간대 (KST 오후 6시)
schedule.every().day.at("18:00").do(job)

# 피크 시간대 (KST 오전 6시)
schedule.every().day.at("06:00").do(job)

print("스케줄러 시작... 대기 중.")
while True:
    schedule.run_pending()
    time.sleep(60)
