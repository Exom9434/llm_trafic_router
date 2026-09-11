"""무인 실행 감시. 침묵 탐지와 하루 요약.

21일 무인 실행에서 러너는 실패보다 조용히 죽는 쪽이 흔하다. 예외로 죽으면
systemd가 다시 띄우지만, 매달린 연결이나 잘못된 대기로 살아 있는 채 아무것도
안 하는 상태는 프로세스가 살아 있으므로 재시작이 걸리지 않는다. 그 상태를
구분하는 유일한 신호가 로그가 자라는지다.

한 시간마다 돌며 두 가지를 한다.

  1. **침묵 탐지** — 콜 로그에 새 줄이 붙지 않은 시간이 기준을 넘으면 알린다.
     슬롯 간격이 3시간이므로 4시간이면 슬롯 하나를 통째로 건너뛴 것이다.
  2. **하루 요약** — 날이 바뀌면 지난 날의 완주 여부, 콜 수, 오류율, 파싱
     실패율, 모델별 지출을 한 번 알린다.

로그를 매번 통째로 읽지 않는다. 본실험 로그가 300MB까지 자라므로 읽은
위치를 기억해 두고 새로 붙은 부분만 센다.

알림 경로는 `.env`의 `NOTIFY_WEBHOOK`이다. Slack·Discord·ntfy처럼 JSON을
받는 주소면 된다. 비어 있으면 journal에만 남는다.

실행:
    python monitor.py            # 한 번 점검한다 (systemd timer가 부른다)
    python monitor.py --summary  # 침묵과 무관하게 지금까지의 요약을 낸다
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from budget import DayLedger
from config import OUTPUT_DIR, apply_capabilities, get_models

MAIN_LOG = OUTPUT_DIR / "main_calls.jsonl"
STATE = OUTPUT_DIR / "monitor_state.json"
SUMMARY_MD = OUTPUT_DIR / "daily_summary.md"

# 슬롯 간격이 3시간이다. 4시간이면 슬롯 하나를 통째로 건너뛴 것이므로
# 그날은 이미 완주가 아니다. 더 늦게 잡을 이유가 없다.
SILENCE_HOURS = 4.0


def notify(title: str, body: str) -> None:
    print(f"== {title} ==\n{body}", flush=True)
    url = os.getenv("NOTIFY_WEBHOOK")
    if not url:
        return
    try:
        requests.post(url, json={"text": f"*{title}*\n{body}"}, timeout=15)
    except requests.RequestException as e:
        print(f"알림 전송 실패: {e}", flush=True)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"offset": 0, "days": {}, "last_summary_day": None, "silent_since": None}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(state: dict) -> int:
    """로그에 새로 붙은 줄만 읽어 날짜·모델별로 센다."""
    if not MAIN_LOG.exists():
        return 0
    size = MAIN_LOG.stat().st_size
    offset = state.get("offset", 0)
    if size < offset:          # 로그가 갈렸다. 처음부터 다시 센다.
        offset, state["days"] = 0, {}

    added = 0
    with open(MAIN_LOG, "r", encoding="utf-8") as f:
        f.seek(offset)
        for line in f:
            if not line.endswith("\n"):     # 쓰는 중인 줄. 다음 회차에 읽는다.
                break
            offset += len(line.encode("utf-8"))
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            day = (rec.get("ts_utc") or "")[:10]
            key = rec.get("model_key")
            if not day or not key:
                continue
            bucket = state["days"].setdefault(day, {}).setdefault(
                key, {"calls": 0, "errors": 0, "unparsed": 0, "latency": 0,
                      "no_ttft": 0})
            bucket["calls"] += 1
            if rec.get("error"):
                bucket["errors"] += 1
            elif rec.get("probe") == "latency":
                bucket["latency"] += 1
                if rec.get("ttft_ms") is None:
                    bucket["no_ttft"] += 1
            elif rec.get("parsed_letter") is None:
                bucket["unparsed"] += 1
            added += 1

    state["offset"] = offset
    return added


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "-"


def day_report(state: dict, day: str, ledger: DayLedger) -> str:
    rows = state["days"].get(day, {})
    status = {}
    if ledger.path.exists():
        for line in ledger.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("day") == day:
                status[r["model_key"]] = r

    out = [f"{day} 요약", ""]
    out.append(f"  {'모델':<28} {'완주':<12} {'콜':>8} {'오류':>7} {'파싱실패':>8}")
    for key in sorted(rows):
        r = rows[key]
        st = status.get(key, {})
        mark = st.get("status", "진행 중")
        if mark == "aborted":
            mark = f"aborted {st.get('completed_slots')}/{st.get('planned_slots')}"
        quality = r["calls"] - r["errors"] - r["latency"]
        out.append(f"  {key:<28} {mark:<12} {r['calls']:>8,} "
                   f"{pct(r['errors'], r['calls']):>7} "
                   f"{pct(r['unparsed'], quality):>8}")
    no_ttft = sum(r["no_ttft"] for r in rows.values())
    if no_ttft:
        out.append(f"\n  TTFT가 빈 지연 콜 {no_ttft}회 — 스트리밍이 안 된 모델이 있다.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="본실험 감시")
    ap.add_argument("--summary", action="store_true", help="지금까지의 요약을 낸다")
    ap.add_argument("--silence-hours", type=float, default=SILENCE_HOURS)
    args = ap.parse_args()

    state = load_state()
    added = ingest(state)
    now = datetime.now(timezone.utc)
    ledger = DayLedger()

    if args.summary:
        for day in sorted(state["days"]):
            print(day_report(state, day, ledger))
            print()
        save_state(state)
        return

    # 침묵 탐지
    if MAIN_LOG.exists():
        quiet_h = (now.timestamp() - MAIN_LOG.stat().st_mtime) / 3600
        if quiet_h >= args.silence_hours:
            if not state.get("silent_since"):
                state["silent_since"] = now.isoformat()
                notify("본실험 침묵",
                       f"콜 로그가 {quiet_h:.1f}시간째 자라지 않는다. "
                       f"슬롯 간격이 3시간이므로 슬롯을 건너뛰고 있다.\n"
                       f"  systemctl status llm-experiment\n"
                       f"  journalctl -u llm-experiment -n 50")
        elif state.get("silent_since"):
            state["silent_since"] = None
            notify("본실험 복구", f"콜 로그가 다시 자란다. 새 줄 {added:,}회.")
    else:
        notify("본실험 로그 없음", f"{MAIN_LOG} 가 아직 없다. 러너가 뜨지 않았을 수 있다.")

    # 하루 요약
    yesterday = (now - timedelta(days=1)).date().isoformat()
    if state.get("last_summary_day") != yesterday and yesterday in state["days"]:
        report = day_report(state, yesterday, ledger)
        notify("본실험 하루 요약", report)
        SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_MD, "a", encoding="utf-8") as f:
            f.write(report + "\n\n")
        state["last_summary_day"] = yesterday

    save_state(state)


if __name__ == "__main__":
    main()
