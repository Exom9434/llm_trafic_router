"""본실험 러너 (설계서 10절 5단계).

보정 패스 러너(`calibrate.py`)와 콜 실행부를 공유한다. 다른 것은 무엇을
언제 쏘느냐뿐이고, 그 무엇과 언제를 정하는 것이 이 파일이다.

세 가지가 이 파일의 구조를 정한다.

  1. **슬롯 번호를 달력에서 센다.** 조건 균형을 만드는 것은 무작위화가
     아니라 주기의 서로소 관계다. 하루 8슬롯과 은행 묶음 주기 3이 서로소라,
     슬롯 번호가 날을 넘겨 이어져야 모든 묶음이 여덟 시각을 고루 통과한다.
     카운터를 파일에 들고 다니면 중단·재개에서 틀어질 수 있으므로 실행
     첫날로부터의 경과일과 그날의 슬롯 위치로 매번 다시 센다.

  2. **조건 라벨을 실행 시각에 박는다.** 어느 슬롯이 피크인지를 사후에
     계산할 수 있으면 계산 규칙이 다시 자유도가 된다. 제공사 공표 구간으로
     정의를 고정하고(설계서 3.2절) 콜마다 라벨을 로그에 넣는다.

  3. **절단은 날 경계에만 떨어뜨린다.** 하루 도중에 멈추면 그날은 앞쪽
     슬롯만 남아 절단이 시간대와 상관된다. 잡음이 아니라 편향이다. 그래서
     하루를 시작하기 전에 그날 치 예산을 확인하고, 모자라면 그날을 아예
     시작하지 않는다(설계서 8.1절).

실행:
    python experiment.py --dry-run          # 일정·콜 수·비용만 본다
    python experiment.py                    # 본실험 시작
    python experiment.py --max-slots 1      # 슬롯 하나만 돌려 보는 리허설
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from budget import BUDGET_PLAN, PROVIDER_BALANCE, DayLedger, SpendGuard, load_balances
from calllog import JsonlLogger, load_done_keys
from config import (
    ANCHOR_SLOT_HOURS_UTC,
    BANK_CYCLE,
    CONSISTENCY_TEMPERATURE,
    DATA_DIR,
    ITEM_ORDER_SEED,
    LATENCY_MAX_TOKENS,
    LATENCY_REPS_PER_SLOT,
    LATENCY_TEMPERATURE,
    MAIN_DESIGN,
    BALANCE_USABLE_FRACTION,
    OUTPUT_DIR,
    RUN_STOP_DAY,
    SLOT_HOURS_UTC,
    US_DST_END,
    ModelSpec,
    apply_capabilities,
    condition_label,
    get_models,
    item_group,
    slot_index,
)
from core import CallSpec, run_batch
import netbase
from providers import build_adapter

MAIN_LOG = OUTPUT_DIR / "main_calls.jsonl"
STATE_FILE = OUTPUT_DIR / "experiment_state.json"
SLOT_STATUS = OUTPUT_DIR / "slot_status.jsonl"
ITEM_BANK = DATA_DIR / "item_bank.json"

# 슬롯 시작 시각에서 이만큼 지났으면 그 슬롯은 버린다.
#
# 재시작이 슬롯 한가운데 떨어지면 그 슬롯을 지금이라도 돌릴 것인가가 문제가
# 된다. 돌리면 관측이 라벨의 시각에서 멀어지고, 버리면 그 슬롯이 빈다.
# 슬롯 간격이 3시간이므로 30분이면 라벨 오차가 간격의 6분의 1 안에 묶인다.
# 그보다 늦으면 다음 슬롯을 기다린다.
SLOT_CATCHUP_MINUTES = 30

# 슬롯 하나를 도는 데 필요한 시간의 넉넉한 상한. 보정 실측에서 슬롯 하나가
# 4~17분이었다. 이 시간이 남지 않은 슬롯은 시작하지 않는다.
SLOT_BUDGET_MINUTES = 45


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")
    print(f"[{stamp}Z] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# 문항 묶음
# ─────────────────────────────────────────────────────────────

def load_groups(bank_path: Path | None = None) -> list[list[dict]]:
    """은행을 고정 시드로 한 번 섞어 세 묶음으로 가른다.

    은행이 과목순으로 정렬돼 있어 그대로 100개씩 자르면 묶음마다 과목 구성이
    달라진다. 묶음이 곧 시각 배정 단위이므로(묶음 하나가 한 슬롯을 돈다)
    그러면 과목과 시각이 교락된다. 시드를 박아 두는 것은 중단·재개를 건너서도
    같은 순서가 나오게 하기 위해서다.
    """
    path = bank_path or ITEM_BANK
    if not path.exists():
        sys.exit(f"문항 은행이 없다: {path}\n먼저 select_bank.py를 돌려야 한다.")
    items = json.loads(path.read_text(encoding="utf-8"))

    size = MAIN_DESIGN["items_per_slot"]
    need = size * BANK_CYCLE
    if len(items) != need:
        sys.exit(f"은행 크기가 {len(items)}이다. 설계는 {need}을 전제한다 "
                 f"(슬롯당 {size} × 묶음 {BANK_CYCLE}).")

    ordered = sorted(items, key=lambda it: str(it.get("item_id")))
    random.Random(ITEM_ORDER_SEED).shuffle(ordered)
    return [ordered[g * size:(g + 1) * size] for g in range(BANK_CYCLE)]


# ─────────────────────────────────────────────────────────────
# 상태
# ─────────────────────────────────────────────────────────────

def load_state(start_day: date | None = None) -> dict:
    """실행 첫날과 run_id. 슬롯 번호의 기준점이라 한 번 정하면 안 바꾼다."""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return state
    state = {
        "start_day": (start_day or datetime.now(timezone.utc).date()).isoformat(),
        "run_id": uuid.uuid4().hex[:12],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def load_slot_status() -> set[tuple[str, str]]:
    """이미 발사한 (모델, 슬롯) 쌍.

    완주 판정을 성공 콜 수가 아니라 발사 여부로 하는 이유가 있다. 영구
    실패하는 콜 하나(예: 안전 필터에 걸리는 문항) 때문에 그날이 통째로
    빠지면, 절단이 데이터의 성질과 상관되어 8.1절의 방어가 무너진다.
    오류율은 별도로 집계해 보고할 문제이지 완주 판정의 조건이 아니다.
    """
    done: set[tuple[str, str]] = set()
    if not SLOT_STATUS.exists():
        return done
    for line in SLOT_STATUS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add((row["model_key"], row["slot"]))
    return done


def mark_slot(model_key: str, slot: str, n_calls: int, n_err: int) -> None:
    SLOT_STATUS.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model_key": model_key,
        "slot": slot,
        "calls": n_calls,
        "errors": n_err,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(SLOT_STATUS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# ─────────────────────────────────────────────────────────────
# 일정
# ─────────────────────────────────────────────────────────────

def slot_hours_for(model: ModelSpec) -> tuple:
    """앵커는 하루 4슬롯이다(설계서 7.4절)."""
    return ANCHOR_SLOT_HOURS_UTC if model.tier == "flagship" else SLOT_HOURS_UTC


def planned_slots(model: ModelSpec) -> int:
    return len(slot_hours_for(model))


# 더 돌릴 것이 없어 정상으로 끝났을 때의 종료 코드. 전 모델 완주, 완주와
# 예산 정지의 조합, 중단일 도달이 여기 해당한다(2026-09-23). 다시 띄워도
# 같은 판정으로 곧장 끝나므로 재시작할 이유가 없다. systemd가 이 코드만 재시작에서
# 빼서, 21일이 끝난 뒤 러너가 30초마다 다시 떠 스트리밍 점검 8콜을 쏘는
# 헛돌이를 막는다. 0을 쓰지 않는 것은 --max-slots로 멈춘 경우나 중단 요청과
# 구별해야 하기 때문이다. 그 둘은 사람이 부른 것이라 다시 떠도 된다.
EXIT_COMPLETE = 10


def slot_starts(from_utc: datetime):
    """from_utc 이후의 슬롯 시작 시각을 차례로 준다."""
    day = from_utc.date()
    while True:
        for hour in SLOT_HOURS_UTC:
            start = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
            if start >= from_utc:
                yield start
        day += timedelta(days=1)


def slot_label(start: datetime) -> str:
    return start.strftime("%Y-%m-%dT%H")


def wait_until(target: datetime, heartbeat_sec: int = 1800) -> None:
    """목표 시각까지 잔다.

    남은 시간을 한 번에 자지 않는다. 서버가 절전이나 일시정지에서 깨면
    시계가 튀므로 미리 계산한 값을 믿을 수 없다. 30분마다 생존 신호를
    남기는 것은 로그가 조용할 때 멈춘 것인지 자는 것인지 가르기 위해서다.
    """
    last_beat = time.monotonic()
    announced = False
    while True:
        now = datetime.now(timezone.utc)
        remain = (target - now).total_seconds()
        if remain <= 0:
            return
        if not announced:
            log(f"다음 슬롯 {slot_label(target)} 까지 {int(remain // 60)}분 대기")
            announced = True
        elif time.monotonic() - last_beat >= heartbeat_sec:
            last_beat = time.monotonic()
            log(f"대기 중 — {slot_label(target)} 까지 {int(remain // 60)}분")
        time.sleep(max(1, min(60, remain)))


# ─────────────────────────────────────────────────────────────
# 콜 조립
# ─────────────────────────────────────────────────────────────

def build_specs(models, start: datetime, start_day: date, groups,
                stream_ok: dict, done: set[str], k: int) -> list[CallSpec]:
    """한 슬롯에서 쏠 콜 전부.

    지연 프로브를 앞에 둔다. 품질 프로브가 슬롯당 300콜을 동시에 밀어넣으므로
    그 뒤에 재면 우리가 만든 부하가 지연에 섞인다. 슬롯 시작 시각에 먼저
    재면 콜마다 같은 조건에서 잰 값이 되고, 슬롯 간 비교가 성립한다.
    """
    idx = slot_index(start.date(), start.hour, start_day)
    group = item_group(idx)
    items = groups[group]
    label = slot_label(start)
    specs: list[CallSpec] = []

    for model in models:
        cond = condition_label(model.region, start.hour, start.date())
        common = dict(phase="main", slot=label, slot_index=idx,
                      item_group=group, condition=cond)

        for rep in range(LATENCY_REPS_PER_SLOT):
            cs = CallSpec(model=model, item=None, mode="latency", probe="latency",
                          temperature=LATENCY_TEMPERATURE,
                          max_tokens=LATENCY_MAX_TOKENS, rep=rep,
                          stream=stream_ok.get(model.key, False), **common)
            if cs.call_key() not in done:
                specs.append(cs)

        for item in items:
            for rep in range(k):
                cs = CallSpec(model=model, item=item, mode="direct", probe="quality",
                              temperature=CONSISTENCY_TEMPERATURE,
                              max_tokens=model.direct_max_tokens, rep=rep,
                              **common)
                if cs.call_key() not in done:
                    specs.append(cs)

    return specs


def probe_streaming(models) -> dict[str, bool]:
    """모델마다 스트리밍이 되는지 한 번 확인한다.

    되는 모델과 안 되는 모델이 섞여도 실험은 성립한다. 지연은 제공사별로
    자기 시간축 안에서만 보므로(설계서 3.3절) 한 모델의 TTFT가 비어도 다른
    모델의 곡선은 온전하다. 다만 실행 중에 방식이 바뀌면 그 모델의 곡선에
    계단이 생기므로, 시작 때 한 번 정하고 21일 내내 유지한다.
    """
    from prompts import build_latency_messages, make_nonce

    out: dict[str, bool] = {}
    for m in models:
        adapter = build_adapter(m)
        ok = False
        last = "확인하지 못했다"
        for attempt in ("usage", "plain"):
            if attempt == "plain":
                if not hasattr(adapter, "stream_usage") or not adapter.stream_usage:
                    break
                adapter.stream_usage = False
            raw = adapter.chat_stream(
                build_latency_messages(make_nonce()),
                temperature=LATENCY_TEMPERATURE, max_tokens=LATENCY_MAX_TOKENS)
            if raw.error is None and raw.ttft_ms is not None:
                ok = True
                suffix = "" if attempt == "usage" else " (stream_options 없이)"
                log(f"  {m.key:28s} 스트리밍 가능{suffix} — TTFT {raw.ttft_ms:.0f}ms")
                break
            last = raw.error or "TTFT 없음"
        if not ok:
            log(f"  {m.key:28s} 스트리밍 불가 — {last}. 비스트리밍으로 돈다(TTFT 없음)")
        out[m.key] = ok
    return out


# ─────────────────────────────────────────────────────────────
# 날짜 완주 기록
# ─────────────────────────────────────────────────────────────

def logged_days(ledger: DayLedger) -> set[tuple[str, str]]:
    """이미 판정을 적어 둔 (모델, 날짜). 재시작에서 중복으로 적지 않는다."""
    out: set[tuple[str, str]] = set()
    if not ledger.path.exists():
        return out
    for line in ledger.path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.add((row["model_key"], row["day"]))
    return out


def close_day(day: date, models, fired: set, gate: dict, ledger: DayLedger,
              already: set, skip=frozenset()) -> None:
    """하루가 끝났다. 모델마다 완주 여부를 적는다.

    skip은 이미 완주일을 채워 멈춘 모델이다. 그 모델에 날마다 not_started를
    적으면 기록이 실제로 멈춘 이유를 가린다.
    """
    key_day = day.isoformat()
    for m in models:
        if (m.key, key_day) in already or m.key in skip:
            continue
        planned = planned_slots(m)
        done = sum(1 for (mk, slot) in fired
                   if mk == m.key and slot.startswith(key_day))
        ok, reason = gate.get(m.key, (True, ""))
        if not ok:
            status, note = "not_started", reason
        elif done >= planned:
            status, note = "complete", ""
        elif done == 0:
            status, note = "not_started", "슬롯 기록 없음"
        else:
            status, note = "aborted", f"슬롯 {done}/{planned}"
        ledger.write(m.key, key_day, planned, done, status, note)
        already.add((m.key, key_day))
        if status != "complete":
            log(f"  {key_day} {m.key:28s} {status} — {note}")


def complete_day_counts(ledger: DayLedger, models) -> dict[str, int]:
    counts = {m.key: 0 for m in models}
    for row in ledger.complete_days():
        if row["model_key"] in counts:
            counts[row["model_key"]] += 1
    return counts


def finished_models(counts: dict[str, int], days: int) -> set[str]:
    """목표 완주일을 채운 모델. 사전등록 3.4절에 따라 그 모델만 멈춘다.

    전 모델이 채울 때까지 모두 돌리면, 하루를 잃은 모델을 기다리는 동안
    나머지가 22일, 23일을 쌓는다. 21일은 묶음마다 각 시각을 정확히 7회,
    요일마다 한 번씩 통과시키는 값이라(설계서 7.4절) 그 위로 붙는 날은
    이 균형을 깨고 돈만 쓴다.
    """
    return {k for k, n in counts.items() if n >= days}


def run_end_reason(models, finished: set, stopped, day: date) -> str | None:
    """실행 전체를 끝낼 이유. 없으면 None.

    사전등록 3.4절의 세 조건 가운데 완주와 예산은 모델 단위이고, 날짜는
    전체에 한 번에 걸린다. 모든 모델이 완주했거나 예산으로 멈췄으면 더
    돌릴 것이 없다.
    """
    if day >= RUN_STOP_DAY:
        return f"중단일 {RUN_STOP_DAY}에 닿았다"
    left = [m.key for m in models if m.key not in finished and m.key not in stopped]
    if not left:
        return "모든 모델이 완주일을 채웠거나 예산으로 멈췄다"
    return None


def describe_balances(models, plan: dict, balances: dict) -> list[str]:
    """계정별 잔액과 21일 투영을 나란히 놓는다. 시작 로그와 dry-run이 쓴다."""
    lines = []
    by_acct: dict[str, list] = {}
    for m in models:
        by_acct.setdefault(m.api_key_env, []).append(m)
    for acct, ms in sorted(by_acct.items()):
        entries = [plan.get("models", {}).get(m.key) or {} for m in ms]
        proj = sum(e.get("projected_total", 0.0) for e in entries)
        reserve = sum(e.get("day_reserve", 0.0) for e in entries)
        names = ", ".join(m.key for m in ms)
        bal = balances.get(acct)
        if bal is None:
            lines.append(f"  {acct:22s} 잔액 미설정 (후불로 보고 모델별 퓨즈만 건다) "
                         f"| 투영 ${proj:.2f} | {names}")
            continue
        usable = bal * BALANCE_USABLE_FRACTION
        days = int(usable // reserve) if reserve else 0
        flag = "" if usable >= proj else "  <- 21일 투영에 못 미친다"
        lines.append(f"  {acct:22s} 잔액 ${bal:.2f} 가용 ${usable:.2f} | 투영 ${proj:.2f} "
                     f"| 하루 예약 ${reserve:.2f}로 {days}일 | {names}{flag}")
    return lines


# ─────────────────────────────────────────────────────────────
# dry-run
# ─────────────────────────────────────────────────────────────

def dry_run(models, groups, start_day: date, plan: dict, k: int,
            balances: dict | None = None) -> None:
    print(f"\n실행 첫날 {start_day} (UTC) 기준 일정\n")
    print(f"  은행 {sum(len(g) for g in groups)}문항 → 묶음 {BANK_CYCLE}개 "
          f"× {len(groups[0])}문항")
    print(f"  슬롯 {', '.join(f'{h:02d}' for h in SLOT_HOURS_UTC)} UTC "
          f"| 앵커 {', '.join(f'{h:02d}' for h in ANCHOR_SLOT_HOURS_UTC)} UTC")
    print(f"  반복 k={k} | 목표 완주일 {MAIN_DESIGN['days']}\n")

    print("  첫 사흘의 묶음·조건 배정 (묶음이 여덟 시각을 한 번씩 통과하는지 확인)\n")
    header = "     날짜        " + " ".join(f"{h:02d}시" for h in SLOT_HOURS_UTC)
    print(header)
    for d in range(3):
        day = start_day + timedelta(days=d)
        cells = []
        for h in SLOT_HOURS_UTC:
            cells.append(f" G{item_group(slot_index(day, h, start_day))} ")
        print(f"     {day}  " + " ".join(cells))

    print("\n  조건 라벨 (첫 주, 지역별)\n")
    for region in ("us", "cn", "kr"):
        line = []
        for h in SLOT_HOURS_UTC:
            n = sum(1 for d in range(7)
                    if condition_label(region, h, start_day + timedelta(days=d)) == "peak")
            line.append(f"{n:>3d}/7")
        print(f"     {region}  " + " ".join(line))
    print("\n     칸은 그 시각 슬롯이 7일 중 며칠 피크로 잡히는지다. "
          "미국만 평일 한정이라 5/7이 나온다.")

    print("\n  콜 수와 비용\n")
    total_q = total_l = 0.0
    grand = 0.0
    for m in models:
        n_slots = planned_slots(m) * MAIN_DESIGN["days"]
        q = n_slots * MAIN_DESIGN["items_per_slot"] * k
        l = n_slots * LATENCY_REPS_PER_SLOT
        total_q += q
        total_l += l
        entry = plan.get("models", {}).get(m.key)
        if entry:
            per = entry["cost_per_call"]
            # 지연 프로브는 입력이 짧고 출력이 상한에 묶여 있다. 품질 콜의
            # 실측 토큰으로 환산하면 과대추정이라 상한으로 직접 계산한다.
            lat = (60 / 1e6 * m.price_in + LATENCY_MAX_TOKENS / 1e6 * m.price_out) \
                * m.tax_multiplier
            cost = q * per + l * lat
            grand += cost
            print(f"  {m.key:28s} 품질 {q:7,}  지연 {l:6,}  ${cost:8.2f}")
        else:
            print(f"  {m.key:28s} 품질 {q:7,}  지연 {l:6,}  (투영 없음)")
    print(f"  {'합계':28s} 품질 {int(total_q):7,}  지연 {int(total_l):6,}  ${grand:8.2f}")
    print(f"\n  budget.py 투영 총액 ${plan.get('projected_grand_total', 0):,.2f} "
          f"— 지연 프로브는 그 투영에 없던 몫이다.")

    print("\n  망 기준선 호스트 (슬롯마다 호스트당 "
          f"{netbase.NET_REPS_PER_SLOT}회, DNS·TCP·TLS)\n")
    for host, ms in netbase.hosts_for(models).items():
        print(f"  {host:45s} {', '.join(m.key for m in ms)}")

    print("\n  계정별 선불 잔액 (provider_balance.json)\n")
    if balances is None:
        print(f"  {PROVIDER_BALANCE.name}이 없다. 본실행은 이 파일 없이 시작하지 않는다.")
    else:
        for line in describe_balances(models, plan, balances):
            print(line)
        print(f"\n     가용은 잔액의 {BALANCE_USABLE_FRACTION:.0%}다. 하루 예약이 가용을 넘는 "
              "날부터 그 모델은 시작하지 않는다.")


# ─────────────────────────────────────────────────────────────
# 본체
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="본실험 실행")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-anchors", action="store_true", help="티어 대조군 제외")
    ap.add_argument("--k", type=int, default=MAIN_DESIGN["k"])
    ap.add_argument("--days", type=int, default=MAIN_DESIGN["days"],
                    help="목표 완주일. 이만큼 채우면 멈춘다")
    ap.add_argument("--max-slots", type=int, default=None,
                    help="슬롯 이만큼만 돌고 멈춘다. 리허설용")
    ap.add_argument("--log", default=None)
    ap.add_argument("--plan", default=None, help="예산 계획 JSON")
    ap.add_argument("--vantage", default="kr-seoul", help="측정 지점 라벨")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dst", action="store_true",
                    help="서머타임 종료일을 넘겨 실행한다. 미국 조건 라벨이 밀린다")
    args = ap.parse_args()

    models = apply_capabilities(
        get_models(args.models, include_anchors=not args.no_anchors))
    if not models:
        sys.exit("돌릴 모델이 없다. .env에 API 키가 있는지 확인할 것.")
    missing = [m.key for m in models if not m.env_key()]
    if missing:
        sys.exit(f"API 키 없음: {missing}")

    groups = load_groups(Path(args.bank) if args.bank else None)

    plan_path = Path(args.plan) if args.plan else BUDGET_PLAN
    if not plan_path.exists():
        sys.exit(f"예산 계획이 없다: {plan_path}\n먼저 budget.py를 돌려야 한다.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    now = datetime.now(timezone.utc)
    state = load_state(now.date())
    start_day = date.fromisoformat(state["start_day"])

    try:
        balances = load_balances()
    except FileNotFoundError:
        balances = None

    if args.dry_run:
        dry_run(models, groups, start_day, plan, args.k, balances)
        return

    if balances is None:
        sys.exit(f"선불 잔액 파일이 없다: {PROVIDER_BALANCE}\n"
                 "계정마다 본실험 시작 시점의 잔액을 적는다. 후불 계정은 null.")

    # 서머타임이 끝나면 미국 피크가 UTC 13~19로 밀려 슬롯 분류가 상수가
    # 아니게 된다. 사전등록이 상수를 전제하므로 여기서 막는다.
    finish_by = start_day + timedelta(days=args.days + 4)
    if finish_by >= US_DST_END and not args.allow_dst:
        sys.exit(
            f"실행 창이 서머타임 종료일({US_DST_END})을 넘긴다.\n"
            f"  첫날 {start_day} + 목표 {args.days}일 + 여유 4일 = {finish_by}\n"
            "미국 피크가 UTC 12~18에서 13~19로 밀려 조건 라벨이 날짜별로 갈린다.\n"
            "실행 창을 앞당기거나, 라벨을 다시 계산할 준비가 됐으면 --allow-dst."
        )

    log_path = Path(args.log) if args.log else MAIN_LOG
    net_path = netbase.net_log_path(log_path)
    net_done = netbase.logged_slots(net_path)
    done = load_done_keys(log_path)
    fired = load_slot_status()
    ledger = DayLedger()
    already = logged_days(ledger)

    guard = SpendGuard(plan, models, balances)
    guard.load_spent(log_path, models)
    by_key = {m.key: m for m in models}

    log(f"본실험 시작 — run_id {state['run_id']}, 첫날 {start_day}, "
        f"측정 지점 {args.vantage}")
    log(f"모델 {len(models)}개: {', '.join(m.key for m in models)}")
    if done:
        log(f"이미 끝난 콜 {len(done):,}회를 로그에서 읽었다. 그만큼 건너뛴다.")
    log("계정별 선불 잔액")
    for line in describe_balances(models, plan, balances):
        log(line)

    log("스트리밍 점검")
    stream_ok = probe_streaming(models)

    # 지난 날 중 아직 판정이 없는 것을 먼저 적는다. 재시작이 날을 건너뛰면
    # 그 날은 루프의 날 경계 처리를 못 만나기 때문이다.
    today = datetime.now(timezone.utc).date()
    finished = finished_models(complete_day_counts(ledger, models), args.days)
    day = start_day
    while day < today:
        close_day(day, models, fired, {}, ledger, already, skip=finished)
        day += timedelta(days=1)

    counts = complete_day_counts(ledger, models)
    if any(counts.values()):
        log("완주일: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))

    logger = JsonlLogger(log_path)
    completed = False
    current_day: date | None = None
    gate: dict[str, tuple[bool, str]] = {}
    n_slots = 0

    try:
        for start in slot_starts(now - timedelta(minutes=SLOT_CATCHUP_MINUTES)):
            if args.max_slots and n_slots >= args.max_slots:
                log(f"--max-slots {args.max_slots} 에 도달했다. 멈춘다.")
                break

            # 날이 바뀌었다. 지난 날을 판정하고, 완주한 모델을 빼고, 새 날의
            # 예산을 예약한다. 완주 판정은 날 경계에서만 바뀌므로 모델이
            # 멈추는 자리도 언제나 날 경계다.
            if start.date() != current_day:
                if current_day is not None:
                    close_day(current_day, models, fired, gate, ledger, already,
                              skip=finished)
                current_day = start.date()

                counts = complete_day_counts(ledger, models)
                newly = finished_models(counts, args.days) - finished
                for key in sorted(newly):
                    log(f"  {key} 완주일 {args.days}을 채웠다. 이 모델은 여기서 멈춘다.")
                finished |= newly

                end = run_end_reason(models, finished, guard.stopped, current_day)
                if end is None:
                    gate = guard.start_day([m.key for m in models if m.key not in finished])
                    end = run_end_reason(models, finished, guard.stopped, current_day)
                if end:
                    log(f"{end}. 끝낸다.")
                    completed = True
                    break

                blocked = [k for k, (ok, _) in gate.items() if not ok]
                log(f"── {current_day} 시작. " +
                    (f"예산으로 제외: {', '.join(blocked)}" if blocked else "전 모델 진행") +
                    (f" | 완주로 멈춤: {', '.join(sorted(finished))}" if finished else ""))

            late = (datetime.now(timezone.utc) - start).total_seconds() / 60
            if late > SLOT_CATCHUP_MINUTES:
                log(f"슬롯 {slot_label(start)} 은 {int(late)}분 지났다. 건너뛴다.")
                continue
            if late < 0:
                wait_until(start)

            active = [m for m in models
                      if m.key not in finished
                      and gate.get(m.key, (True, ""))[0]
                      and m.key not in guard.stopped
                      and start.hour in slot_hours_for(m)]
            if not active:
                continue

            specs = build_specs(active, start, start_day, groups, stream_ok,
                                done, args.k)
            if not specs:
                log(f"슬롯 {slot_label(start)} 은 이미 로그에 다 있다. 건너뛴다.")
                for m in active:
                    fired.add((m.key, slot_label(start)))
                continue

            idx = slot_index(start.date(), start.hour, start_day)
            log(f"슬롯 {slot_label(start)} (#{idx}, 묶음 G{item_group(idx)}) — "
                f"모델 {len(active)}개, 콜 {len(specs):,}회")

            # 네트워크 기준선을 지연 프로브 바로 앞에 잰다. 품질 콜이 밀려들기
            # 전이라 우리 콜이 만든 혼잡이 섞이지 않는다. 기준선이 실패해도
            # 슬롯은 돈다. 완주 판정의 조건이 아니다.
            label_now = slot_label(start)
            if label_now not in net_done:
                try:
                    rows = netbase.probe_slot(active, label_now, idx, start,
                                              state["run_id"], args.vantage)
                    netbase.append_rows(net_path, rows)
                    net_done.add(label_now)
                    bad = sum(1 for r in rows if r["error"])
                    tcp = sorted(r["tcp_ms"] for r in rows if r["tcp_ms"] is not None)
                    med = f"{tcp[len(tcp) // 2]:.0f}ms" if tcp else "없음"
                    log(f"  망 기준선 {len(rows)}회, 실패 {bad}회, TCP 중앙값 {med}")
                except Exception as e:  # noqa: BLE001
                    log(f"  망 기준선 실패 — {type(e).__name__}: {e}")

            tally = {m.key: [0, 0] for m in active}

            def on_done(rec):
                model = by_key.get(rec.model_key)
                if model is None:
                    return
                guard.record(rec, model)
                t = tally.setdefault(rec.model_key, [0, 0])
                t[0] += 1
                if rec.error:
                    t[1] += 1
                else:
                    done.add(rec.call_key)

            t0 = time.monotonic()
            run_batch(specs, state["run_id"], logger, vantage=args.vantage,
                      on_done=on_done)
            took = (time.monotonic() - t0) / 60

            label = slot_label(start)
            for m in active:
                n, err = tally.get(m.key, [0, 0])
                mark_slot(m.key, label, n, err)
                fired.add((m.key, label))
                ok, reason = guard.check_mid_day(m.key)
                if not ok:
                    log(f"  긴급 정지 {m.key} — {reason}")

            total_err = sum(v[1] for v in tally.values())
            spent = sum(guard.spent.values())
            log(f"슬롯 {label} 완료 — {took:.1f}분, 오류 {total_err}회, "
                f"누적 ${spent:.2f}")
            n_slots += 1

    except KeyboardInterrupt:
        log("중단 요청을 받았다. 지금 슬롯까지 기록하고 끝낸다.")
    finally:
        # 지금 날은 판정하지 않는다. 재시작해서 남은 슬롯을 마저 채우면
        # 그날은 완주가 되는데, 여기서 aborted로 적어 두면 뒤집을 수 없다.
        # 판정은 다음 날로 넘어갈 때, 또는 다음 실행의 시작 점검에서 한다.
        logger.close()

    log("지출 요약")
    for row in guard.summary():
        mark = f" — {row['stopped']}" if row["stopped"] else ""
        log(f"  {row['model_key']:28s} ${row['spent']:7.2f} / ${row['cap']:7.2f}{mark}")

    counts = complete_day_counts(ledger, models)
    log("완주일")
    for key, n in sorted(counts.items()):
        log(f"  {key:28s} {n}일")

    if completed:
        sys.exit(EXIT_COMPLETE)


if __name__ == "__main__":
    main()
