"""지출 가드와 사전 비용 투영.

설계 의도가 하나 있다. **절단은 막을 수 없지만, 절단이 떨어지는 자리는 고를 수 있다.**

상한 없이 돌리면 잔액 소진이나 결제 실패로 아무 때나 멈춘다. 십중팔구 하루
중간이고, 그러면 그날은 앞쪽 슬롯만 남는다. 절단이 시간대와 상관되어 버려서
잡음이 아니라 편향이 된다. 리뷰어가 정확히 여기를 찌른다.

그래서 이 모듈은 세 가지를 한다.

  1. 사전 투영 — 보정 패스의 **실측** 토큰 수로 본실험 총비용을 미리 계산한다.
     9일차에 발견할 일을 0일차에 발견하는 것이 진짜 해법이다.
  2. 하루 단위 예약 — 하루치 예산이 남지 않으면 그날을 아예 시작하지 않는다.
     절단이 항상 날 경계에 떨어진다.
  3. 완료 여부 기록 — day_status.jsonl에 하루의 완주 여부를 남긴다. 분석은
     완료된 날만 쓴다. 사후에 데이터를 보고 자르는 것이 아니라 사전에 정한
     규칙이므로 방어가 된다.

상한 자체는 퓨즈이지 조절기가 아니다. 투영치의 3배로 걸어 두면 정상 운영에서는
터지지 않는다. 터졌다면 추론 차단 실패나 재시도 폭주 같은 고장이며, 그 상태로
모으던 데이터는 어차피 정상이 아니다.

실행:
    python budget.py                      # 보정 패스 실측 → 본실험 비용 투영
    python budget.py --design anchor      # 앵커 축소 일정으로 투영
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

from calllog import read_records
from config import (
    ANCHOR_DESIGN,
    BALANCE_USABLE_FRACTION,
    KRW_PER_USD,
    DAY_RESERVE_MARGIN,
    DATA_DIR,
    MAIN_DESIGN,
    OUTPUT_DIR,
    SPEND_CAP_MULTIPLIER,
    ModelSpec,
    apply_capabilities,
    get_models,
)

CALIBRATION_LOG = OUTPUT_DIR / "calibration_calls.jsonl"
ITEM_BANK = DATA_DIR / "item_bank.json"
BUDGET_PLAN = OUTPUT_DIR / "budget_plan.json"
DAY_STATUS_LOG = OUTPUT_DIR / "day_status.jsonl"

# 계정(API 키)별 선불 잔액. 코드와 함께 서버로 올라가야 하므로 outputs가
# 아니라 러너 디렉터리에 둔다(push.sh가 outputs를 제외한다).
PROVIDER_BALANCE = Path(__file__).resolve().parent / "provider_balance.json"


# ─────────────────────────────────────────────────────────────
# 1. 실측 토큰 프로파일
# ─────────────────────────────────────────────────────────────

@dataclass
class TokenProfile:
    """보정 패스에서 실제로 관측된 콜당 토큰 수."""
    model_key: str
    n_calls: int
    mean_input: float
    mean_output: float          # 추론 토큰을 포함한 청구 대상 출력
    mean_reasoning: float
    error_rate: float

    def cost_per_call(self, m: ModelSpec, price_factor: float = 1.0) -> float:
        gross = (self.mean_input / 1e6 * m.price_in
                 + self.mean_output / 1e6 * m.price_out)
        return gross * price_factor * m.tax_multiplier


def is_peak(model: ModelSpec, hour_utc: int) -> bool:
    return any(start <= hour_utc < end for start, end in model.peak_hours_utc)


def schedule_price_factor(model: ModelSpec, slot_hours) -> float:
    """일정이 피크와 오프피크에 어떻게 걸치는지로 평균 단가 계수를 낸다.

    DeepSeek만 시각에 따라 단가가 갈린다(피크 UTC 01~04, 06~10, 오프피크는
    반값). 전부 피크로 계산하면 그 모델을 45% 과대평가하고, 그 과대평가가
    지출 상한과 하루 예약에 그대로 실린다. 상한이 느슨해지는 것보다 하루
    예약이 과해져 멀쩡한 날을 시작 거부하는 쪽이 실제 위험이다(8.1절).

    슬롯 배치는 피크 경계를 걸치라는 설계가 정한 것이고(3.5절), 여기서는
    그 배치가 만든 요금을 계산할 뿐이다. 요금이 배치를 정하지 않는다.
    """
    if not model.peak_hours_utc or not slot_hours:
        return 1.0
    weights = [1.0 if is_peak(model, h) else model.offpeak_multiplier
               for h in slot_hours]
    return sum(weights) / len(weights)


def call_price_factor(model: ModelSpec, ts_utc: str | None) -> float:
    """실제 콜 하나의 단가 계수. 누적 지출 계산에 쓴다.

    시각을 모르면 피크로 본다. 모르는 쪽을 비싸게 잡아야 가드가 안전하다.
    """
    if not model.peak_hours_utc:
        return 1.0
    if not ts_utc:
        return 1.0
    try:
        hour = int(str(ts_utc)[11:13])
    except (ValueError, IndexError):
        return 1.0
    return 1.0 if is_peak(model, hour) else model.offpeak_multiplier


def bank_item_ids(path: Path | None = None) -> set[str] | None:
    """확정된 문항 은행의 id 집합. 아직 없으면 None."""
    p = path or ITEM_BANK
    if not p.exists():
        return None
    try:
        items = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ids = {i.get("item_id") for i in items if i.get("item_id")}
    return ids or None


def measure_token_profiles(log_path: Path | None = None,
                           restrict_to_bank: bool = True) -> dict[str, TokenProfile]:
    """보정 패스 로그에서 모델별 실측 토큰 프로파일을 뽑는다.

    추정이 아니라 실측이라는 점이 핵심이다. 추론 차단이 안 먹었다면
    mean_reasoning이 0이 아니게 나오고, 투영 비용이 그만큼 뛴다.

    은행이 확정돼 있으면 그 문항들만 센다. 본실험이 도는 것은 은행 문항이고
    후보 풀에는 밴드 밖 문항이 절반 가까이 섞여 있다. 출력량이 난이도에
    크게 좌우되므로 풀 전체로 재면 은행의 실제 비용과 어긋난다.
    """
    records = read_records(log_path or CALIBRATION_LOG)
    bank = bank_item_ids() if restrict_to_bank else None
    if bank:
        in_bank = [r for r in records if r.get("item_id") in bank]
        if in_bank:
            records = in_bank
    buckets: dict[str, list[dict]] = {}
    for rec in records:
        key = rec.get("model_key")
        if key:
            buckets.setdefault(key, []).append(rec)

    profiles: dict[str, TokenProfile] = {}
    for key, recs in buckets.items():
        ok = [r for r in recs if r.get("error") is None]
        if not ok:
            continue
        n = len(ok)

        def mean(field: str) -> float:
            vals = [r.get(field) or 0 for r in ok]
            return sum(vals) / n if n else 0.0

        profiles[key] = TokenProfile(
            model_key=key,
            n_calls=len(recs),
            mean_input=round(mean("input_tokens"), 1),
            mean_output=round(mean("output_tokens"), 1),
            mean_reasoning=round(mean("reasoning_tokens"), 1),
            error_rate=round((len(recs) - n) / len(recs), 4),
        )
    return profiles


# ─────────────────────────────────────────────────────────────
# 2. 본실험 비용 투영
# ─────────────────────────────────────────────────────────────

def calls_per_day(design: dict) -> int:
    return design["items_per_slot"] * design["slots_per_day"] * design["k"]


def total_calls(design: dict) -> int:
    return calls_per_day(design) * design["days"]


def project(models: list[ModelSpec], profiles: dict[str, TokenProfile],
            design_for) -> dict:
    """모델별 본실험 비용 투영. design_for(model)이 그 모델의 일정을 준다."""
    out = {"models": {}, "missing_profile": [], "unpriced": []}
    grand = 0.0

    for m in models:
        prof = profiles.get(m.key)
        if prof is None:
            out["missing_profile"].append(m.key)
            continue
        if m.price_in == 0 and m.price_out == 0:
            out["unpriced"].append(m.key)

        design = design_for(m)
        factor = schedule_price_factor(m, design.get("slot_hours"))
        per_call = prof.cost_per_call(m, factor)
        day_cost = per_call * calls_per_day(design)
        run_cost = per_call * total_calls(design)
        grand += run_cost

        out["models"][m.key] = {
            "tier": m.tier,
            "slots_per_day": design["slots_per_day"],
            "days": design["days"],
            "calls_per_day": calls_per_day(design),
            "total_calls": total_calls(design),
            "measured_input_tokens": prof.mean_input,
            "measured_output_tokens": prof.mean_output,
            "measured_reasoning_tokens": prof.mean_reasoning,
            "price_factor": round(factor, 4),
            "tax_multiplier": m.tax_multiplier,
            "cost_per_call": round(per_call, 8),
            "projected_day_cost": round(day_cost, 4),
            "projected_total": round(run_cost, 2),
            "spend_cap": round(run_cost * SPEND_CAP_MULTIPLIER, 2),
            "day_reserve": round(day_cost * DAY_RESERVE_MARGIN, 4),
        }

    out["projected_grand_total"] = round(grand, 2)
    out["cap_multiplier"] = SPEND_CAP_MULTIPLIER
    return out


# ─────────────────────────────────────────────────────────────
# 3. 지출 가드
# ─────────────────────────────────────────────────────────────

def load_balances(path: Path | None = None) -> dict[str, float | None]:
    """provider_balance.json을 읽어 계정별 선불 잔액(USD)을 준다.

    값은 본실험 시작 시점의 잔액이다. 가드는 본실험 로그의 지출만 더하므로
    리허설이나 보정에 쓴 돈은 이미 빠진 금액을 적어야 한다. null은 선불
    잔액이 없는 계정(후불)이며 모델별 퓨즈만 걸린다. 원화는 {"krw": 5000}
    으로 적으면 고정 환율로 바꾼다. 부가세가 붙는 계정은 부가세 포함
    금액을 적는다. 가드의 지출 추정이 부가세를 곱하기 때문이다.
    """
    p = Path(path or PROVIDER_BALANCE)
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, float | None] = {}
    for acct, v in (raw.get("balances") or {}).items():
        if v is None:
            out[acct] = None
        elif isinstance(v, dict) and "krw" in v:
            out[acct] = float(v["krw"]) / KRW_PER_USD
        elif isinstance(v, dict) and "usd" in v:
            out[acct] = float(v["usd"])
        else:
            out[acct] = float(v)
    return out


class SpendGuard:
    """모델별로 독립적으로 예산을 지킨다.

    프로바이더 내 종단 설계라(설계서 3.1절) 한 모델이 멈춰도 다른 모델의
    결과는 온전하다. 그래서 정지는 언제나 모델 단위다.
    """

    def __init__(self, plan: dict, models: list[ModelSpec] | None = None,
                 balances: dict[str, float | None] | None = None):
        self.caps: dict[str, float] = {}
        self.reserves: dict[str, float] = {}
        for key, entry in plan.get("models", {}).items():
            self.caps[key] = entry["spend_cap"]
            self.reserves[key] = entry["day_reserve"]
        self.spent: dict[str, float] = {k: 0.0 for k in self.caps}
        self.stopped: dict[str, str] = {}          # model_key → 정지 사유
        self._lock = threading.Lock()

        # 계정 단위 선불 잔액 (2026-09-23).
        #
        # 모델별 상한(투영의 3배)은 고장을 끊는 퓨즈라 정상 운영에서는 걸리지
        # 않는다. 실제로 실험을 끊는 것은 제공사 선불 잔액이고, 그건 날
        # 경계가 아니라 하루 중간에 바닥난다. 가드가 잔액을 알아야 절단을
        # 날 경계로 끌어올 수 있다. 같은 키를 쓰는 모델(haiku와 sonnet5,
        # luna와 sol)은 잔액 하나를 나눠 쓰므로 계정 단위로 센다.
        self.account: dict[str, str] = {}
        self.priority: dict[str, int] = {}
        for m in models or []:
            self.account[m.key] = m.api_key_env
            # 잔액이 한 모델 몫밖에 안 남으면 주 측정군을 먼저 돌린다.
            self.priority[m.key] = 1 if m.tier == "flagship" else 0
        self.balances: dict[str, float] = {
            a: v for a, v in (balances or {}).items() if v is not None}

    def _account_spent(self, account: str) -> float:
        return sum(v for k, v in self.spent.items() if self.account.get(k) == account)

    def set_balances(self, balances: dict[str, float | None]) -> None:
        """잔액 파일을 다시 읽었을 때 바꿔 끼운다. 이미 멈춘 모델은 되살리지 않는다."""
        with self._lock:
            self.balances = {a: v for a, v in balances.items() if v is not None}

    def days_left(self) -> dict[str, float]:
        """계정마다 가용 잔액이 하루 예약 몇 번 치를 더 버티는가."""
        out = {}
        with self._lock:
            for acct, bal in self.balances.items():
                keys = [k for k, a in self.account.items()
                        if a == acct and k not in self.stopped]
                need = sum(self.reserves.get(k, 0.0) for k in keys)
                if need:
                    left = bal * BALANCE_USABLE_FRACTION - self._account_spent(acct)
                    out[acct] = left / need
        return out

    def usable_balance(self, account: str) -> float | None:
        bal = self.balances.get(account)
        return None if bal is None else bal * BALANCE_USABLE_FRACTION

    # ── 누적 ──
    def load_spent(self, log_path: Path, models: list[ModelSpec]) -> None:
        """기존 로그에서 이미 쓴 비용을 복원한다. 재개해도 예산이 이어진다."""
        by_key = {m.key: m for m in models}
        for rec in read_records(log_path):
            key = rec.get("model_key")
            if key not in by_key:
                continue
            m = by_key[key]
            factor = call_price_factor(m, rec.get("ts_utc")) * m.tax_multiplier
            self.spent[key] = self.spent.get(key, 0.0) + factor * (
                (rec.get("input_tokens") or 0) / 1e6 * m.price_in
                + (rec.get("output_tokens") or 0) / 1e6 * m.price_out
            )

    def record(self, rec, model: ModelSpec) -> None:
        factor = call_price_factor(model, getattr(rec, "ts_utc", None)) * model.tax_multiplier
        cost = factor * ((rec.input_tokens or 0) / 1e6 * model.price_in
                         + (rec.output_tokens or 0) / 1e6 * model.price_out)
        with self._lock:
            self.spent[model.key] = self.spent.get(model.key, 0.0) + cost

    # ── 판정 ──
    def can_start_day(self, model_key: str) -> tuple[bool, str]:
        """하루를 시작해도 되는가.

        하루치 예산이 남지 않으면 시작하지 않는다. 하루를 중간에 끊는 것보다
        아예 시작하지 않는 편이 낫다. 절단이 날 경계에 떨어져야 그 이전
        데이터가 온전히 쓰인다.
        """
        with self._lock:
            if model_key in self.stopped:
                return False, self.stopped[model_key]
            cap = self.caps.get(model_key)
            if cap is None:
                return True, "상한 미설정"
            spent = self.spent.get(model_key, 0.0)
            reserve = self.reserves.get(model_key, 0.0)
            if spent + reserve > cap:
                reason = (f"하루치 예산 부족 (사용 ${spent:.2f} + 예약 ${reserve:.2f} "
                          f"> 상한 ${cap:.2f})")
                self.stopped[model_key] = reason
                return False, reason
            return True, f"잔여 ${cap - spent:.2f}"

    def start_day(self, model_keys) -> dict[str, tuple[bool, str]]:
        """그날 돌릴 모델들을 한꺼번에 판정한다.

        모델별 상한을 먼저 보고, 계정에 선불 잔액이 적혀 있으면 그 계정의
        누적 지출과 그날 이미 예약한 몫에 이 모델의 하루 예약을 더해 가용
        잔액을 넘는지 본다. 넘으면 그 모델은 그날부터 멈춘다. 잔액이
        다시 늘 일이 없으므로 정지는 되돌리지 않는다. 한 계정의 모델끼리는
        주 측정군이 대조군보다 먼저 예약한다.
        """
        out: dict[str, tuple[bool, str]] = {}
        reserved: dict[str, float] = {}
        for key in sorted(model_keys, key=lambda k: (self.priority.get(k, 0), k)):
            ok, reason = self.can_start_day(key)
            acct = self.account.get(key)
            usable = self.usable_balance(acct) if acct else None
            if not ok or usable is None:
                out[key] = (ok, reason)
                continue
            with self._lock:
                used = self._account_spent(acct) + reserved.get(acct, 0.0)
                need = self.reserves.get(key, 0.0)
                if used + need > usable:
                    why = (f"계정 잔액 부족 ({acct}: 사용·예약 ${used:.2f} + "
                           f"예약 ${need:.2f} > 가용 ${usable:.2f})")
                    self.stopped[key] = why
                    out[key] = (False, why)
                    continue
                reserved[acct] = reserved.get(acct, 0.0) + need
            out[key] = (True, f"{reason}, 계정 가용 ${usable - used - need:.2f}")
        return out

    def check_mid_day(self, model_key: str) -> tuple[bool, str]:
        """하루 도중의 비상 점검. 투영이 크게 빗나갔을 때만 걸린다.

        여기서 걸리면 그날은 완주하지 못하므로 aborted로 기록되고 분석에서
        빠진다. 정상 운영에서 이 경로가 실행되면 안 된다 — 실행됐다면
        투영이 틀렸다는 신호이므로 로그를 확인해야 한다.
        """
        with self._lock:
            cap = self.caps.get(model_key)
            if cap is None:
                return True, ""
            spent = self.spent.get(model_key, 0.0)
            if spent > cap:
                reason = f"상한 초과로 긴급 정지 (${spent:.2f} > ${cap:.2f})"
                self.stopped[model_key] = reason
                return False, reason
            return True, ""

    def summary(self) -> list[dict]:
        with self._lock:
            return [{
                "model_key": k,
                "spent": round(self.spent.get(k, 0.0), 4),
                "cap": round(cap, 2),
                "remaining": round(cap - self.spent.get(k, 0.0), 4),
                "stopped": self.stopped.get(k),
            } for k, cap in sorted(self.caps.items())]


# ─────────────────────────────────────────────────────────────
# 4. 날짜 완주 기록
# ─────────────────────────────────────────────────────────────

class DayLedger:
    """모델별로 하루가 완주됐는지 기록한다.

    분석은 status == "complete"인 날만 쓴다. 이 규칙은 데이터를 보기 전에
    정해 둔 것이며, 그래서 사후 선택이 아니다.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or DAY_STATUS_LOG)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, model_key: str, day: str, planned_slots: int,
              completed_slots: int, status: str, note: str = "") -> None:
        row = {
            "model_key": model_key,
            "day": day,
            "planned_slots": planned_slots,
            "completed_slots": completed_slots,
            "status": status,          # "complete" | "aborted" | "not_started"
            "note": note,
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    def complete_days(self, model_key: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") != "complete":
                    continue
                if model_key and row.get("model_key") != model_key:
                    continue
                rows.append(row)
        return rows


# ─────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="보정 패스 실측 → 본실험 비용 투영")
    ap.add_argument("--log", default=None, help="보정 패스 로그 (기본: outputs/calibration_calls.jsonl)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else CALIBRATION_LOG
    if not log_path.exists():
        raise SystemExit(
            f"보정 패스 로그가 없다: {log_path}\n"
            "먼저 calibrate.py를 돌려야 실측 토큰으로 투영할 수 있다."
        )

    models = apply_capabilities(get_models(args.models, include_anchors=not args.no_anchors))
    profiles = measure_token_profiles(log_path)

    def design_for(m: ModelSpec) -> dict:
        return ANCHOR_DESIGN if m.tier == "flagship" else MAIN_DESIGN

    plan = project(models, profiles, design_for)

    out = Path(args.out) if args.out else BUDGET_PLAN
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"보정 패스 실측 기준 본실험 투영 (상한 = 투영치 × {SPEND_CAP_MULTIPLIER})\n")
    width = max((len(k) for k in plan["models"]), default=10)
    print(f"  {'모델':<{width}}  {'출력토큰':>8} {'추론토큰':>8} {'콜 수':>9} "
          f"{'투영':>9} {'상한':>9} {'하루예약':>9}")
    for key, e in plan["models"].items():
        print(f"  {key:<{width}}  {e['measured_output_tokens']:>8.1f} "
              f"{e['measured_reasoning_tokens']:>8.1f} {e['total_calls']:>9,} "
              f"${e['projected_total']:>8,.2f} ${e['spend_cap']:>8,.2f} "
              f"${e['day_reserve']:>8.2f}")
    print(f"\n  투영 총액: ${plan['projected_grand_total']:,.2f}")

    # 추론을 일부러 켠 모델은 경고 대상이 아니다. 그 모델들에서 추론 토큰이
    # 0이 아닌 것은 주력 지표가 살아 있다는 뜻이다(설계서 3.4절).
    on_purpose = {m.key for m in models if m.reasoning_on_purpose}
    intended = [k for k, e in plan["models"].items()
                if e["measured_reasoning_tokens"] > 0 and k in on_purpose]
    leaking = [k for k, e in plan["models"].items()
               if e["measured_reasoning_tokens"] > 0 and k not in on_purpose]
    if intended:
        print(f"\n  추론을 켜고 운용하는 모델 — {', '.join(intended)}")
        print("        추론 토큰이 0이 아닌 것이 정상이다. 이 값이 주력 지표다.")
    if leaking:
        print(f"\n  경고: 추론 토큰이 0이 아닌 모델 — {', '.join(leaking)}")
        print("        config.py의 extra_body가 안 먹었을 수 있다. 비용이 크게 뛴다.")
    if plan["missing_profile"]:
        print(f"\n  보정 패스 기록이 없어 투영 불가: {', '.join(plan['missing_profile'])}")
    if plan["unpriced"]:
        print(f"  단가 미확인이라 0으로 잡힌 모델: {', '.join(plan['unpriced'])} "
              f"— 총액이 과소평가된다.")

    print(f"\n계획 저장: {out}")


if __name__ == "__main__":
    main()
