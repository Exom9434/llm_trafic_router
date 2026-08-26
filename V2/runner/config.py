"""
V2 실험 설정 — 라인업, 가격, 프로브 파라미터.

2026-08-21 전면 개정. 3월에 짠 라인업이 한 세대 통째로 밀려 있었다.
`deepseek-chat`은 7/24에 서비스가 끝났고, OpenAI의 비-reasoning GA는 사라졌고,
Gemini 3.x는 logprobs를 아예 안 준다. 개정 근거는 V2/EXPERIMENT_DESIGN.md
부록 B(2026-08 라인업 재조사)에 남겼다.

두 가지가 이 파일의 구조를 결정한다.

  1. logprob이 9개 중 2개(DeepSeek·Qwen)에서만 된다. 그래서 주력 지표가
     reasoning 토큰 수와 자기일관성으로 옮겨 갔다. `supports_logprobs`는
     이제 대부분 "no"이고, 그건 버그가 아니라 현실이다.

  2. 대부분의 현행 모델이 thinking/reasoning을 기본으로 켠다. 끄지 않으면
     max_tokens 8이 추론 토큰에 먼저 소진돼 빈 응답이 나온다. 프로바이더마다
     끄는 방법이 달라서 `extra_body`에 모델별로 박아 뒀다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

def _force_utf8_console() -> None:
    """윈도우 콘솔의 기본 코드페이지(한국어 환경은 cp949)를 UTF-8로 바꾼다.

    출력에 →·—… 같은 기호가 섞여 있어 cp949 콘솔에서 UnicodeEncodeError로
    죽을 수 있다. 2주 무인 실행 중에 출력 한 줄 때문에 러너가 멎으면
    곤란하므로 import 시점에 손봐 둔다. macOS·Linux에서는 아무 일도 없다.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_console()

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent           # llm_trafic_router/
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CAPABILITIES_FILE = ROOT / "capabilities.json"
ENV_FILE = REPO_ROOT / ".env"


def _load_env() -> None:
    """저장소 루트의 .env를 읽어 환경변수로 올린다.

    러너가 V2/runner/에 있고 .env는 저장소 루트에 있어서, 경로를 명시하지
    않으면 못 찾는다. 이미 셸에 설정된 값은 덮어쓰지 않는다.
    python-dotenv가 없으면 직접 파싱한다 — 키 몇 개 읽자고 실행이 막히면
    곤란하기 때문이다.
    """
    if not ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
        return
    except ImportError:
        pass

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()

# ─────────────────────────────────────────────────────────────
# 모델 라인업
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    key: str                     # 로그·분석에서 쓰는 고유 식별자
    provider: str
    model: str                   # API에 보내는 모델 문자열
    adapter: str                 # "openai_compat" | "anthropic" | "hyperclova"
    api_key_env: str
    base_url: str | None = None
    tier: str = "budget"         # "budget" | "flagship"
    region: str = "us"           # "us" | "cn" | "kr" — 부하 주기가 다른 단위
    supports_logprobs: str = "unknown"   # "yes" | "no" | "unknown"
    price_in: float = 0.0        # USD / 1M input tokens
    price_out: float = 0.0       # USD / 1M output tokens
    max_concurrency: int = 4

    # 출력 상한 파라미터 이름. reasoning 모델은 max_tokens를 안 받는다.
    max_tokens_param: str = "max_tokens"
    # 직답 프로브의 출력 상한. thinking을 완전히 못 끄는 모델은 더 줘야 한다.
    direct_max_tokens: int = 8
    # 실측 평균 출력 토큰(2026-08-24 diag_reasoning). 비용 추정에 쓴다.
    # 상한을 그대로 쓰면 과대평가된다 — Gemini는 상한 24인데 실측이 1이다.
    measured_output_tokens: int | None = None
    # thinking/reasoning을 끄거나 최소화하는 프로바이더별 파라미터.
    extra_body: dict = field(default_factory=dict)
    # temperature를 받는가. 2026-08 기준 claude-sonnet-5가 거부한다.
    # 못 받으면 일관성 프로브는 같은 요청을 반복해 모델 기본 샘플링에 맡긴다.
    supports_temperature: bool = True
    # 버전 고정이 가능한가. 불가능하면 fingerprint·반환모델로 사후 탐지한다.
    pinned: bool = False
    notes: str = ""

    def env_key(self) -> str | None:
        return os.getenv(self.api_key_env)


# CLOVA Studio만 KRW로 과금한다. 다른 프로바이더와 나란히 놓으려면 환산이 필요하다.
# 환율은 고정값으로 박아 둔다 — 실행 중에 환율을 조회하면 같은 로그의 비용이
# 시점마다 달라져 재현이 안 된다. 갱신하려면 이 상수만 고치면 된다.
KRW_PER_USD = 1387.0          # 2026-08-22 중간환율

# 단가는 2026-08-21 조사 기준이다. 비용 추정용이며 청구 근거가 아니다.
LINEUP: list[ModelSpec] = [
    ModelSpec(
        key="openai_gpt56_luna",
        measured_output_tokens=4,
        provider="openai",
        model="gpt-5.6-luna",
        adapter="openai_compat",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        supports_logprobs="no",
        price_in=0.20, price_out=1.20,
        max_tokens_param="max_completion_tokens",
        direct_max_tokens=64,    # probe 실측 p100 기준, 절단 0%
        extra_body={"reasoning_effort": "none"},
        pinned=False,
        notes=(
            "GA 2026-07-09. gpt-4o-mini의 사실상 후계. 날짜 스냅샷 ID가 없어 고정 불가. "
            "reasoning 기본값이 medium이라 effort=none을 반드시 보내야 한다. "
            "안 보내면 추론 토큰이 출력 상한을 먹고 빈 응답이 온다. "
            "2026-08-24 실측: Chat Completions에서 effort는 none만 받는다. "
            "minimal·low·medium 모두 Unsupported value로 거절된다. 따라서 추론을 "
            "켜서 추론 토큰을 재는 길이 막혀 있다. 필드는 있으나 항상 0이다. "
            "logprobs도 반환하지 않는다."
        ),
    ),
    ModelSpec(
        key="google_gemini_flash_lite",
        measured_output_tokens=1,
        provider="google",
        model="gemini-3.5-flash-lite",
        adapter="openai_compat",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        supports_logprobs="no",
        price_in=0.30, price_out=2.50,
        direct_max_tokens=64,    # probe 실측 p100 기준, 절단 0%
        extra_body={"reasoning_effort": "minimal"},
        pinned=False,
        notes=(
            "GA 2026-07. 3.x 세대는 logprobs를 반환하지 않는다 — Google 공식 답변 "
            "'WAI, no longer returned for 3.X models'(2026-08-05). 네이티브·OpenAI호환 둘 다 안 됨. "
            "thinking을 완전히 끌 수 없고 minimal이 최소라 출력 상한을 넉넉히 잡았다. "
            "비용을 아끼려면 gemini-3.1-flash-lite(GA, $0.25/$1.50)로 내려도 된다."
        ),
    ),
    ModelSpec(
        key="deepseek_v4_flash",
        measured_output_tokens=2689,
        region="cn",
        provider="deepseek",
        model="deepseek-v4-flash",
        adapter="openai_compat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        supports_logprobs="no",             # 2026-08-24 실측: 문서와 달리 반환하지 않는다
        price_in=0.44, price_out=1.32,      # 피크 단가. 오프피크는 정확히 반값.
        direct_max_tokens=16384,    # probe 실측 p100 기준, 절단 0%
        pinned=False,
        notes=(
            "구 deepseek-chat은 2026-07-24 서비스 종료. v4-flash가 후계다. "
            "logprobs·top_logprobs(0~20) 지원, system_fingerprint 제공 — 라인업에서 "
            "가장 정보량이 많은 프로바이더다. "
            "2026-08-16부터 피크/오프피크 이중 요금제. 피크 01:00-04:00, 06:00-10:00 UTC. "
            "이 공표 자체가 '시간대=부하' 가정의 외부 검증 재료다(설계서 3.5절). "
            "2026-08-24 실측: 추론을 끄는 파라미터가 없고 항상 켜져 있다. 같은 문항에 "
            "53~72토큰으로 콜마다 달라진다. 이 변동이 곧 우리가 재려는 신호이며, "
            "평상시 변동 폭은 보정 패스의 노이즈 바닥선이 확정한다."
        ),
    ),
    ModelSpec(
        key="qwen_flash",
        measured_output_tokens=2040,
        region="cn",
        provider="qwen",
        model="qwen3.7-flash-2026-07-15",
        adapter="openai_compat",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        supports_logprobs="unknown",        # 2026-08-24 실측: logprobs 요청이 거절된다. 사유 확인 필요
        price_in=0.03, price_out=0.13,
        direct_max_tokens=8192,    # probe 실측 p100 기준, 절단 0%
        extra_body={"enable_thinking": True},
        pinned=True,
        notes=(
            "qwen3.5-flash는 legacy로 밀렸다. 3.7-flash에 날짜 스냅샷이 있어 "
            "라인업에서 유일하게 진짜 버전 고정이 된다. "
            "2026-08-24 실측: thinking을 끄면 평균속도 문제를 틀리고(A), 켜면 맞힌다(B). "
            "켠 상태에서 추론 430~457토큰을 쓴다. "
            "추론이 정답률에 직결된다는 증거이며, 우리 가설의 인과 사슬을 이 모델이 실증한다. "
            "그래서 켠 채로 운용한다. 추론 477토큰을 써도 단가가 낮아 콜당 $0.00007에 그친다. "
            "단가는 서드파티 출처라 콘솔에서 재확인할 것."
        ),
    ),
    ModelSpec(
        key="upstage_solar_pro3",
        measured_output_tokens=185,
        region="kr",
        provider="upstage",
        model="solar-pro3",
        adapter="openai_compat",
        api_key_env="UPSTAGE_API_KEY",
        base_url="https://api.upstage.ai/v1",
        supports_logprobs="no",
        price_in=0.15, price_out=0.60,
        direct_max_tokens=1024,    # probe 실측 p100 기준, 절단 0%
        notes=(
            "solar-pro4에서 되돌렸고 실측으로 확정했다. pro4는 출력 상한 512토큰을 "
            "추론으로 전부 소진하고도 답을 내지 못했다(추론 512 = 출력 512). "
            "pro3는 출력 3토큰·추론 0으로 정답을 내며 콜당 비용이 pro4의 1/21이다. "
            "추론 토큰 필드는 있으나 값이 0이다. 켜는 파라미터를 찾으면 측정 대상이 될 수 있다."
        ),
    ),
    ModelSpec(
        key="anthropic_haiku",
        measured_output_tokens=131,
        direct_max_tokens=1024,    # probe 실측 p100 기준, 절단 0%
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        supports_logprobs="no",
        price_in=1.00, price_out=5.00,
        pinned=True,
        notes=(
            "Haiku 5는 없다. 3월 라인업에서 유일하게 그대로 살아남은 모델이다. "
            "thinking 기본 OFF라 별도 차단이 필요 없다 — 라인업에서 가장 깨끗한 케이스. "
            "은퇴는 2026-10-15 이전 없음."
        ),
    ),
    ModelSpec(
        key="naver_hcx_dash",
        measured_output_tokens=2,
        direct_max_tokens=64,    # probe 실측 p100 기준, 절단 0%
        region="kr",
        provider="naver",
        model="HCX-DASH-002",
        adapter="hyperclova",
        api_key_env="CLOVASTUDIO_API_KEY",
        base_url="https://clovastudio.stream.ntruss.com",
        supports_logprobs="no",
        # 1,000토큰당 입력 0.25원 / 출력 1원 (2026-08-22 콘솔 확인).
        # 1M 토큰 환산: 입력 250원, 출력 1,000원.
        price_in=250.0 / KRW_PER_USD,
        price_out=1000.0 / KRW_PER_USD,
        max_concurrency=2,
        pinned=True,
        notes=(
            "여전히 현행 최신 경량 모델. DASH-003은 없다. "
            "temperature 0을 허용한다(0.00~1.00) — 3월에 걱정하던 문제는 없었다. "
            "요금은 1,000토큰당 입력 0.25원·출력 1원이며 부가세 별도다. "
            "위 단가는 부가세 전이므로 실제 청구액은 약 10% 더 나온다. "
            "다른 프로바이더와 비교 가능하게 두려고 부가세를 빼고 적었다."
        ),
    ),
]

# 티어 대조군(flagship 앵커). 설계서 5.3절.
ANCHORS: list[ModelSpec] = [
    ModelSpec(
        key="openai_gpt56_sol",
        measured_output_tokens=4,
        provider="openai",
        model="gpt-5.6-sol",
        adapter="openai_compat",
        api_key_env="OPENAI_API_KEY",
        tier="flagship",
        supports_logprobs="no",
        price_in=5.00, price_out=30.00,
        max_tokens_param="max_completion_tokens",
        direct_max_tokens=64,    # probe 실측 p100 기준, 절단 0%
        extra_body={"reasoning_effort": "none"},
        notes="티어 대조군 A. GA 2026-07-09. 중간 티어 gpt-5.6-terra($2/$12)도 대안.",
    ),
    ModelSpec(
        key="anthropic_sonnet5",
        measured_output_tokens=26,
        direct_max_tokens=1024,    # probe 실측 p100 기준, 절단 0%
        provider="anthropic",
        model="claude-sonnet-5",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        tier="flagship",
        supports_logprobs="no",
        price_in=2.00, price_out=10.00,
        extra_body={"thinking": {"type": "disabled"}},
        supports_temperature=False,
        pinned=True,
        notes=(
            "티어 대조군 B. GA 2026-06-30. 구 claude-sonnet-4-5-20250929는 은퇴 하한이 "
            "2026-09-29라 실험 기간과 겹칠 위험이 있었다. "
            "adaptive thinking이 기본 ON이라 disabled를 명시해야 한다. "
            "2026-08-24 실측: temperature를 보내면 400을 낸다"
            "(\'temperature is deprecated for this model\'). 라인업에서 유일하다."
        ),
    ),
]

ALL_MODELS: list[ModelSpec] = LINEUP + ANCHORS


def get_models(keys: list[str] | None = None, include_anchors: bool = True,
               region: str | None = None) -> list[ModelSpec]:
    """키 목록으로 모델을 고른다. keys가 None이면 키가 설정된 것만 준다.

    region을 주면 그 지역 프로바이더만 남긴다. 보정 패스를 지역별로 나눠
    돌리기 위한 필터다.
    """
    pool = ALL_MODELS if include_anchors else LINEUP
    if region:
        pool = [m for m in pool if m.region == region]
    if keys is None:
        return [m for m in pool if m.env_key()]
    by_key = {m.key: m for m in ALL_MODELS}
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise KeyError(f"알 수 없는 모델 키: {missing}")
    chosen = [by_key[k] for k in keys]
    if region:
        chosen = [m for m in chosen if m.region == region]
    return chosen


def apply_capabilities(models: list[ModelSpec]) -> list[ModelSpec]:
    """smoke_test.py가 남긴 실측 결과로 supports_logprobs를 덮어쓴다."""
    if not CAPABILITIES_FILE.exists():
        return models
    caps = json.loads(CAPABILITIES_FILE.read_text(encoding="utf-8"))
    for m in models:
        entry = caps.get(m.key)
        if entry and entry.get("supports_logprobs") in ("yes", "no"):
            m.supports_logprobs = entry["supports_logprobs"]
    return models


def spec_dict(m: ModelSpec) -> dict:
    return asdict(m)


# ─────────────────────────────────────────────────────────────
# 프로브 파라미터
# ─────────────────────────────────────────────────────────────

# 직답 프로브의 기본 출력 상한. 모델별로 direct_max_tokens가 덮어쓴다.
DIRECT_MAX_TOKENS = 8
DIRECT_TEMPERATURE = 0.0

# 일관성 프로브: 같은 문항을 k회 샘플링한다.
# logprob을 대부분 잃은 뒤 이쪽이 주력 지표가 됐다(설계서 3.4절).
CONSISTENCY_TEMPERATURE = 1.0
CONSISTENCY_K = 5

# logprob 요청 시 받아올 상위 토큰 수. DeepSeek·Qwen에서만 쓰인다.
TOP_LOGPROBS = 10

OPTION_LETTERS = list("ABCDEFGHIJ")

# DeepSeek이 공표한 피크 구간(UTC). 시간대=부하 가정의 외부 검증에 쓴다.
DEEPSEEK_PEAK_WINDOWS_UTC = [(1, 4), (6, 10)]

# 보정 패스를 돌려야 할 시각 (KST, 시작시 ~ 종료시).
#
# 9개 모델이 동시에 한가한 시각은 없다. 프로바이더의 주 사용자층이 어느
# 표준시에 사는지에 따라 부하 주기가 갈리기 때문이다. 미국 업무시간은
# KST 밤이고, 중국·한국 업무시간은 KST 낮이다.
#
# 노이즈 바닥선을 그 모델의 피크 시간에 재면 바닥선 자체가 부하를 먹은
# 값이 된다. 본실험에서 고부하와 비교할 때 대비가 줄어 검정력을 깎는다.
# 그래서 보정 패스는 지역별로 나눠 돌린다(설계서 6절).
CALIBRATION_WINDOWS_KST = {
    "us": (10, 17),    # 미국 업무시간(KST 22~06시)을 피한다
    "cn": (23, 7),     # DeepSeek 공표 피크 KST 10~13·15~19시를 피한다
    "kr": (23, 7),     # 한국 업무시간을 피한다
}

REGION_LABELS = {"us": "미국", "cn": "중국", "kr": "한국"}


def in_window(hour_kst: int, window: tuple[int, int]) -> bool:
    """자정을 넘어가는 구간(예: 23~7시)도 처리한다."""
    start, end = window
    if start <= end:
        return start <= hour_kst < end
    return hour_kst >= start or hour_kst < end


KST = timezone(timedelta(hours=9))


def _fmt_span(sec: int) -> str:
    h, m = sec // 3600, sec % 3600 // 60
    if h and m:
        return f"{h}시간 {m}분"
    return f"{h}시간" if h else f"{m}분"


def _seconds_to_next_start(now: datetime, window: tuple[int, int]) -> int:
    """지금이 열려 있든 아니든, 다음 시간대 시작까지 남은 초."""
    target = now.replace(hour=window[0], minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(0, int((target - now).total_seconds()))


def seconds_until_window(now: datetime, window: tuple[int, int]) -> int:
    """저부하 시간대가 열릴 때까지 남은 초. 이미 열려 있으면 0."""
    if in_window(now.hour, window):
        return 0
    return _seconds_to_next_start(now, window)


def seconds_left_in_window(now: datetime, window: tuple[int, int]) -> int:
    """저부하 시간대가 닫히기까지 남은 초. 닫혀 있으면 0."""
    if not in_window(now.hour, window):
        return 0
    target = now.replace(hour=window[1] % 24, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(0, int((target - now).total_seconds()))


def wait_until_window(
    window: tuple[int, int],
    label: str = "",
    min_remaining_sec: int = 0,
    poll_sec: int = 60,
    heartbeat_sec: int = 1800,
    log=print,
) -> None:
    """저부하 시간대가 열릴 때까지 기다린다. 이미 열려 있으면 즉시 반환한다.

    `min_remaining_sec`을 주면 "열려 있지만 곧 닫히는" 경우도 기다린다.
    닫히기까지 남은 시간이 그보다 짧으면 다음 회차를 노린다. 시간대 끝을
    넘겨 실행하면 노이즈 바닥선이 부하를 먹어 본실험의 대비가 줄기 때문이다.

    남은 시간을 한 번에 자지 않고 poll_sec 간격으로 현재 시각을 다시 본다.
    랩탑이 절전에서 깨면 시계가 튀므로 미리 계산한 값을 믿을 수 없다.
    대기 중에는 heartbeat_sec마다 생존 신호를 남긴다 — 로그가 조용하면
    멈춘 것인지 자는 것인지 구분할 수 없다.

    본실험 스케줄러(로드맵 3단계)가 슬롯 대기에 그대로 쓸 수 있도록
    calibrate.py가 아니라 여기에 둔다.
    """
    tag = f"{label} " if label else ""
    waiting_since = None
    last_beat = None

    while True:
        now = datetime.now(KST)
        left = seconds_left_in_window(now, window)

        if left > 0 and left >= min_remaining_sec:
            if waiting_since is not None:
                waited = int((now - waiting_since).total_seconds())
                log(f"[대기 종료] {now:%m-%d %H:%M} KST — {tag}저부하 시간대가 열렸다"
                    f"({_fmt_span(waited)} 기다렸다). 시작한다.")
            return

        if left > 0:
            reason = (f"지금 열려 있으나 닫히기까지 {_fmt_span(left)}뿐이라"
                      f" {_fmt_span(min_remaining_sec)}짜리 작업을 시작하지 않는다")
        else:
            reason = "아직 열리지 않았다"
        remain = _seconds_to_next_start(now, window)

        if waiting_since is None:
            waiting_since = now
            last_beat = now
            log(f"[대기 시작] {now:%m-%d %H:%M} KST — {tag}저부하 시간대는 "
                f"KST {window[0]:02d}~{window[1]:02d}시다. {reason}.")
            log(f"             다음 시작까지 약 {_fmt_span(remain)}. "
                f"{poll_sec}초마다 시각을 다시 본다. Ctrl+C로 중단한다.")
        elif (now - last_beat).total_seconds() >= heartbeat_sec:
            last_beat = now
            log(f"[대기 중] {now:%m-%d %H:%M} KST — {tag}다음 시작까지 약 {_fmt_span(remain)}")

        time.sleep(max(1, min(poll_sec, remain)))


# ─────────────────────────────────────────────────────────────
# 본실험 규모 (설계서 7.4절)
# ─────────────────────────────────────────────────────────────

# 라인업 7개의 일정. 슬롯은 DeepSeek 공표 피크 경계를 걸치도록 배치한다.
MAIN_DESIGN = {
    "bank": 300,
    "items_per_slot": 100,     # 은행을 3슬롯에 한 바퀴 훑는다
    "slots_per_day": 8,
    "days": 14,
    "k": 5,                    # 자기일관성용 반복
}

# 앵커는 티어 효과의 유무만 판정하면 되므로 슬롯을 줄인다(설계서 7.4절).
ANCHOR_DESIGN = {**MAIN_DESIGN, "slots_per_day": 2}

# 모델별 반복 수 배분(설계서 7.5절). 문항 은행 300은 모든 모델에 그대로 두고
# 비용은 k로만 조절한다 — tau는 반복이 아니라 문항 수로만 줄기 때문에,
# 문항을 깎아 반복을 사는 것은 방향이 반대다.
#
# DeepSeek만 줄이는 이유는 출력 토큰이 실측 평균 2,689로 라인업 나머지의
# 20~2,700배이기 때문이다. k=5면 그 모델 하나가 $207로 전체의 60%를 먹는다.
# k=2면 $83이면서 문항 300이 유지되므로 추론량·정확도 검정력은 온전하고,
# 잃는 것은 자기일관성의 방문당 정밀도뿐이다. DeepSeek은 추론 토큰을
# 보고하는 두 모델 중 하나라 더 예민한 지표를 이미 갖고 있다.
K_BY_MODEL = {
    "deepseek_v4_flash": 2,
}


def k_for_main(model_key: str) -> int:
    """본실험에서 이 모델에 쓸 반복 수."""
    return K_BY_MODEL.get(model_key, MAIN_DESIGN["k"])

# 지출 상한은 투영치의 몇 배로 잡을 것인가.
# 퓨즈이지 조절기가 아니다. 정상 운영에서 터지면 안 되는 위치여야 한다.
SPEND_CAP_MULTIPLIER = 3.0

# 하루를 시작하기 전에 예약할 예산의 여유율.
# 투영이 조금 빗나가도 하루를 중간에 끊지 않기 위한 완충이다.
DAY_RESERVE_MARGIN = 1.25

# 재시도
MAX_RETRIES = 4
RETRY_BASE_SLEEP = 2.0
REQUEST_TIMEOUT = 60.0
