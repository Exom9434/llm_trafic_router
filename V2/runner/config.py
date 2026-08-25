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
from dataclasses import dataclass, field, asdict
from pathlib import Path

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
        provider="openai",
        model="gpt-5.6-luna",
        adapter="openai_compat",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        supports_logprobs="no",
        price_in=0.20, price_out=1.20,
        max_tokens_param="max_completion_tokens",
        direct_max_tokens=16,
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
        provider="google",
        model="gemini-3.5-flash-lite",
        adapter="openai_compat",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        supports_logprobs="no",
        price_in=0.30, price_out=2.50,
        direct_max_tokens=24,
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
        region="cn",
        provider="deepseek",
        model="deepseek-v4-flash",
        adapter="openai_compat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        supports_logprobs="no",             # 2026-08-24 실측: 문서와 달리 반환하지 않는다
        price_in=0.44, price_out=1.32,      # 피크 단가. 오프피크는 정확히 반값.
        direct_max_tokens=128,              # 실측: 추론에 53~56토큰을 쓴다. 여유를 둔다
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
        region="cn",
        provider="qwen",
        model="qwen3.7-flash-2026-07-15",
        adapter="openai_compat",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        supports_logprobs="unknown",        # 2026-08-24 실측: logprobs 요청이 거절된다. 사유 확인 필요
        price_in=0.03, price_out=0.13,
        direct_max_tokens=768,              # 실측: 추론에 477토큰을 쓴다
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
        region="kr",
        provider="upstage",
        model="solar-pro3",
        adapter="openai_compat",
        api_key_env="UPSTAGE_API_KEY",
        base_url="https://api.upstage.ai/v1",
        supports_logprobs="no",
        price_in=0.15, price_out=0.60,
        direct_max_tokens=16,
        notes=(
            "solar-pro4에서 되돌렸고 실측으로 확정했다. pro4는 출력 상한 512토큰을 "
            "추론으로 전부 소진하고도 답을 내지 못했다(추론 512 = 출력 512). "
            "pro3는 출력 3토큰·추론 0으로 정답을 내며 콜당 비용이 pro4의 1/21이다. "
            "추론 토큰 필드는 있으나 값이 0이다. 켜는 파라미터를 찾으면 측정 대상이 될 수 있다."
        ),
    ),
    ModelSpec(
        key="anthropic_haiku",
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
        provider="openai",
        model="gpt-5.6-sol",
        adapter="openai_compat",
        api_key_env="OPENAI_API_KEY",
        tier="flagship",
        supports_logprobs="no",
        price_in=5.00, price_out=30.00,
        max_tokens_param="max_completion_tokens",
        direct_max_tokens=16,
        extra_body={"reasoning_effort": "none"},
        notes="티어 대조군 A. GA 2026-07-09. 중간 티어 gpt-5.6-terra($2/$12)도 대안.",
    ),
    ModelSpec(
        key="anthropic_sonnet5",
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
