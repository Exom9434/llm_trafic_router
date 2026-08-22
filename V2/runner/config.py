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
            "logprobs 미지원(공식 명문화는 확인 불가, smoke_test로 실측할 것)."
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
        provider="deepseek",
        model="deepseek-v4-flash",
        adapter="openai_compat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        supports_logprobs="yes",
        price_in=0.44, price_out=1.32,      # 피크 단가. 오프피크는 정확히 반값.
        pinned=False,
        notes=(
            "구 deepseek-chat은 2026-07-24 서비스 종료. v4-flash가 후계다. "
            "logprobs·top_logprobs(0~20) 지원, system_fingerprint 제공 — 라인업에서 "
            "가장 정보량이 많은 프로바이더다. "
            "2026-08-16부터 피크/오프피크 이중 요금제. 피크 01:00-04:00, 06:00-10:00 UTC. "
            "이 공표 자체가 '시간대=부하' 가정의 외부 검증 재료다(설계서 3.5절)."
        ),
    ),
    ModelSpec(
        key="qwen_flash",
        provider="qwen",
        model="qwen3.7-flash-2026-07-15",
        adapter="openai_compat",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        supports_logprobs="yes",
        price_in=0.03, price_out=0.13,
        extra_body={"enable_thinking": False},
        pinned=True,
        notes=(
            "qwen3.5-flash는 legacy로 밀렸다. 3.7-flash에 날짜 스냅샷이 있어 "
            "라인업에서 유일하게 진짜 버전 고정이 된다. "
            "thinking 기본 ON이라 enable_thinking=false 필수 — 긴 CoT가 단일 최대 비용 요인이다. "
            "단가는 서드파티 출처라 콘솔에서 재확인할 것."
        ),
    ),
    ModelSpec(
        key="upstage_solar_pro4",
        provider="upstage",
        model="solar-pro4",
        adapter="openai_compat",
        api_key_env="UPSTAGE_API_KEY",
        base_url="https://api.upstage.ai/v1",
        supports_logprobs="no",
        price_in=0.30, price_out=1.20,
        direct_max_tokens=16,
        extra_body={"reasoning_effort": "low"},
        pinned=False,
        notes=(
            "GA 2026-08-10. solar-pro3($0.15/$0.60)도 아직 살아 있어 비용이 문제면 대안. "
            "logprobs는 지원 파라미터 목록에 없어 미지원으로 판단했으나 단정은 불가 — smoke_test로 확정할 것."
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
        pinned=True,
        notes=(
            "티어 대조군 B. GA 2026-06-30. 구 claude-sonnet-4-5-20250929는 은퇴 하한이 "
            "2026-09-29라 실험 기간과 겹칠 위험이 있었다. "
            "adaptive thinking이 기본 ON이라 disabled를 명시해야 한다."
        ),
    ),
]

ALL_MODELS: list[ModelSpec] = LINEUP + ANCHORS


def get_models(keys: list[str] | None = None, include_anchors: bool = True) -> list[ModelSpec]:
    """키 목록으로 모델을 고른다. keys가 None이면 키가 설정된 것만 준다."""
    pool = ALL_MODELS if include_anchors else LINEUP
    if keys is None:
        return [m for m in pool if m.env_key()]
    by_key = {m.key: m for m in ALL_MODELS}
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise KeyError(f"알 수 없는 모델 키: {missing}")
    return [by_key[k] for k in keys]


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
