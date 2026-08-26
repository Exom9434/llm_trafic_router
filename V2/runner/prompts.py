"""프롬프트 조립과 답 파싱.

두 프로브가 서로 다른 프롬프트를 쓴다.
  - 직답 프로브: 글자 하나만 뱉게 해서 출력 토큰을 8개 이하로 묶는다.
  - CoT 프로브: 추론 예산 삭감을 잡기 위한 소규모 서브셋용이다.
캐시 히트를 막는 nonce는 system 메시지 꼬리에 주석 형태로 붙인다.
프롬프트 본문(문항)은 건드리지 않으므로 문항 간 비교가 깨지지 않는다.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from config import OPTION_LETTERS

DIRECT_SYSTEM = (
    "You are answering multiple-choice exam questions.\n"
    "Reply with exactly one capital letter naming the correct option.\n"
    "No explanation, no punctuation, no other text."
)

COT_SYSTEM = (
    "You are a logical reasoning assistant answering multiple-choice exam questions.\n"
    "Think step by step inside <Reason></Reason>, then give the single option letter "
    "inside <Answer></Answer>. Output nothing outside these tags."
)


def make_nonce() -> str:
    """콜마다 유일한 짧은 토큰. 프롬프트 캐시 히트를 막는다."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def letters_for(item: dict) -> list[str]:
    return OPTION_LETTERS[: len(item["options"])]


def format_question(item: dict) -> str:
    lines = [item["question"].strip(), ""]
    for letter, opt in zip(letters_for(item), item["options"]):
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def build_messages(item: dict, mode: str, nonce: str | None = None) -> list[dict]:
    """mode: "direct" | "cot"."""
    if mode not in ("direct", "cot"):
        raise ValueError(f"unknown mode: {mode}")
    system = DIRECT_SYSTEM if mode == "direct" else COT_SYSTEM
    if nonce:
        system = f"{system}\n[session:{nonce}]"
    user = format_question(item)
    if mode == "direct":
        user = f"{user}\n\nAnswer with one letter only."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_ANSWER_TAG = re.compile(r"<Answer>\s*([A-J])\s*</Answer>", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"^\(?\s*([A-J])\s*\)")
# "answer is C", "Answer: D", "correct option — B" 처럼 답을 명시한 꼴만 인정한다.
_ANSWER_PHRASE = re.compile(
    r"\b(?:answer|option|choice)\b[^A-Za-z0-9]{0,12}(?:is|:|=)?[^A-Za-z0-9]{0,6}([A-J])\b",
    re.IGNORECASE,
)
# 답 글자 뒤에 올 수 있는 구분자. 공백은 뺀다 —
# "I cannot answer"의 I를 선택지 I로 오독하는 사고가 여기서 난다.
_OPTION_MARKERS = ".):],-*"

# 모델이 지시를 어기고 풀이를 쓴 뒤 마지막에 글자만 툭 놓는 꼴.
# 2026-08-26 probe에서 sonnet-5가 "... E = 2ix - 2iy + iz  D"로 끝냈다.
# 답이 분명히 거기 있는데 위 규칙들이 전부 놓쳤다. 끝에서만, 앞뒤가
# 글자·숫자가 아닐 때만 인정한다 — 본문 중간의 대문자를 주워 오면
# 조용한 오답이 되므로 위치를 끝으로 못박는다.
_TRAILING_LETTER = re.compile(r"(?:^|[^A-Za-z0-9])\(?([A-J])\)?[.\s]*$")


def parse_letter(text: str | None, valid: list[str], truncated: bool = False) -> str | None:
    """모델 출력에서 선택지 글자를 뽑는다. 못 뽑으면 None(파싱 실패).

    애매하면 추측하지 않고 None을 준다. 잘못 뽑은 답은 조용히 정확도를
    오염시키지만, 파싱 실패는 로그에 남아 집계된다.

    `truncated`는 출력이 상한에 닿아 끊겼는지다. 끊긴 응답에서는 마지막
    글자 규칙을 쓰지 않는다 — 끊긴 자리의 글자는 답이 아니다.
    """
    if not text:
        return None
    stripped = text.strip()

    m = _ANSWER_TAG.search(stripped)
    if m:
        letter = m.group(1).upper()
        return letter if letter in valid else None

    m = _PAREN_LETTER.match(stripped)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 직답 프로브의 정상 경로: 앞머리가 곧 답이다.
    head = stripped.lstrip("*_ \n\t")
    if head:
        first = head[0].upper()
        if first in valid and (len(head) == 1 or head[1] in _OPTION_MARKERS):
            return first

    m = _ANSWER_PHRASE.search(stripped)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 풀이를 늘어놓고 맨 끝에 글자만 남긴 경우. 잘리지 않은 응답에만 쓴다 —
    # 잘린 응답의 마지막 글자는 답이 아니라 문장이 끊긴 자리일 뿐이다.
    if not truncated:
        m = _TRAILING_LETTER.search(stripped)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    return None


def letter_distribution(top_logprobs: list[dict] | None, valid: list[str]) -> dict[str, float]:
    """첫 토큰의 top-logprob 목록을 선택지 글자 위의 확률분포로 접는다.

    top_logprobs 원소: {"token": "A", "logprob": -0.12}
    같은 글자로 접히는 토큰들(" A", "A")의 확률은 더한다. 마지막에 정규화한다.
    """
    import math

    if not top_logprobs:
        return {}
    acc: dict[str, float] = {}
    for entry in top_logprobs:
        tok = (entry.get("token") or "").strip().upper()
        if len(tok) != 1 or tok not in valid:
            continue
        acc[tok] = acc.get(tok, 0.0) + math.exp(entry["logprob"])
    total = sum(acc.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted(acc.items())}
