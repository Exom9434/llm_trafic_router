"""API 키 설정 상태 점검.

저장소 루트의 .env를 읽어, 라인업의 각 모델이 실행 가능한지 한눈에 본다.
키 값은 출력하지 않는다. 앞뒤 몇 글자만 보여 준다.

실행:
    python check_env.py
"""

from __future__ import annotations

import sys

from config import ALL_MODELS, ENV_FILE, LINEUP


def mask(value: str) -> str:
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:6]}…{value[-4:]} ({len(value)}자)"


def main() -> None:
    print(f".env 경로: {ENV_FILE}")
    print(f"존재 여부: {'있음' if ENV_FILE.exists() else '없음 — 만들어야 한다'}")
    print()

    ready, missing = [], []
    width = max(len(m.key) for m in ALL_MODELS)

    for m in ALL_MODELS:
        tier = "앵커" if m.tier == "flagship" else "라인업"
        value = m.env_key()
        if value:
            ready.append(m)
            print(f"  O {m.key:<{width}}  {tier:4s}  {m.api_key_env:<22s} {mask(value)}")
        else:
            missing.append(m)
            print(f"  X {m.key:<{width}}  {tier:4s}  {m.api_key_env:<22s} 없음")

    print()
    print(f"실행 가능: {len(ready)}/{len(ALL_MODELS)}개")

    if missing:
        print()
        print(".env에 추가할 줄:")
        for m in sorted({m.api_key_env: m for m in missing}.values(), key=lambda x: x.api_key_env):
            print(f"  {m.api_key_env}=")
        print()
        print("키가 없는 모델은 --models로 빼고 부분 실행할 수 있다.")
        sys.exit(1)

    print("전부 설정됐다. 다음: python smoke_test.py")


if __name__ == "__main__":
    main()
