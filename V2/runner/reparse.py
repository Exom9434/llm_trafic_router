"""저장된 raw_text로 답을 다시 읽어 로그를 현행 파서에 맞춘다.

파서를 고치면 이미 쌓인 로그의 `parsed_letter`와 `correct`는 옛 규칙이 쓴
값으로 남는다. `select_bank.py`는 그 필드를 그대로 읽으므로, 고친 파서와
문항 은행의 근거가 어긋난 채로 확정된다. 콜을 다시 쏠 필요는 없다.
`raw_text`가 로그에 다 들어 있기 때문이다.

2026-09-01에 답 명시 꼴을 첫 매치에서 마지막 매치로 바꾸고 절단 응답의
본문 탐색을 막았다. 그 수정이 이 스크립트가 만들어진 이유다.

선택지 수는 문항마다 다르므로 후보 풀에서 문항을 찾아 유효 글자를 정한다.
풀에 없는 문항은 A~J로 둔다. 열 글자로 뭉뚱그리면 선택지가 넷인 문항에서
E 이후 글자를 답으로 인정하게 된다.

기본은 미리보기다. 무엇이 어떻게 바뀌는지만 출력하고 파일은 건드리지 않는다.
`--write`를 줘야 실제로 쓰며, 그때 원본을 `.bak`으로 남긴다.

실행:
    uv run reparse.py                 # 미리보기
    uv run reparse.py --write         # 적용
    uv run reparse.py --log outputs/probe_caps_haiku.jsonl --write
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import prompts
from config import OPTION_LETTERS, OUTPUT_DIR
from itembank import load_pool

DEFAULT_LOG = OUTPUT_DIR / "calibration_calls.jsonl"
REPORT = OUTPUT_DIR / "reparse_report.md"
FALLBACK_LETTERS = list(OPTION_LETTERS[:10])


def letters_by_item(pool_path: Path | None) -> dict[str, list[str]]:
    out = {}
    for it in load_pool(pool_path):
        if it.get("item_id"):
            out[it["item_id"]] = prompts.letters_for(it)
    return out


def is_truncated(rec: dict) -> bool:
    """core.py와 같은 규칙. 상한에서 2토큰 안쪽이면 끊긴 것으로 본다."""
    out = rec.get("output_tokens")
    cap = rec.get("max_tokens")
    return bool(out is not None and cap and out >= cap - 2)


def reparse_record(rec: dict, letters: dict[str, list[str]]) -> tuple[str | None, int | None]:
    valid = letters.get(rec.get("item_id"), FALLBACK_LETTERS)
    letter = prompts.parse_letter(rec.get("raw_text"), valid, truncated=is_truncated(rec))
    gold = rec.get("gold_letter")
    correct = None if letter is None or gold is None else int(letter == gold)
    return letter, correct


def main() -> None:
    ap = argparse.ArgumentParser(description="raw_text로 답을 다시 읽는다")
    ap.add_argument("--log", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--write", action="store_true", help="실제로 파일을 고친다")
    ap.add_argument("--samples", type=int, default=4, help="종류마다 보여 줄 예시 수")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else DEFAULT_LOG
    if not log_path.exists():
        raise SystemExit(f"로그가 없다: {log_path}")
    letters = letters_by_item(Path(args.pool) if args.pool else None)

    kinds = ("답이 바뀜", "읽던 답을 버림", "못 읽던 답을 읽음")
    counts: dict[str, Counter] = defaultdict(Counter)
    acc_before: dict[str, list[int]] = defaultdict(list)
    acc_after: dict[str, list[int]] = defaultdict(list)
    samples: dict[str, list[str]] = defaultdict(list)
    total = changed = 0

    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    out_fh = tmp_path.open("w", encoding="utf-8", newline="\n") if args.write else None

    try:
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                if rec.get("error") is None and rec.get("raw_text") is not None:
                    old_letter = rec.get("parsed_letter")
                    new_letter, new_correct = reparse_record(rec, letters)
                    if new_letter != old_letter:
                        changed += 1
                        key = rec.get("model_key") or "?"
                        kind = ("답이 바뀜" if old_letter and new_letter else
                                "읽던 답을 버림" if old_letter else "못 읽던 답을 읽음")
                        counts[key][kind] += 1
                        gold = rec.get("gold_letter")
                        if gold:
                            acc_before[kind].append(int(old_letter == gold) if old_letter else 0)
                            acc_after[kind].append(int(new_letter == gold) if new_letter else 0)
                        if len(samples[kind]) < args.samples:
                            raw = " ".join((rec.get("raw_text") or "").split())
                            samples[kind].append(
                                f"{rec.get('item_id')} / {key} / "
                                f"{old_letter} -> {new_letter} (정답 {gold}) :: {raw[-110:]}")
                        rec["parsed_letter"] = new_letter
                        rec["correct"] = new_correct
                        rec["reparsed"] = True
                if out_fh:
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out_fh:
            out_fh.close()

    lines = [
        "# 재파싱 결과",
        "",
        f"- 로그: `{log_path.name}` ({total:,}행)",
        f"- 값이 바뀌는 콜: {changed:,}건 ({changed / total * 100:.2f}%)",
        f"- 시각: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 모델별",
        "",
        "| 모델 | " + " | ".join(kinds) + " | 합계 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in sorted(counts):
        c = counts[key]
        lines.append(f"| {key} | " + " | ".join(str(c[k]) for k in kinds) +
                     f" | {sum(c.values())} |")
    lines += ["", "## 정답률 변화", "",
              "10지선다이므로 우연 수준은 0.100이다. 버린 답의 정답률이 우연에 가까우면",
              "그 값들은 답이 아니라 잡음이었다는 뜻이다.", "",
              "| 변화 | 건수 | 이전 | 이후 |", "|---|---:|---:|---:|"]
    for kind in kinds:
        b, a = acc_before[kind], acc_after[kind]
        if not b:
            continue
        lines.append(f"| {kind} | {len(b)} | {statistics.mean(b):.3f} | {statistics.mean(a):.3f} |")
    for kind in kinds:
        if samples[kind]:
            lines += ["", f"## 예시 — {kind}", ""] + [f"- `{s}`" for s in samples[kind]]

    report = "\n".join(lines) + "\n"
    print(report)

    if not args.write:
        print("미리보기다. 파일은 그대로다. 적용하려면 --write를 줄 것.")
        return

    backup = log_path.with_suffix(log_path.suffix + ".bak")
    shutil.copy2(log_path, backup)
    os.replace(tmp_path, log_path)
    REPORT.write_text(report, encoding="utf-8")
    print(f"적용했다. 원본은 {backup.name}, 보고서는 {REPORT.name}.")
    print("바뀐 레코드에는 reparsed=true가 붙는다. 나중에 무엇이 손대졌는지 추적할 수 있다.")


if __name__ == "__main__":
    main()
