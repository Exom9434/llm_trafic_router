"""로그에서 특정 모델·문항의 콜을 걷어낸다. 재보정 전 준비 작업이다.

재개 키가 `phase|slot|모델|문항|모드|temp|rep`뿐이라 출력 상한이 들어가지
않는다. 상한을 올리고 그 모델을 다시 돌리면 러너는 옛 상한으로 성공한 콜을
완료로 보고 건너뛴다. 다시 재고 싶은 행을 먼저 빼야 하는 이유다.

**걷어낸 행은 버리지 않고 `outputs/pruned_calls.jsonl`에 옮겨 붙인다.**
1차 보정 패스를 폐기하게 만든 절단 편향의 증거가 그 행들이고, 논문의 근거
자료다. 로그에서 뺀다는 것은 재수집 대상으로 되돌린다는 뜻이지 없던 일로
만든다는 뜻이 아니다.

기본은 미리보기다. 무엇이 빠지는지만 출력하고 파일은 건드리지 않는다.

**러너가 도는 중에는 쓰지 말 것.** 임시 파일을 만들어 원본 자리에 밀어
넣으므로, 로그를 append로 잡고 있는 프로세스가 있으면 그쪽이 쓴 행이
사라진다. 배치가 끝난 뒤에 돌린다.

실행:
    uv run prune_calls.py --models anthropic_haiku --items-file data/haiku_redo_items.json
    uv run prune_calls.py --models anthropic_haiku --items-file data/haiku_redo_items.json --write
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import OUTPUT_DIR

DEFAULT_LOG = OUTPUT_DIR / "calibration_calls.jsonl"
PRUNED = OUTPUT_DIR / "pruned_calls.jsonl"


def load_ids(path: Path | None, inline: list[str] | None) -> set[str]:
    ids: set[str] = set(inline or [])
    if path:
        ids |= set(json.loads(path.read_text(encoding="utf-8")))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="로그에서 특정 모델·문항의 콜을 걷어낸다")
    ap.add_argument("--log", default=None)
    ap.add_argument("--models", nargs="+", required=True, help="대상 모델 키")
    ap.add_argument("--items-file", default=None, help="문항 id 목록 JSON")
    ap.add_argument("--items", nargs="*", default=None, help="문항 id 직접 나열")
    ap.add_argument("--all-items", action="store_true",
                    help="문항을 가리지 않고 그 모델의 모든 콜을 뺀다")
    ap.add_argument("--write", action="store_true", help="실제로 파일을 고친다")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else DEFAULT_LOG
    if not log_path.exists():
        raise SystemExit(f"로그가 없다: {log_path}")

    items = load_ids(Path(args.items_file) if args.items_file else None, args.items)
    if not items and not args.all_items:
        raise SystemExit("뺄 문항이 없다. --items-file, --items, --all-items 중 하나를 줄 것.")
    models = set(args.models)

    kept = dropped = 0
    by_model = Counter()
    by_reason = Counter()
    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    out_fh = tmp_path.open("w", encoding="utf-8", newline="\n") if args.write else None
    pruned_fh = PRUNED.open("a", encoding="utf-8", newline="\n") if args.write else None
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                hit = (rec.get("model_key") in models
                       and (args.all_items or rec.get("item_id") in items))
                if hit:
                    dropped += 1
                    by_model[rec.get("model_key")] += 1
                    out = rec.get("output_tokens")
                    cap = rec.get("max_tokens") or 0
                    if rec.get("error"):
                        by_reason["오류"] += 1
                    elif out is not None and cap and out >= cap - 2:
                        by_reason["상한 소진"] += 1
                    elif rec.get("parsed_letter") is None:
                        by_reason["파싱 실패"] += 1
                    else:
                        by_reason["정상"] += 1
                    if pruned_fh:
                        rec["pruned_at"] = stamp
                        rec["pruned_from"] = log_path.name
                        pruned_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                kept += 1
                if out_fh:
                    out_fh.write(line + "\n")
    finally:
        if out_fh:
            out_fh.close()
        if pruned_fh:
            pruned_fh.close()

    total = kept + dropped
    print(f"로그: {log_path.name} ({total:,}행)")
    print(f"대상 모델: {', '.join(sorted(models))}")
    print(f"대상 문항: {'전부' if args.all_items else f'{len(items)}개'}")
    print(f"\n뺄 행 {dropped:,}건, 남을 행 {kept:,}건")
    for k, v in by_model.most_common():
        print(f"  {k:26} {v:>6,}")
    print("\n빠지는 행의 성격")
    for k, v in by_reason.most_common():
        print(f"  {k:12} {v:>6,}")

    if not args.write:
        if out_fh is None and tmp_path.exists():
            tmp_path.unlink()
        print("\n미리보기다. 파일은 그대로다. 적용하려면 --write를 줄 것.")
        return

    backup = log_path.with_suffix(log_path.suffix + ".bak")
    shutil.copy2(log_path, backup)
    os.replace(tmp_path, log_path)
    print(f"\n적용했다. 원본은 {backup.name}, 걷어낸 행은 {PRUNED.name}에 옮겨 붙였다.")
    print("이제 그 모델을 다시 돌리면 빠진 콜만 새 상한으로 재수집된다.")


if __name__ == "__main__":
    main()
