"""원본 이미지 그룹을 보존하며 train/validation/test를 다시 나눈다."""
import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def source_id(path: Path) -> str:
    # Roboflow가 붙인 증강 이미지 식별자 제거
    return path.name.rsplit(".rf.", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data_clean"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; remove it explicitly to rebuild")

    groups = defaultdict(list)
    for path in args.source.glob("*/*/*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            groups[(path.parent.name, source_id(path))].append(path)

    rng = random.Random(args.seed)
    by_class = defaultdict(list)
    for (label, sid), paths in groups.items():
        by_class[label].append((sid, paths))

    totals = defaultdict(int)
    for label, items in sorted(by_class.items()):
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * args.test_ratio))
        n_val = max(1, round(n * args.val_ratio))
        boundaries = {"test": items[:n_test], "validation": items[n_test:n_test+n_val], "train": items[n_test+n_val:]}
        for split, split_items in boundaries.items():
            for _, paths in split_items:
                for src in paths:
                    dst = args.output / split / label / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                    totals[(split, label)] += 1

    for (split, label), count in sorted(totals.items()):
        print(f"{split:10s} {label:32s} {count:4d}")


if __name__ == "__main__":
    main()

