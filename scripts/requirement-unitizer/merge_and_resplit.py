#!/usr/bin/env python3
"""
Merge existing train/holdout splits and create:
- 4k train split
- remaining test split

Author: You, architect of structured overengineering.
"""

import json
import random
from pathlib import Path
from datetime import datetime


# -----------------------
# CONFIG
# -----------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # adjust if needed
SPLITS_DIR = BASE_DIR / "classifier" / "outputs" / "splits"

HOLDOUT_FILE = SPLITS_DIR / "holdout_60.jsonl"
TRAIN_FILE = SPLITS_DIR / "train_40.jsonl"

OUTPUT_TRAIN_FILE = SPLITS_DIR / "train_4k.jsonl"
OUTPUT_TEST_FILE = SPLITS_DIR / "test_rest.jsonl"

TRAIN_TARGET_SIZE = 4000
RANDOM_SEED = 42


# -----------------------
# UTILS
# -----------------------

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def write_jsonl(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# -----------------------
# MAIN
# -----------------------

def main():
    print("Loading datasets...")

    holdout_data = read_jsonl(HOLDOUT_FILE)
    train_data = read_jsonl(TRAIN_FILE)

    print(f"Holdout size: {len(holdout_data)}")
    print(f"Train size:   {len(train_data)}")

    merged = holdout_data + train_data
    print(f"Total merged: {len(merged)}")

    if len(merged) < TRAIN_TARGET_SIZE:
        raise ValueError(
            f"Not enough samples. Needed {TRAIN_TARGET_SIZE}, got {len(merged)}"
        )

    print("Shuffling dataset...")
    random.seed(RANDOM_SEED)
    random.shuffle(merged)

    new_train = merged[:TRAIN_TARGET_SIZE]
    new_test = merged[TRAIN_TARGET_SIZE:]

    print(f"New train size: {len(new_train)}")
    print(f"New test size:  {len(new_test)}")

    write_jsonl(OUTPUT_TRAIN_FILE, new_train)
    write_jsonl(OUTPUT_TEST_FILE, new_test)

    print("Done.")
    print(f"Train written to: {OUTPUT_TRAIN_FILE}")
    print(f"Test written to:  {OUTPUT_TEST_FILE}")


if __name__ == "__main__":
    main()
