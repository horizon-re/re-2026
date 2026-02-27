#!/usr/bin/env python3
"""
req_pipeline — MPNet Phase-1.5 Full Ablation Grid Trainer

Trains all required combinations for:
- Context window sizes: n = 1, 2, 3
- Feature sets: sent_ctx, sent_ctx_diff

Excludes:
- sent_only (Phase-1 baseline)
- full (already trained canonical Phase-1.5)
"""

from __future__ import annotations
import subprocess
import sys

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
PYTHON = sys.executable

SCRIPT = "scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py"

N_VALUES = [1, 2, 3]
FEATURES = [
    "sent_ctx",
    "sent_ctx_diff",
]

COMMON_ARGS = [
    "--epochs", "50",
    "--patience", "5",
    "--batch-train", "256",
]

# ---------------------------------------------------------
# RUN GRID
# ---------------------------------------------------------
def main():
    runs = []

    for n in N_VALUES:
        for feat in FEATURES:
            runs.append((n, feat))

    print("=" * 72)
    print("[ablation-grid] Planned runs")
    print("=" * 72)
    for n, feat in runs:
        print(f"  n={n}, features={feat}")
    print("=" * 72)

    for n, feat in runs:
        cmd = [
            PYTHON,
            SCRIPT,
            "--n-before", str(n),
            "--n-after", str(n),
            "--features", feat,
            *COMMON_ARGS,
        ]

        print("\n" + "=" * 72)
        print(f"[run] n={n}, features={feat}")
        print("=" * 72)
        print(" ".join(cmd))
        print("-" * 72)

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[error] run failed for n={n}, features={feat}")
            sys.exit(1)

    print("\n" + "=" * 72)
    print("[done] Ablation grid training complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
