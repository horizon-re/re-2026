#!/usr/bin/env python3
"""
Step 3b — Context Window Builder
--------------------------------
Builds context windows for borderline atomic units.
NO ML. Deterministic only.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
STEP1_DIR = ROOT / "classifier" / "step_1"
STEP3A_DIR = ROOT / "classifier" / "step_3a"
STEP3B_DIR = ROOT / "classifier" / "step_3b"
STEP3B_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 2  # sentences before / after

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def load_jsonl(p: Path) -> List[Dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def write_jsonl(p: Path, rows: List[Dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    for domain_dir in STEP3A_DIR.iterdir():
        if not domain_dir.is_dir():
            continue

        for prompt_dir in domain_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            for req_dir in prompt_dir.iterdir():
                borderline_path = req_dir / "borderline.jsonl"
                if not borderline_path.exists():
                    continue

                borderline_units = load_jsonl(borderline_path)
                if not borderline_units:
                    continue

                sent_path = STEP1_DIR / domain_dir.name / prompt_dir.name / req_dir.name / "sentences_merged.jsonl"
                if not sent_path.exists():
                    continue

                sentences = load_jsonl(sent_path)
                sent_index = {s["sent_id"]: i for i, s in enumerate(sentences)}

                contextual = []

                for u in borderline_units:
                    sid = u["parent_sent_id"]
                    if sid not in sent_index:
                        continue

                    idx = sent_index[sid]
                    before = sentences[max(0, idx - WINDOW): idx]
                    after = sentences[idx + 1: idx + 1 + WINDOW]

                    contextual.append({
                        "unit_id": u["unit_id"],
                        "req_id": u["req_id"],
                        "doc_id": u["doc_id"],
                        "target": u["text"],
                        "context_before": [s["text"] for s in before],
                        "context_after": [s["text"] for s in after],
                        "window_size": WINDOW,
                        "context_scope": "same_requirement",
                        "source_parent_sent": sid,
                        "local_score": u["candidate_score"],
                    })

                out_dir = STEP3B_DIR / domain_dir.name / prompt_dir.name / req_dir.name
                write_jsonl(out_dir / "contextual_units.jsonl", contextual)

    print("[done] Step 3b context window construction complete")


if __name__ == "__main__":
    main()
