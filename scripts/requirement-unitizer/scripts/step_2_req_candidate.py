#!/usr/bin/env python3
"""
Step 2 — Requirement Candidate Detection + Metrics
--------------------------------------------------
Semantic gate that identifies requirement-bearing sentences
and exports full acceptance matrices for audit and research.
"""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
import spacy

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
STEP1_DIR = ROOT / "classifier" / "step_1"
STEP2_DIR = ROOT / "classifier" / "step_2"
STEP2_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------
NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

MODAL_RE = re.compile(r"\b(shall|must|should|will|may|required to)\b", re.I)
CONSTRAINT_RE = re.compile(
    r"(<=|>=|<|>|at least|no more than|\bms\b|\bsec\b|\b%\b|\btps\b)",
    re.I
)

SYSTEM_NOUNS = {
    "system", "application", "service", "api",
    "platform", "backend", "frontend", "server", "client"
}

# ---------------------------------------------------------------------
# SIGNAL FUNCTIONS
# ---------------------------------------------------------------------
def has_action_verb(doc) -> bool:
    return any(tok.pos_ == "VERB" and tok.dep_ in {"ROOT", "conj"} for tok in doc)

def has_modal(text: str) -> bool:
    return bool(MODAL_RE.search(text))

def has_constraint(text: str) -> bool:
    return bool(CONSTRAINT_RE.search(text))

def has_system_subject(doc) -> bool:
    return any(
        tok.dep_ == "nsubj" and tok.text.lower() in SYSTEM_NOUNS
        for tok in doc
    )

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    t0 = time.time()

    TOTAL_REQS = 0
    TOTAL_SENTENCES = 0
    TOTAL_CANDIDATES = 0

    GLOBAL_SIGNAL_COUNTS = Counter()
    GLOBAL_ACCEPTED_SIGNALS = Counter()
    GLOBAL_SIGNAL_COMBOS = Counter()
    DOMAIN_STATS = defaultdict(lambda: {"sentences": 0, "candidates": 0})

    for domain_dir in STEP1_DIR.iterdir():
        if not domain_dir.is_dir():
            continue

        for prompt_dir in domain_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            for req_dir in prompt_dir.iterdir():
                merged_path = req_dir / "sentences_merged.jsonl"
                if not merged_path.exists():
                    continue

                TOTAL_REQS += 1
                out_dir = STEP2_DIR / domain_dir.name / prompt_dir.name / req_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)

                candidates, non_candidates = [], []
                local_signal_counts = Counter()
                local_signal_combos = Counter()

                with merged_path.open(encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        text = rec["text"]
                        doc = NLP(text)

                        s_action = has_action_verb(doc)
                        s_modal = has_modal(text)
                        s_system = has_system_subject(doc)
                        s_constraint = has_constraint(text)

                        signals = {
                            "action": s_action,
                            "modal": s_modal,
                            "system": s_system,
                            "constraint": s_constraint
                        }

                        TOTAL_SENTENCES += 1
                        DOMAIN_STATS[domain_dir.name]["sentences"] += 1

                        for k, v in signals.items():
                            if v:
                                GLOBAL_SIGNAL_COUNTS[k] += 1
                                local_signal_counts[k] += 1

                        is_candidate = (
                            s_action and (s_modal or s_system or s_constraint)
                        )

                        record = {
                            "doc_id": rec["doc_id"],
                            "req_id": rec["req_id"],
                            "sent_id": rec["sent_id"],
                            "order": rec["order"],
                            "text": text,
                            "is_candidate": is_candidate,
                            "signals": signals
                        }

                        if is_candidate:
                            candidates.append(record)
                            TOTAL_CANDIDATES += 1
                            DOMAIN_STATS[domain_dir.name]["candidates"] += 1

                            for k, v in signals.items():
                                if v:
                                    GLOBAL_ACCEPTED_SIGNALS[k] += 1

                            combo = "+".join(sorted(k for k, v in signals.items() if v))
                            GLOBAL_SIGNAL_COMBOS[combo] += 1
                            local_signal_combos[combo] += 1
                        else:
                            non_candidates.append(record)

                # Write JSONL outputs
                def dump_jsonl(path, rows):
                    with path.open("w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                dump_jsonl(out_dir / "candidates.jsonl", candidates)
                dump_jsonl(out_dir / "non_candidates.jsonl", non_candidates)

                # Per-requirement matrices
                total = len(candidates) + len(non_candidates)
                signal_matrix = {
                    "req_id": req_dir.name,
                    "num_sentences": total,
                    "num_candidates": len(candidates),
                    "acceptance_rate": round(len(candidates) / max(1, total), 3),
                    "signal_counts": dict(local_signal_counts),
                    "top_signal_combinations": [
                        {"pattern": k, "count": v}
                        for k, v in local_signal_combos.most_common(5)
                    ]
                }

                (out_dir / "signal_matrix.json").write_text(
                    json.dumps(signal_matrix, indent=2),
                    encoding="utf-8"
                )

                manifest = {
                    "req_id": req_dir.name,
                    "domain": domain_dir.name,
                    "prompt_id": prompt_dir.name,
                    "num_sentences": total,
                    "num_candidates": len(candidates)
                }

                (out_dir / "candidate_manifest.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8"
                )

    # -----------------------------------------------------------------
    # GLOBAL MATRICES
    # -----------------------------------------------------------------
    global_metrics = {
        "total_requirements": TOTAL_REQS,
        "total_sentences": TOTAL_SENTENCES,
        "total_candidates": TOTAL_CANDIDATES,
        "global_acceptance_rate": round(TOTAL_CANDIDATES / max(1, TOTAL_SENTENCES), 3),
        "signal_acceptance_rates": {
            k: round(GLOBAL_ACCEPTED_SIGNALS[k] / max(1, GLOBAL_SIGNAL_COUNTS[k]), 3)
            for k in GLOBAL_SIGNAL_COUNTS
        }
    }

    (STEP2_DIR / "_global_metrics.json").write_text(
        json.dumps(global_metrics, indent=2),
        encoding="utf-8"
    )

    (STEP2_DIR / "_signal_cooccurrence.json").write_text(
        json.dumps(dict(GLOBAL_SIGNAL_COMBOS.most_common()), indent=2),
        encoding="utf-8"
    )

    domain_matrix = {
        d: {
            "sentences": v["sentences"],
            "acceptance_rate": round(v["candidates"] / max(1, v["sentences"]), 3)
        }
        for d, v in DOMAIN_STATS.items()
    }

    (STEP2_DIR / "_domain_matrix.json").write_text(
        json.dumps(domain_matrix, indent=2),
        encoding="utf-8"
    )

    print(
        f"[done] Step 2 complete — "
        f"{TOTAL_REQS} reqs | "
        f"{TOTAL_SENTENCES} sentences | "
        f"{TOTAL_CANDIDATES} candidates | "
        f"{time.time() - t0:.1f}s"
    )

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
