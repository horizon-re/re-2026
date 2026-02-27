#!/usr/bin/env python3
import json
import pathlib
import sys
import hashlib
from datetime import datetime
from typing import List, Tuple

import spacy
import requests

# ---------------------------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

STEP2_DIR = ROOT / "classifier" / "step_2"
RAW_DIR = ROOT / "02_raw_requirements"

OUTPUT_ROOT = ROOT / "classifier" / "outputs"
MASTER_OUT = OUTPUT_ROOT / "all_annotations.jsonl"
BY_CAT_DIR = OUTPUT_ROOT / "by_category"
SPLIT_DIR = OUTPUT_ROOT / "splits"

PROMPT_PATH = ROOT / "prompts/requirement-unitization/manual_annotation_llm.md"
EXAMPLES_PATH = ROOT / "prompts/requirement-unitization/manual_annotation_examples.json"

LLM_ENDPOINT = "http://localhost:5007/gpt-5-mini"

WINDOW = 5  # 5 sentences before + 5 after

for d in [OUTPUT_ROOT, BY_CAT_DIR, SPLIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# SpaCy is used ONLY to split raw text ONCE
NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_text(path: pathlib.Path):
    return path.read_text(encoding="utf-8")

def split_raw_sentences(raw_text: str) -> List[str]:
    doc = NLP(raw_text)
    return [s.text.strip() for s in doc.sents]

def resolve_context(item: dict) -> Tuple[List[str], List[str]]:
    """
    Uses item['order'] as the canonical sentence index.
    Does NOT attempt to re-identify or match sentences by text.
    """
    raw_path = (
        RAW_DIR
        / item["domain"]
        / item["prompt_id"]
        / f"{item['req_id']}_raw.txt"
    )
    if not raw_path.exists():
        return [], []

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    sentences = split_raw_sentences(raw_text)

    idx = item["order"] - 1  # order is 1-based
    if idx < 0 or idx >= len(sentences):
        return [], []

    before = sentences[max(0, idx - WINDOW): idx]
    after = sentences[idx + 1: min(len(sentences), idx + WINDOW + 1)]

    return before, after

def stable_split(key: str, threshold=0.4) -> str:
    h = hashlib.sha256(key.encode()).hexdigest()
    ratio = int(h[:8], 16) / 0xFFFFFFFF
    return "train" if ratio < threshold else "holdout"

# ---------------------------------------------------------------------
# OUTPUT WRITERS
# ---------------------------------------------------------------------
def write_master(record: dict):
    with MASTER_OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_by_category(record: dict):
    for label in record["llm_labels"]:
        out = BY_CAT_DIR / f"{label}.jsonl"
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_split(record: dict):
    key = f"{record['req_id']}::{record['sent_id']}"
    split = stable_split(key)

    out = SPLIT_DIR / (
        "train_40.jsonl" if split == "train" else "holdout_60.jsonl"
    )
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def persist(record: dict):
    write_master(record)
    write_by_category(record)
    write_split(record)

# ---------------------------------------------------------------------
# PROMPTING
# ---------------------------------------------------------------------
def craft_prompt(
    item: dict,
    context_before: List[str],
    context_after: List[str],
    examples: list,
) -> str:
    template = load_text(PROMPT_PATH)
    return (
        template
        .replace("{{ examples_json }}", json.dumps(examples, indent=2))
        .replace("{{ pipeline_label }}", item["pipeline_label"])
        .replace("{{ sentence }}", item["text"])
        .replace("{{ context_before }}", json.dumps(context_before, indent=2))
        .replace("{{ context_after }}", json.dumps(context_after, indent=2))
    )

def call_llm(prompt: str) -> dict:
    try:
        resp = requests.post(
            LLM_ENDPOINT,
            json={"text": prompt},
            timeout=60,
        )
        data = resp.json()
        return json.loads(data.get("response", "{}"))
    except Exception:
        return {
            "llm_labels": ["review_later"],
            "reason": "LLM output could not be parsed as valid JSON.",
            "confidence": 0.0,
        }

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    examples = load_json(EXAMPLES_PATH)

    for domain_dir in STEP2_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name

        for prompt_dir in domain_dir.iterdir():
            if not prompt_dir.is_dir():
                continue
            prompt_id = prompt_dir.name

            for req_dir in prompt_dir.iterdir():
                if not req_dir.is_dir():
                    continue
                req_id = req_dir.name

                for fname in ("candidates.jsonl", "non_candidates.jsonl"):
                    path = req_dir / fname
                    if not path.exists():
                        continue

                    pipeline_label = (
                        "candidate" if fname == "candidates.jsonl" else "non_candidate"
                    )

                    for item in load_jsonl(path):
                        item.update(
                            {
                                "domain": domain,
                                "prompt_id": prompt_id,
                                "req_id": req_id,
                                "pipeline_label": pipeline_label,
                            }
                        )

                        context_before, context_after = resolve_context(item)
                        prompt = craft_prompt(
                            item, context_before, context_after, examples
                        )
                        llm_out = call_llm(prompt)

                        record = {
                            "req_id": req_id,
                            "sent_id": item["sent_id"],
                            "domain": domain,
                            "prompt_id": prompt_id,
                            "order": item["order"],
                            "sentence": item["text"],
                            "context_before": context_before,
                            "context_after": context_after,
                            "pipeline_label": pipeline_label,
                            "llm_labels": llm_out.get("llm_labels", ["review_later"]),
                            "reason": llm_out.get("reason"),
                            "confidence": llm_out.get("confidence", 0.0),
                            "annotated_by": "gpt-5-mini",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }

                        persist(record)

    print("✅ LLM annotation pipeline complete.")

if __name__ == "__main__":
    main()
