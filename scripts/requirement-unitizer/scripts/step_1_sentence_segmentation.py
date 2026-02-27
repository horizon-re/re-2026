#!/usr/bin/env python3
"""
Step 1 — Sentence Segmentation (Linguistic Only)
-----------------------------------------------
Splits raw requirement text into sentences using:
  - spaCy
  - Stanza

Produces:
  classifier/step_1/<domain>/<prompt>/<req_id>/
    - sentences_spacy.jsonl
    - sentences_stanza.jsonl
    - sentences_merged.jsonl
    - segmentation_manifest.json
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict

import spacy
import stanza

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]

STEP0_DIR = ROOT / "classifier" / "step_0"
STEP1_DIR = ROOT / "classifier" / "step_1"

STEP1_DIR.mkdir(parents=True, exist_ok=True)

NLP_SPACY = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
stanza.download("en")
NLP_STANZA = stanza.Pipeline(
    lang="en",
    processors="tokenize",
    tokenize_no_ssplit=False,
    use_gpu=False,
    verbose=False,
)

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def write_jsonl(path: Path, records: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_sent_id(req_id: str, idx: int) -> str:
    return f"{req_id}::s{idx:03d}"


# ---------------------------------------------------------------------
# SPLITTERS
# ---------------------------------------------------------------------
def split_spacy(text: str):
    doc = NLP_SPACY(text)
    sents = []
    for i, sent in enumerate(doc.sents, start=1):
        sents.append({
            "order": i,
            "text": sent.text.strip(),
            "char_start": sent.start_char,
            "char_end": sent.end_char,
            "token_count": len(sent),
        })
    return sents


def split_stanza(text: str):
    doc = NLP_STANZA(text)
    sents = []
    i = 1
    for sent in doc.sentences:
        tokens = sent.tokens
        start = tokens[0].start_char
        end = tokens[-1].end_char
        sents.append({
            "order": i,
            "text": sent.text.strip(),
            "char_start": start,
            "char_end": end,
            "token_count": sum(len(t.words) for t in tokens),
        })
        i += 1
    return sents


# ---------------------------------------------------------------------
# MERGE LOGIC (ORDER-PRESERVING)
# ---------------------------------------------------------------------
def merge_splits(spacy_sents, stanza_sents):
    """
    Conservative merge:
    - trust spaCy order
    - mark verification if stanza count ~= spaCy count
    """
    merged = []
    stanza_len = len(stanza_sents)

    for s in spacy_sents:
        verified = abs(stanza_len - len(spacy_sents)) <= max(1, int(0.1 * len(spacy_sents)))
        merged.append({
            **s,
            "verified_by_other": verified,
        })

    return merged


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    processed = 0
    t0 = time.time()

    for domain_dir in STEP0_DIR.iterdir():
        if not domain_dir.is_dir():
            continue

        for prompt_dir in domain_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            for req_dir in prompt_dir.iterdir():
                raw_path = req_dir / "raw.txt"
                hlj_path = req_dir / "hlj.json"

                if not raw_path.exists() or not hlj_path.exists():
                    continue

                with raw_path.open(encoding="utf-8") as f:
                    text = f.read()

                hlj = json.loads(hlj_path.read_text(encoding="utf-8"))
                req_id = hlj["req_id"]
                doc_id = hlj["doc_id"]

                out_dir = STEP1_DIR / domain_dir.name / prompt_dir.name / req_id
                out_dir.mkdir(parents=True, exist_ok=True)

                spacy_sents = split_spacy(text)
                stanza_sents = split_stanza(text)

                spacy_records = []
                for s in spacy_sents:
                    spacy_records.append({
                        "doc_id": doc_id,
                        "req_id": req_id,
                        "sent_id": make_sent_id(req_id, s["order"]),
                        "order": s["order"],
                        "text": s["text"],
                        "char_start": s["char_start"],
                        "char_end": s["char_end"],
                        "splitter": "spacy",
                        "token_count": s["token_count"],
                    })

                stanza_records = []
                for s in stanza_sents:
                    stanza_records.append({
                        "doc_id": doc_id,
                        "req_id": req_id,
                        "sent_id": make_sent_id(req_id, s["order"]),
                        "order": s["order"],
                        "text": s["text"],
                        "char_start": s["char_start"],
                        "char_end": s["char_end"],
                        "splitter": "stanza",
                        "token_count": s["token_count"],
                    })

                merged = merge_splits(spacy_records, stanza_records)
                merged_records = []
                for s in merged:
                    merged_records.append({
                        **s,
                        "doc_id": doc_id,
                        "req_id": req_id,
                        "sent_id": make_sent_id(req_id, s["order"]),
                        "splitter": "merged",
                    })

                write_jsonl(out_dir / "sentences_spacy.jsonl", spacy_records)
                write_jsonl(out_dir / "sentences_stanza.jsonl", stanza_records)
                write_jsonl(out_dir / "sentences_merged.jsonl", merged_records)

                manifest = {
                    "req_id": req_id,
                    "doc_id": doc_id,
                    "num_spacy": len(spacy_records),
                    "num_stanza": len(stanza_records),
                    "num_merged": len(merged_records),
                    "agreement_ok": abs(len(spacy_records) - len(stanza_records)) <= max(1, int(0.1 * len(spacy_records))),
                    "timestamp": time.time(),
                }
                (out_dir / "segmentation_manifest.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8"
                )

                processed += 1

    print(f"[done] Step 1 segmentation complete — {processed} requirements in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
