#!/usr/bin/env python3
"""
Step 2 — Sentence Extraction (spaCy + Stanza)
---------------------------------------------
Splits each requirement text into clean, ID-stable sentences with validation.

Inputs:
    classifier/step_1/<domain>/<prompt>/<req_id>_raw.txt

Outputs:
    classifier/step_2/<domain>/<prompt>/<req_id>/
        ├── sentences_spacy.jsonl
        ├── sentences_stanza.jsonl
        ├── merged_sentences.jsonl
    classifier/step_2/_manifest.jsonl  (summary per requirement)
"""

import os, sys, json, time, re
from pathlib import Path
from tqdm import tqdm
import spacy, stanza
from typing import Dict

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
STEP1_DIR = ROOT / "classifier" / "step_1"
STEP2_DIR = ROOT / "classifier" / "step_2"
LANG = "en"
os.makedirs(STEP2_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_read(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            pass
    return p.read_bytes().decode(errors="ignore")

def clean_text(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt.strip())
    return txt

def extract_sections(text: str) -> Dict[int, str]:
    """Detect simple section headers by line patterns."""
    sections = {}
    current_section = "Unknown"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^(#+\s+|[A-Z][A-Za-z ]+:$|\[.*\]$)", line.strip()):
            current_section = line.strip().strip("#:[] ")
        sections[i] = current_section
    return sections

def split_spacy(text: str, nlp):
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

def split_stanza(text: str, nlp):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sentences if s.text.strip()]

def check_alignment(a, b):
    """Return simple agreement metrics between two sentence lists."""
    if not a or not b:
        return False, 0.0, []
    diff_ratio = abs(len(a) - len(b)) / max(len(a), len(b))
    misaligned = []
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if abs(len(a[i]) - len(b[i])) > 15:
            misaligned.append(i)
    agree = diff_ratio < 0.1 and len(misaligned) < 0.1 * min_len
    return agree, diff_ratio, misaligned

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"[init] Loading spaCy + Stanza pipelines...")
    nlp_spacy = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    if not nlp_spacy.has_pipe("parser"):
        nlp_spacy.add_pipe("parser")
    try:
        nlp_stanza = stanza.Pipeline(LANG, processors="tokenize", verbose=False)
    except Exception:
        stanza.download(LANG)
        nlp_stanza = stanza.Pipeline(LANG, processors="tokenize", verbose=False)

    manifest_records = []

    for domain_path in tqdm(list(STEP1_DIR.iterdir()), desc="Domains"):
        if not domain_path.is_dir():
            continue
        for prompt_path in domain_path.iterdir():
            if not prompt_path.is_dir():
                continue
            for raw_file in prompt_path.glob("*_raw.txt"):
                req_id = raw_file.stem.replace("_raw", "")
                out_dir = STEP2_DIR / domain_path.name / prompt_path.name / req_id
                out_dir.mkdir(parents=True, exist_ok=True)

                text = safe_read(raw_file)
                text = clean_text(text)
                sections = extract_sections(text)

                spacy_sents = split_spacy(text, nlp_spacy)
                stanza_sents = split_stanza(text, nlp_stanza)
                agree, diff_ratio, misaligned = check_alignment(spacy_sents, stanza_sents)

                # --- spaCy sentences
                spacy_path = out_dir / "sentences_spacy.jsonl"
                with open(spacy_path, "w", encoding="utf-8") as f:
                    for i, sent in enumerate(spacy_sents):
                        sent_id = f"{req_id}::s{str(i+1).zfill(3)}"
                        rec = {
                            "req_id": req_id,
                            "sent_id": sent_id,
                            "index": i+1,
                            "text": sent,
                            "splitter": "spacy",
                            "section": "Unknown",
                            "token_count": len(sent.split())
                        }
                        f.write(json.dumps(rec) + "\n")

                # --- stanza sentences
                stanza_path = out_dir / "sentences_stanza.jsonl"
                with open(stanza_path, "w", encoding="utf-8") as f:
                    for i, sent in enumerate(stanza_sents):
                        sent_id = f"{req_id}::s{str(i+1).zfill(3)}"
                        rec = {
                            "req_id": req_id,
                            "sent_id": sent_id,
                            "index": i+1,
                            "text": sent,
                            "splitter": "stanza",
                            "token_count": len(sent.split())
                        }
                        f.write(json.dumps(rec) + "\n")

                # --- merged (spaCy primary, stanza validation)
                merged_path = out_dir / "merged_sentences.jsonl"
                with open(merged_path, "w", encoding="utf-8") as f:
                    for i, sent in enumerate(spacy_sents):
                        sent_id = f"{req_id}::s{str(i+1).zfill(3)}"
                        rec = {
                            "req_id": req_id,
                            "sent_id": sent_id,
                            "index": i+1,
                            "text": sent,
                            "splitter": "spacy",
                            "verified_by_stanza": i < len(stanza_sents),
                            "alignment_offset": abs(len(spacy_sents[i]) - len(stanza_sents[i])) if i < len(stanza_sents) else None
                        }
                        f.write(json.dumps(rec) + "\n")

                # --- manifest record
                manifest_records.append({
                    "req_id": req_id,
                    "domain": domain_path.name,
                    "prompt_id": prompt_path.name,
                    "num_sent_spacy": len(spacy_sents),
                    "num_sent_stanza": len(stanza_sents),
                    "agreement": agree,
                    "diff_ratio": round(diff_ratio, 3),
                    "misaligned_indices": misaligned[:10],
                    "path": str(out_dir.relative_to(ROOT))
                })

    # Write manifest
    manifest_path = STEP2_DIR / "_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    print(f"[done] Sentence extraction complete in {time.time()-t0:.1f}s → {STEP2_DIR}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
