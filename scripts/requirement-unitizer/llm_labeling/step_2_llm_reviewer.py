#!/usr/bin/env python3
"""
Step 2b — DeepSeek LLM Reviewer for Requirement Candidates (Batch)
------------------------------------------------------------------
Reads Step 2 outputs (candidates + non_candidates) and asks a local
DeepSeek (via your Flask/Ollama proxy) to label each sentence as:
- requirement_candidate
- not_requirement

Writes:
1) Global combined JSONL (traceable, ML-ready)
2) Per-requirement JSON (audit-friendly)

Also exports metrics:
- acceptance / flip rates
- reason_code distributions
- disagreement counts
"""

from __future__ import annotations

import pathlib
import json
import sys
import re
import time
import logging
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

# ✅ import your existing DeepSeek client wrapper
# It should return: {"response": "<raw model text>"}
from services.llm_clients import call_deepseek_validator  # adjust name if needed

# ── Paths ─────────────────────────────────────────────────────────────
VAULT       = ROOT_DIR
STEP2_DIR   = VAULT / "classifier" / "step_2"

PROMPT_PATH = VAULT / "prompts" / "requirement-unitization" / "step_2b_llm_candidate_reviewer.md"

OUT_GLOBAL_DIR   = VAULT / "classifier" / "llm_reviews"
OUT_BY_REQ_DIR   = VAULT / "classifier" / "llm_reviews_by_req"
PIPELINE_LOG_DIR = VAULT / "05_pipeline_runs" / "current" / "step_2b_llm_review"

OUT_GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_BY_REQ_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_JSONL_PATH = OUT_GLOBAL_DIR / "deepseek_reviews.jsonl"
METRICS_PATH      = OUT_GLOBAL_DIR / "_deepseek_review_metrics.json"

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME   = "deepseek-r1:7b"
BATCH_SIZE   = 20
MAX_RETRIES  = 2
SLEEP_SEC    = 0.05

ALLOWED_LABELS = {"requirement_candidate", "not_requirement"}
ALLOWED_REASON_CODES = {
    "SYSTEM_BEHAVIOR", "SYSTEM_CONSTRAINT", "SECURITY_REQUIREMENT",
    "PERFORMANCE_REQUIREMENT", "COMPLIANCE_REQUIREMENT",
    "ARCHITECTURE_DESCRIPTION", "BACKGROUND_CONTEXT",
    "COST_ANALYSIS", "DESIGN_RATIONALE", "EXAMPLE_OR_EXPLANATION"
}

# ── UTF-8 console handler (Windows safe) ───────────────────────────────
class Utf8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record) + self.terminator
            self.stream.buffer.write(msg.encode("utf-8", errors="replace"))
            self.flush()
        except Exception:
            self.handleError(record)

# ── Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger("step_2b_llm_review")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(PIPELINE_LOG_DIR / "step_2b_llm_review.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

ch = Utf8StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# ── Helpers ────────────────────────────────────────────────────────────
def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def dump_json(path: pathlib.Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def read_jsonl(path: pathlib.Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def append_jsonl(path: pathlib.Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def strip_think_blocks(text: str) -> str:
    # remove <think> blocks + ``` fences
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()
    return text

def safe_json_loads(text: str):
    cleaned = strip_think_blocks(text)
    return json.loads(cleaned)

def chunked(items: List[Tuple[str, dict]], size: int):
    for i in range(0, len(items), size):
        yield items[i:i+size]

def craft_prompt(template: str, sentences: List[dict]) -> str:
    return template.replace("{{ sentences_json }}", json.dumps(sentences, ensure_ascii=False, indent=2))

# ── Validation of LLM output ───────────────────────────────────────────
def validate_llm_items(items: list, expected_sent_ids: set[str]) -> list:
    """
    Enforces schema + allowed enums. Returns a cleaned list.
    If missing items, we fill them as not_requirement with low confidence.
    """
    by_id = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sid = it.get("sent_id")
        if sid not in expected_sent_ids:
            continue

        label = it.get("label", "").strip()
        if label not in ALLOWED_LABELS:
            label = "not_requirement"

        try:
            conf = float(it.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        reason_code = (it.get("reason_code") or "").strip()
        if reason_code not in ALLOWED_REASON_CODES:
            reason_code = "BACKGROUND_CONTEXT"

        reason_text = (it.get("reason_text") or "").strip()
        if not reason_text:
            reason_text = "No reasoning provided."

        by_id[sid] = {
            "sent_id": sid,
            "label": label,
            "confidence": conf,
            "reason_code": reason_code,
            "reason_text": reason_text
        }

    # fill missing
    for sid in expected_sent_ids:
        if sid not in by_id:
            by_id[sid] = {
                "sent_id": sid,
                "label": "not_requirement",
                "confidence": 0.0,
                "reason_code": "BACKGROUND_CONTEXT",
                "reason_text": "Missing from LLM output; defaulted."
            }

    # return stable order matching input set (we'll reorder later anyway)
    return list(by_id.values())

# ── DeepSeek batch call ───────────────────────────────────────────────
def call_deepseek_batch(prompt: str) -> list[dict]:
    """
    Uses your Flask proxy: call_deepseek_validator(prompt_text) -> dict
    Expected dict: {"response": "..."}
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = call_deepseek_validator(prompt)  # your existing wrapper
            raw = resp.get("response", "")
            if not raw:
                raise ValueError("Empty LLM response")
            parsed = safe_json_loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("LLM output is not a JSON array")
            return parsed
        except Exception as e:
            last_err = e
            logger.warning(f"RETRY attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(0.4)
    raise RuntimeError(f"DeepSeek batch call failed: {last_err}")

# ── Core processing ───────────────────────────────────────────────────
def gather_step2_rows(req_dir: pathlib.Path) -> List[Tuple[str, dict]]:
    """
    Returns list of (pipeline_label, record) where pipeline_label in {"candidate","non_candidate"}
    """
    rows: List[Tuple[str, dict]] = []
    cand = req_dir / "candidates.jsonl"
    nonc = req_dir / "non_candidates.jsonl"

    if cand.exists():
        for r in read_jsonl(cand):
            rows.append(("candidate", r))
    if nonc.exists():
        for r in read_jsonl(nonc):
            rows.append(("non_candidate", r))

    return rows

def process_requirement(domain: str, prompt_id: str, req_id: str, req_dir: pathlib.Path, prompt_template: str,
                        metrics_accum: dict):
    rows = gather_step2_rows(req_dir)
    if not rows:
        return

    logger.info(f"REQ   {domain}/{prompt_id}/{req_id} → {len(rows)} sentences to review")

    per_req_reviews = []
    per_req_counts = Counter()

    # batch call
    for batch in chunked(rows, BATCH_SIZE):
        sent_payload = [{"sent_id": rec["sent_id"], "text": rec["text"]} for _, rec in batch]
        expected_ids = {x["sent_id"] for x in sent_payload}

        prompt = craft_prompt(prompt_template, sent_payload)

        llm_items_raw = call_deepseek_batch(prompt)
        llm_items = validate_llm_items(llm_items_raw, expected_ids)
        llm_by_id = {x["sent_id"]: x for x in llm_items}

        # merge + write global jsonl
        for pipeline_label, rec in batch:
            llm = llm_by_id.get(rec["sent_id"])

            review = {
                "doc_id": rec.get("doc_id"),
                "req_id": rec.get("req_id"),
                "sent_id": rec.get("sent_id"),
                "order": rec.get("order"),
                "domain": domain,
                "prompt_id": prompt_id,
                "text": rec.get("text"),

                "pipeline_label": pipeline_label,
                "llm_label": llm["label"],
                "llm_confidence": llm["confidence"],
                "reason_code": llm["reason_code"],
                "reason_text": llm["reason_text"],

                "signals": rec.get("signals", {}),
                "model": MODEL_NAME,
                "review_timestamp": utc_now_iso()
            }

            append_jsonl(GLOBAL_JSONL_PATH, review)
            per_req_reviews.append(review)

            # metrics
            per_req_counts["total"] += 1
            if llm["label"] == "requirement_candidate":
                per_req_counts["llm_pos"] += 1
            else:
                per_req_counts["llm_neg"] += 1

            if pipeline_label == "candidate":
                per_req_counts["pipe_pos"] += 1
            else:
                per_req_counts["pipe_neg"] += 1

            if pipeline_label == "candidate" and llm["label"] == "not_requirement":
                per_req_counts["flip_pos_to_neg"] += 1
            if pipeline_label == "non_candidate" and llm["label"] == "requirement_candidate":
                per_req_counts["flip_neg_to_pos"] += 1

            metrics_accum["reason_codes"][llm["reason_code"]] += 1
            metrics_accum["label_counts"][llm["label"]] += 1
            metrics_accum["flip_counts"]["pos_to_neg"] += int(pipeline_label == "candidate" and llm["label"] == "not_requirement")
            metrics_accum["flip_counts"]["neg_to_pos"] += int(pipeline_label == "non_candidate" and llm["label"] == "requirement_candidate")
            metrics_accum["total"] += 1

        time.sleep(SLEEP_SEC)

    # write per-req JSON
    out_req_dir = OUT_BY_REQ_DIR / domain / prompt_id / req_id
    out_path = out_req_dir / "deepseek_review.json"

    dump_json(out_path, {
        "req_id": req_id,
        "doc_id": per_req_reviews[0].get("doc_id"),
        "domain": domain,
        "prompt_id": prompt_id,
        "model": MODEL_NAME,
        "review_timestamp": utc_now_iso(),
        "counts": dict(per_req_counts),
        "reviews": [
            {
                "sent_id": r["sent_id"],
                "order": r["order"],
                "text": r["text"],
                "pipeline_label": r["pipeline_label"],
                "llm_label": r["llm_label"],
                "confidence": r["llm_confidence"],
                "reason_code": r["reason_code"],
                "reason_text": r["reason_text"],
            }
            for r in sorted(per_req_reviews, key=lambda x: x.get("order") or 0)
        ]
    })

    logger.info(
        f"DONE  {req_id}: pipe_pos={per_req_counts['pipe_pos']} "
        f"llm_pos={per_req_counts['llm_pos']} flips(+→-)={per_req_counts['flip_pos_to_neg']} flips(-→+)={per_req_counts['flip_neg_to_pos']}"
    )

# ── Entry ────────────────────────────────────────────────────────────
def main():
    prompt_template = load_text(PROMPT_PATH)
    if not prompt_template.strip():
        raise RuntimeError(f"Missing prompt template: {PROMPT_PATH}")

    logger.info("START Step 2b — DeepSeek requirement candidate review")
    logger.info(f"INPUT  {STEP2_DIR}")
    logger.info(f"PROMPT {PROMPT_PATH}")
    logger.info(f"OUT    {GLOBAL_JSONL_PATH}")
    logger.info(f"BATCH  {BATCH_SIZE} | MODEL {MODEL_NAME}")

    metrics_accum = {
        "started_at": utc_now_iso(),
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "total": 0,
        "label_counts": Counter(),
        "reason_codes": Counter(),
        "flip_counts": {"pos_to_neg": 0, "neg_to_pos": 0},
    }

    reqs_seen = 0

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

                try:
                    process_requirement(domain, prompt_id, req_id, req_dir, prompt_template, metrics_accum)
                    reqs_seen += 1
                except Exception as e:
                    logger.error(f"FAIL  {domain}/{prompt_id}/{req_id}: {e}")

    # finalize metrics
    metrics_accum["finished_at"] = utc_now_iso()
    metrics_accum["label_counts"] = dict(metrics_accum["label_counts"].most_common())
    metrics_accum["reason_codes"] = dict(metrics_accum["reason_codes"].most_common())

    dump_json(METRICS_PATH, metrics_accum)

    logger.info(f"END   Step 2b — reviewed {metrics_accum['total']} sentences across {reqs_seen} requirements")
    logger.info(f"METRICS saved: {METRICS_PATH}")

if __name__ == "__main__":
    main()
