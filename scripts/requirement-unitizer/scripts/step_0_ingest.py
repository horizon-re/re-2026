#!/usr/bin/env python3
"""
Step 0 — Ingestion & Canonicalization (req_pipeline Unitization Pipeline)
--------------------------------------------------------------------
Creates lossless, traceable per-requirement packets under classifier/step_0/.

Nav-friendly: reads dataset_nav.yaml and only processes files that match:
- 02_raw_requirements/**/_raw.txt or -raw.txt
- optionally attaches 03_refined_json_normalized/**/_refined.json or -refined.json

Outputs:
  classifier/step_0/<domain>/<prompt>/<req_id>/
    - raw.txt
    - hlj.json
    - meta.json
"""

from __future__ import annotations

import argparse
import json
import hashlib
import time
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

try:
    import yaml  # pip install pyyaml
except ImportError:
    print("Missing dependency: pyyaml (pip install pyyaml)", file=sys.stderr)
    raise

import uuid


# -----------------------------
# CONFIG
# -----------------------------
RAW_ROOTS = {"02_raw_requirements", "raw_requirements"}
REFINED_ROOTS = {"03_refined_json_normalized", "03_refined_json", "refined_json"}
OUT_ROOT = Path("classifier") / "step_0"

RAW_SUFFIXES = ("_raw.txt", "-raw.txt")
REFINED_SUFFIXES = ("_refined.json", "-refined.json")

REQ_ID_RX = re.compile(r"(REQ[-_ ]?\d{1,8})", re.I)


# -----------------------------
# UTILS
# -----------------------------
def safe_read_text(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            pass
    return p.read_bytes().decode(errors="ignore")


def sha256_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_req_id_from_filename(filename: str) -> str:
    """
    Try to extract REQ-### from filename; fall back to stem.
    """
    m = REQ_ID_RX.search(filename)
    if m:
        s = m.group(1).upper().replace(" ", "").replace("_", "-")
        if "-" not in s:
            # REQ001 -> REQ-001
            s = s[:3] + "-" + s[3:]
        return s
    # fallback: strip known suffixes
    stem = Path(filename).stem
    stem = stem.replace("_raw", "").replace("-raw", "")
    stem = stem.replace("_refined", "").replace("-refined", "")
    return stem.upper()


def uuid_v5_for_req(req_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, req_key))


def parse_rel(rel_path: str) -> Tuple[str, str, str]:
    """
    rel_path like: 02_raw_requirements/FinTech/prompt_001/REQ-001_raw.txt
    returns: (domain, prompt, req_id)
    """
    parts = Path(rel_path).parts
    # Expect at least: root/domain/prompt/file
    if len(parts) < 4:
        raise ValueError(f"Bad path (too short): {rel_path}")
    domain = parts[1]
    prompt = parts[2]
    req_id = extract_req_id_from_filename(parts[-1])
    return domain, prompt, req_id


def load_nav(nav_path: Path) -> List[Dict[str, Any]]:
    nav = yaml.safe_load(nav_path.read_text(encoding="utf-8"))
    return nav.get("files", [])


def is_raw_entry(e: Dict[str, Any]) -> bool:
    rp = e.get("rel_path", "")
    return any(rp.startswith(root + "/") for root in RAW_ROOTS) and rp.endswith(RAW_SUFFIXES)


def is_refined_entry(e: Dict[str, Any]) -> bool:
    rp = e.get("rel_path", "")
    return any(rp.startswith(root + "/") for root in REFINED_ROOTS) and rp.endswith(REFINED_SUFFIXES)


def build_refined_index(nav_files: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], str]:
    """
    Map (domain, prompt, req_id) -> refined_rel_path
    """
    idx: Dict[Tuple[str, str, str], str] = {}
    for e in nav_files:
        if not is_refined_entry(e):
            continue
        rel = e["rel_path"]
        try:
            d, p, req_id = parse_rel(rel)
        except Exception:
            continue
        idx[(d, p, req_id)] = rel
    return idx


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root (where dataset_nav.yaml lives)")
    ap.add_argument("--nav", default="dataset_nav.yaml", help="Path to dataset_nav.yaml")
    ap.add_argument("--out", default=str(OUT_ROOT), help="Output root directory")
    ap.add_argument("--domain", default=None, help="Optional filter: only process a specific domain")
    ap.add_argument("--prompt", default=None, help="Optional filter: only process a specific prompt_id")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing step_0 outputs")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    nav_path = (root / args.nav).resolve()
    out_root = (root / args.out).resolve()

    if not nav_path.exists():
        raise FileNotFoundError(f"dataset nav not found: {nav_path}")

    ensure_dir(out_root)

    nav_files = load_nav(nav_path)
    refined_idx = build_refined_index(nav_files)

    raw_entries = [e for e in nav_files if is_raw_entry(e)]
    if args.domain:
        raw_entries = [e for e in raw_entries if f"/{args.domain}/" in e["rel_path"]]
    if args.prompt:
        raw_entries = [e for e in raw_entries if f"/{args.prompt}/" in e["rel_path"]]
    if args.limit:
        raw_entries = raw_entries[: args.limit]

    print(f"[step0] nav entries: {len(nav_files)} | raw candidates: {len(raw_entries)}")

    written = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for e in raw_entries:
        rel_raw = e["rel_path"]
        try:
            domain, prompt, req_id = parse_rel(rel_raw)
            req_key = f"{domain}::{prompt}::{req_id}"
            doc_id = uuid_v5_for_req(req_key)

            # paths
            abs_raw = root / rel_raw
            rel_refined = refined_idx.get((domain, prompt, req_id))
            abs_refined = (root / rel_refined) if rel_refined else None

            if not abs_raw.exists():
                print(f"[warn] missing raw file: {rel_raw}")
                errors += 1
                continue

            out_dir = out_root / domain / prompt / req_id
            if out_dir.exists() and not args.overwrite:
                skipped += 1
                continue
            ensure_dir(out_dir)

            raw_text = safe_read_text(abs_raw)

            refined_obj = None
            refined_sha = None
            if abs_refined and abs_refined.exists():
                try:
                    refined_text = safe_read_text(abs_refined)
                    refined_obj = json.loads(refined_text)
                    refined_sha = sha256_text(refined_text)
                except Exception:
                    # keep pipeline robust; refined is optional
                    refined_obj = None
                    refined_sha = None

            # write raw verbatim
            (out_dir / "raw.txt").write_text(raw_text, encoding="utf-8")

            # hlj packet
            hlj = {
                "doc_id": doc_id,
                "req_key": req_key,
                "req_id": req_id,
                "domain": domain,
                "prompt_id": prompt,
                "source": {
                    "raw_path": rel_raw,
                    "refined_path": rel_refined,
                    "origin": "both" if rel_refined else "raw_only",
                },
                "text": raw_text,
                "structured_data": refined_obj,
            }
            (out_dir / "hlj.json").write_text(json.dumps(hlj, ensure_ascii=False, indent=2), encoding="utf-8")

            meta = {
                "doc_id": doc_id,
                "req_key": req_key,
                "req_id": req_id,
                "domain": domain,
                "prompt_id": prompt,
                "char_count": len(raw_text),
                "line_count": raw_text.count("\n") + 1 if raw_text else 0,
                "sha256_raw": sha256_text(raw_text),
                "sha256_refined": refined_sha,
                "created_at": time.time(),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            written += 1

        except Exception as ex:
            errors += 1
            print(f"[error] {rel_raw}: {ex}")

    dt = time.time() - t0
    print(f"[done] step0 ingest: written={written} skipped={skipped} errors={errors} in {dt:.1f}s")
    print(f"[out] {out_root}")

if __name__ == "__main__":
    main()
