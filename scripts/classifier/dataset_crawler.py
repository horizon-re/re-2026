#!/usr/bin/env python3
"""
--------------------------------------------------------------

Generates structured output files in classifier/step_1/ directory:
- <req_id>_raw.txt - Original raw text content
- <req_id>_hlj.json - High-level JSON structured data

Output structure: classifier/step_1/<domain>/<prompt>/

Usage:
    python scripts/classifier/generate_raw_outputs.py
    python scripts/classifier/generate_raw_outputs.py --domain healthcare --prompt prompt_001
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------
# SMART ROOT DETECTION
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # -> req_pipeline-dataset/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print(f"[init] autodetected project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RAW_DIR = "02_raw_requirements"
REFINED_DIR = "03_refined_json_normalized"
CLASSIFIER_DIR = "classifier/step_1"

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def ensure_dir(p: Path):
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)

def safe_read(p: Path) -> str:
    """Read file with multiple encoding attempts"""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            pass
    return p.read_bytes().decode(errors="ignore")

def extract_text_from_json(obj: Dict[str, Any]) -> str:
    """Extract readable text from JSON structure"""
    if "text" in obj and isinstance(obj["text"], str):
        return obj["text"]
    
    # Extract from common fields
    text_keys = ["summary", "description", "rationale", "acceptance_criteria", 
                 "additionalNotes", "notes", "content", "requirement"]
    parts = []
    
    for key in text_keys:
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            parts.append(obj[key].strip())
    
    if parts:
        return "\n\n".join(parts)
    
    # Fallback to full JSON
    return json.dumps(obj, ensure_ascii=False, indent=2)

def find_source_files(root: Path, domain: Optional[str] = None, 
                      prompt: Optional[str] = None) -> List[Tuple[Path, str, str]]:
    """
    Find all source files (both raw .txt and refined .json)
    Returns: List of (file_path, domain, prompt_id)
    """
    files = []
    
    # Search in raw requirements (both underscore and dash formats)
    raw_root = root / RAW_DIR
    if raw_root.exists():
        for suffix in ["_raw.txt", "-raw.txt"]:
            pattern = f"{domain or '*'}/{prompt or '*'}/*{suffix}"
            for p in raw_root.glob(pattern):
                rel_parts = p.relative_to(raw_root).parts
                if len(rel_parts) >= 3:
                    d, pr = rel_parts[0], rel_parts[1]
                    files.append((p, d, pr))
    
    # Search in refined requirements (both underscore and dash formats)
    refined_root = root / REFINED_DIR
    if refined_root.exists():
        for suffix in ["_refined.json", "-refined.json"]:
            pattern = f"{domain or '*'}/{prompt or '*'}/*{suffix}"
            for p in refined_root.glob(pattern):
                rel_parts = p.relative_to(refined_root).parts
                if len(rel_parts) >= 3:
                    d, pr = rel_parts[0], rel_parts[1]
                    files.append((p, d, pr))
    
    return sorted(set(files))

def extract_req_id(filename: str) -> str:
    """Extract requirement ID from filename"""
    stem = Path(filename).stem
    # Remove -raw or -refined suffix
    stem = stem.replace("-raw", "").replace("_raw", "")
    stem = stem.replace("-refined", "").replace("_refined", "")
    return stem

def generate_output_files(source_path: Path, domain: str, prompt_id: str, 
                         root: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    Generate output files for a single requirement in classifier directory only
    _raw.txt = content from the *_raw.txt file in 02_raw_requirements
    _hlj.json = structured JSON from refined sources
    Returns: Statistics dict
    """
    stats = {"raw_txt": 0, "hlj_json": 0, "errors": 0}
    
    try:
        req_id = extract_req_id(source_path.name)
        is_raw_txt = source_path.suffix == ".txt"
        
        # Find the corresponding raw requirement file for _raw.txt output
        # Try underscore format first (primary format)
        raw_req_path = root / RAW_DIR / domain / prompt_id / f"{req_id}_raw.txt"
        if not raw_req_path.exists():
            # Fallback to dash format
            raw_req_path = root / RAW_DIR / domain / prompt_id / f"{req_id}-raw.txt"
        
        # Also check refined directory if not found in raw
        if not raw_req_path.exists():
            raw_req_path = root / REFINED_DIR / domain / prompt_id / f"{req_id}_raw.txt"
        if not raw_req_path.exists():
            raw_req_path = root / REFINED_DIR / domain / prompt_id / f"{req_id}-raw.txt"
        
        # Read the actual raw requirement text
        if raw_req_path.exists():
            raw_text = safe_read(raw_req_path)
        else:
            print(f"[warn] no raw requirement file found for {req_id}, skipping")
            stats["errors"] += 1
            return stats
        
        # Read source content for HLJ
        if is_raw_txt:
            hlj_data = {
                "requirement_id": req_id,
                "prompt_id": prompt_id,
                "domain": domain,
                "text": raw_text,
                "source_path": str(source_path.relative_to(root)),
                "origin": "raw_txt"
            }
        else:  # JSON file
            try:
                json_obj = json.loads(safe_read(source_path))
            except json.JSONDecodeError:
                print(f"[warn] invalid JSON in {source_path}, skipping")
                stats["errors"] += 1
                return stats
            
            hlj_data = {
                "requirement_id": req_id,
                "prompt_id": prompt_id,
                "domain": domain,
                "text": raw_text,  # Still use raw text for consistency
                "source_path": str(source_path.relative_to(root)),
                "origin": "refined_json",
                "structured_data": json_obj
            }
        
        if dry_run:
            print(f"[dry-run] {req_id} -> {domain}/{prompt_id} ({len(raw_text)} chars)")
            stats["raw_txt"] += 1
            stats["hlj_json"] += 1
            return stats
        
        # Create output directory in classifier/step_1/<domain>/<prompt>/
        classifier_dir = root / CLASSIFIER_DIR / domain / prompt_id
        ensure_dir(classifier_dir)
        
        # Write raw text file
        raw_txt_filename = f"{req_id}_raw.txt"
        classifier_raw_path = classifier_dir / raw_txt_filename
        classifier_raw_path.write_text(raw_text, encoding="utf-8")
        stats["raw_txt"] += 1
        print(f"[raw] {classifier_raw_path.relative_to(root)}")
        
        # Write HLJ JSON file
        hlj_filename = f"{req_id}_hlj.json"
        classifier_hlj_path = classifier_dir / hlj_filename
        with open(classifier_hlj_path, "w", encoding="utf-8") as f:
            json.dump(hlj_data, f, ensure_ascii=False, indent=2)
        stats["hlj_json"] += 1
        print(f"[hlj] {classifier_hlj_path.relative_to(root)}")
        
    except Exception as e:
        print(f"[error] failed processing {source_path}: {e}")
        stats["errors"] += 1
    
    return stats

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate raw requirements output files in classifier/step_1/ only"
    )
    ap.add_argument("--root", default=str(PROJECT_ROOT), 
                   help="Project root directory")
    ap.add_argument("--domain", help="Filter by specific domain")
    ap.add_argument("--prompt", help="Filter by specific prompt ID")
    ap.add_argument("--dry-run", action="store_true",
                   help="Show what would be done without writing files")
    args = ap.parse_args()
    
    t0 = time.time()
    root = Path(args.root).resolve()
    
    print(f"[scan] searching for source files...")
    source_files = find_source_files(root, args.domain, args.prompt)
    print(f"[scan] found {len(source_files)} source files")
    
    if not source_files:
        print("[warn] no source files found")
        return
    
    # Process all files
    total_stats = {"raw_txt": 0, "hlj_json": 0, "errors": 0}
    
    for source_path, domain, prompt_id in source_files:
        stats = generate_output_files(source_path, domain, prompt_id, root, args.dry_run)
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Print summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"[summary] Processing complete in {elapsed:.1f}s")
    print(f"  Output directory:   {CLASSIFIER_DIR}")
    print(f"  Raw text files:     {total_stats['raw_txt']}")
    print(f"  HLJ JSON files:     {total_stats['hlj_json']}")
    print(f"  Errors:             {total_stats['errors']}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()