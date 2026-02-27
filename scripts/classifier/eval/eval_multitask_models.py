#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Multi-Task DeBERTa Models
-----------------------------------

Evaluates all trained multi-task models on the holdout test set:
- Different model architectures (DeBERTa, MiniLM, MPNet, etc.)
- Different hyperparameters (LoRA rank, learning rate, context size)
- Compares against baseline (Phase-1 MPNet)

Outputs:
- Comprehensive metrics for all models
- Comparison table
- Best model identification
- Per-task performance breakdown
"""

from __future__ import annotations

import os, sys, json, time, argparse
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import csv

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
# Navigate to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[init] project root: {PROJECT_ROOT}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODELS_ROOT = Path("classifier/models/multitask_deberta_lora")
BASELINE_ROOT = Path("classifier/models/mpnet_phase1")
TEST_DATA_PATH = Path("classifier/outputs/splits/train_40.jsonl")  # Adjust if different
OUT_ROOT = Path("classifier/outputs/eval")

MAX_LENGTH = 128

# Model architectures
MODEL_OPTIONS = {
    "deberta": "microsoft/deberta-v3-base",
    "minilm": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet": "microsoft/mpnet-base",
    "paraphrase-mpnet": "sentence-transformers/paraphrase-mpnet-base-v2",
}

# ---------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------
@dataclass
class Sample:
    """Test sample"""
    text: str
    is_requirement: Optional[int] = None
    is_functional: Optional[int] = None
    is_ambiguous: Optional[int] = None
    source: str = "test"

# ---------------------------------------------------------------------
# MODEL HEADS
# ---------------------------------------------------------------------
class MultiTaskHead(nn.Module):
    """Multi-task head (same as training)"""
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.req_head = nn.Linear(256, 1)
        self.func_head = nn.Linear(256, 1)
        self.amb_head = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_features = self.shared(x)
        req_logits = self.req_head(shared_features).squeeze(-1)
        func_logits = self.func_head(shared_features).squeeze(-1)
        amb_logits = self.amb_head(shared_features).squeeze(-1)
        return req_logits, func_logits, amb_logits


class BaselineHead(nn.Module):
    """Baseline single-task head (Phase-1)"""
    def __init__(self, state_dict: dict):
        super().__init__()
        w0 = state_dict["net.0.weight"]
        in_dim = w0.shape[1]
        hidden1 = w0.shape[0]
        
        if "net.3.weight" in state_dict:
            # 2-layer
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden1, 1),
            )
        else:
            # 1-layer
            self.net = nn.Sequential(nn.Linear(in_dim, 1))
        
        self.load_state_dict(state_dict)
    
    def forward(self, x):
        out = self.net(x)
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[warn] File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_pooling(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def embed_batch(texts: List[str], tokenizer, encoder, batch_size: int = 32) -> torch.Tensor:
    """Embed texts in batches"""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = encoder(**enc).last_hidden_state
        pooled = mean_pooling(out, enc["attention_mask"])
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, task_name: str = "") -> Dict[str, float]:
    """Compute comprehensive metrics"""
    # Filter NaN
    valid_idx = ~np.isnan(y_true)
    if valid_idx.sum() == 0:
        return {f"{task_name}_n": 0}
    
    y_true = y_true[valid_idx]
    probs = probs[valid_idx]
    preds = (probs >= 0.5).astype(int)
    
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    acc = accuracy_score(y_true, preds)
    
    try:
        roc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    except:
        roc = float("nan")
    
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except:
        pr_auc = float("nan")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    return {
        f"{task_name}_n": int(valid_idx.sum()),
        f"{task_name}_accuracy": float(acc),
        f"{task_name}_precision": float(p),
        f"{task_name}_recall": float(r),
        f"{task_name}_f1": float(f1),
        f"{task_name}_roc_auc": float(roc),
        f"{task_name}_pr_auc": float(pr_auc),
        f"{task_name}_tp": int(tp),
        f"{task_name}_fp": int(fp),
        f"{task_name}_tn": int(tn),
        f"{task_name}_fn": int(fn),
    }


def build_context(row: Dict[str, Any], n_context: int) -> str:
    """Build context text from row"""
    before = (row.get("context_before") or [])[-n_context:]
    after = (row.get("context_after") or [])[:n_context]
    sent = (row.get("sentence") or "").strip()
    
    parts = []
    if before:
        parts.append(" ".join(b.strip() for b in before if isinstance(b, str) and b.strip()))
    parts.append(sent)
    if after:
        parts.append(" ".join(a.strip() for a in after if isinstance(a, str) and a.strip()))
    
    return " [CTX] ".join([p for p in parts if p])


def load_test_data(path: Path) -> Tuple[List[Sample], np.ndarray, np.ndarray, np.ndarray]:
    """Load test data and extract labels"""
    rows = load_jsonl(path)
    samples = []
    
    for r in rows:
        labels = set(r.get("llm_labels", []) or [])
        text = r.get("sentence", "").strip()
        
        if not text:
            continue
        
        # Task 1: requirement classification
        is_req = 1 if ("requirement" in labels or "with_context" in labels) else 0 if "non_requirement" in labels else None
        
        # Task 2: functional classification (if available)
        # You may need to adjust this based on your test data structure
        is_func = None  # Not available in standard test set
        
        # Task 3: ambiguity
        is_amb = 1 if "ambiguous" in labels else 0 if "non_ambiguous" in labels else None
        
        samples.append(Sample(
            text=text,
            is_requirement=is_req,
            is_functional=is_func,
            is_ambiguous=is_amb,
            source="test"
        ))
    
    # Convert to arrays
    y_req = np.array([s.is_requirement if s.is_requirement is not None else np.nan for s in samples], dtype=np.float32)
    y_func = np.array([s.is_functional if s.is_functional is not None else np.nan for s in samples], dtype=np.float32)
    y_amb = np.array([s.is_ambiguous if s.is_ambiguous is not None else np.nan for s in samples], dtype=np.float32)
    
    return samples, y_req, y_func, y_amb


# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------
def evaluate_multitask_model(
    model_dir: Path,
    samples: List[Sample],
    y_req: np.ndarray,
    y_func: np.ndarray,
    y_amb: np.ndarray,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate a single multi-task model"""
    
    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"[skip] No config.json in {model_dir}")
        return None
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Load checkpoint
    ckpt_path = model_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "last.pt"
        if not ckpt_path.exists():
            print(f"[skip] No checkpoint in {model_dir}")
            return None
    
    print(f"\n[eval] {model_dir.name}")
    print(f"  Model: {config.get('model', 'unknown')}")
    print(f"  Context: n={config.get('n_context', 'unknown')}")
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Get model name and load encoder
    model_name = config.get("model", MODEL_OPTIONS.get(config.get("model_type", "deberta")))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    
    # Load LoRA weights if present
    if "encoder_state" in ckpt:
        from peft import PeftModel, LoraConfig
        
        # Try to load as LoRA model
        try:
            lora_config = LoraConfig(
                r=config["lora"]["r"],
                lora_alpha=config["lora"]["alpha"],
                lora_dropout=config["lora"]["dropout"],
                target_modules=config["lora"]["target_modules"],
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            from peft import get_peft_model
            encoder = get_peft_model(encoder, lora_config)
            encoder.load_state_dict(ckpt["encoder_state"])
        except Exception as e:
            print(f"  [warn] Could not load LoRA weights: {e}")
    
    encoder.eval()
    
    # Load head
    emb_dim = config.get("lora", {}).get("embedding_dim", 768)
    if hasattr(encoder.base_model if hasattr(encoder, 'base_model') else encoder, 'config'):
        model_config = encoder.base_model.config if hasattr(encoder, 'base_model') else encoder.config
        if hasattr(model_config, 'hidden_size'):
            emb_dim = model_config.hidden_size
    
    head = MultiTaskHead(in_dim=emb_dim).to(DEVICE)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    
    # Build texts with context
    n_context = config.get("n_context", 1)
    
    # For test data, we'll use sentence only (unless you have context in test set)
    texts = [s.text for s in samples]
    
    print(f"  Embedding {len(texts)} samples...")
    
    # Embed
    embs = embed_batch(texts, tokenizer, encoder, batch_size)
    
    # Predict
    print(f"  Predicting...")
    with torch.no_grad():
        logits_req, logits_func, logits_amb = head(embs.to(DEVICE))
        probs_req = torch.sigmoid(logits_req).cpu().numpy()
        probs_func = torch.sigmoid(logits_func).cpu().numpy()
        probs_amb = torch.sigmoid(logits_amb).cpu().numpy()
    
    # Compute metrics for each task
    metrics_req = compute_metrics(y_req, probs_req, "req")
    metrics_func = compute_metrics(y_func, probs_func, "func")
    metrics_amb = compute_metrics(y_amb, probs_amb, "amb")
    
    print(f"  Task 1 (Req): F1={metrics_req.get('req_f1', 0):.4f}, Acc={metrics_req.get('req_accuracy', 0):.4f}")
    print(f"  Task 2 (F/NFR): F1={metrics_func.get('func_f1', 0):.4f}, n={metrics_func.get('func_n', 0)}")
    print(f"  Task 3 (Amb): F1={metrics_amb.get('amb_f1', 0):.4f}, n={metrics_amb.get('amb_n', 0)}")
    
    return {
        "model_dir": str(model_dir.name),
        "config": config,
        "metrics": {
            **metrics_req,
            **metrics_func,
            **metrics_amb,
        },
        "checkpoint_used": ckpt_path.name,
    }


def evaluate_baseline(
    samples: List[Sample],
    y_req: np.ndarray,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate baseline Phase-1 model"""
    
    if not BASELINE_ROOT.exists():
        print(f"[skip] Baseline not found at {BASELINE_ROOT}")
        return None
    
    # Get latest baseline run
    runs = sorted([d for d in BASELINE_ROOT.iterdir() if d.is_dir()])
    if not runs:
        print(f"[skip] No baseline runs found")
        return None
    
    baseline_dir = runs[-1]
    ckpt_path = baseline_dir / "best.pt"
    
    if not ckpt_path.exists():
        print(f"[skip] No baseline checkpoint at {ckpt_path}")
        return None
    
    print(f"\n[baseline] {baseline_dir.name}")
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Load encoder
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    
    # Load head
    head = BaselineHead(ckpt["head_state"]).to(DEVICE)
    head.eval()
    
    # Embed (sentence only)
    texts = [s.text for s in samples]
    embs = embed_batch(texts, tokenizer, encoder, batch_size)
    
    # Predict
    with torch.no_grad():
        logits = head(embs.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    
    metrics = compute_metrics(y_req, probs, "req")
    
    print(f"  F1={metrics.get('req_f1', 0):.4f}, Acc={metrics.get('req_accuracy', 0):.4f}")
    
    return {
        "model_dir": "baseline_phase1",
        "config": ckpt.get("config", {}),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate multi-task DeBERTa models")
    ap.add_argument("--test-data", type=str, default=str(TEST_DATA_PATH), help="Path to test data")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    ap.add_argument("--include-baseline", action="store_true", help="Include baseline Phase-1 model")
    args = ap.parse_args()
    
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"multitask_eval_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("[eval] Multi-Task Model Evaluation")
    print(f"{'='*70}\n")
    
    # Load test data
    print(f"[data] Loading test data from {args.test_data}")
    test_path = Path(args.test_data)
    samples, y_req, y_func, y_amb = load_test_data(test_path)
    
    if args.limit:
        samples = samples[:args.limit]
        y_req = y_req[:args.limit]
        y_func = y_func[:args.limit]
        y_amb = y_amb[:args.limit]
    
    print(f"[data] Loaded {len(samples)} test samples")
    print(f"  Task 1 (Req): {np.sum(~np.isnan(y_req)):.0f} labeled ({np.nansum(y_req):.0f} positive)")
    print(f"  Task 2 (F/NFR): {np.sum(~np.isnan(y_func)):.0f} labeled")
    print(f"  Task 3 (Amb): {np.sum(~np.isnan(y_amb)):.0f} labeled")
    
    # Find all trained models
    all_results = []
    
    if MODELS_ROOT.exists():
        # Find all run directories
        run_dirs = sorted([d for d in MODELS_ROOT.iterdir() if d.is_dir()])
        
        for run_dir in run_dirs:
            # Each run may contain multiple model types
            model_dirs = [d for d in run_dir.iterdir() if d.is_dir() and (d / "best.pt").exists() or (d / "last.pt").exists()]
            
            for model_dir in model_dirs:
                result = evaluate_multitask_model(
                    model_dir,
                    samples,
                    y_req,
                    y_func,
                    y_amb,
                    args.batch_size,
                )
                if result:
                    all_results.append(result)
    
    # Evaluate baseline if requested
    if args.include_baseline:
        baseline_result = evaluate_baseline(samples, y_req, args.batch_size)
        if baseline_result:
            all_results.append(baseline_result)
    
    if not all_results:
        print("\n[error] No models found to evaluate!")
        return
    
    # Create summary table
    print(f"\n{'='*70}")
    print("[summary] Evaluation Results")
    print(f"{'='*70}\n")
    
    summary_rows = []
    for result in all_results:
        config = result["config"]
        metrics = result["metrics"]
        
        summary_rows.append({
            "model": result["model_dir"],
            "architecture": config.get("model_type", config.get("model", "unknown")),
            "n_context": config.get("n_context", "N/A"),
            "lora_r": config.get("lora", {}).get("r", "N/A"),
            "lr": config.get("lr", "N/A"),
            "req_f1": metrics.get("req_f1", 0),
            "req_acc": metrics.get("req_accuracy", 0),
            "req_p": metrics.get("req_precision", 0),
            "req_r": metrics.get("req_recall", 0),
            "func_f1": metrics.get("func_f1", 0),
            "amb_f1": metrics.get("amb_f1", 0),
        })
    
    # Sort by Task 1 F1 (primary metric)
    summary_rows.sort(key=lambda x: x["req_f1"], reverse=True)
    
    # Print table
    print(f"{'Model':<40} {'Arch':<15} {'n':<3} {'LoRA':<5} {'Req F1':>8} {'Req Acc':>8} {'F/NFR F1':>9} {'Amb F1':>8}")
    print("-" * 120)
    
    for row in summary_rows:
        print(f"{row['model']:<40} "
              f"{row['architecture']:<15} "
              f"{str(row['n_context']):<3} "
              f"{str(row['lora_r']):<5} "
              f"{row['req_f1']:>8.4f} "
              f"{row['req_acc']:>8.4f} "
              f"{row['func_f1']:>9.4f} "
              f"{row['amb_f1']:>8.4f}")
    
    # Find best model
    best_model = summary_rows[0]
    print(f"\n[best] Best model (by Req F1): {best_model['model']}")
    print(f"  F1={best_model['req_f1']:.4f}, Acc={best_model['req_acc']:.4f}")
    
    # Save results
    results_json = {
        "metadata": {
            "timestamp": run_id,
            "test_data": str(args.test_data),
            "n_samples": len(samples),
            "n_models_evaluated": len(all_results),
        },
        "results": all_results,
        "summary": summary_rows,
        "best_model": best_model,
    }
    
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    
    # Save summary as CSV
    with open(out_dir / "summary_table.csv", "w", newline="", encoding="utf-8") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    
    print(f"\n{'='*70}")
    print(f"[done] Evaluation complete")
    print(f"[done] Results saved to: {out_dir}")
    print(f"  - evaluation_results.json (full details)")
    print(f"  - summary_table.csv (quick view)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()