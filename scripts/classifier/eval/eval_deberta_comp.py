#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive DeBERTa Evaluation with Multiple Context Windows
---------------------------------------------------------------

Evaluates DeBERTa model with different context window sizes (n=1,2,3)
and saves detailed predictions for each configuration.

Evaluation Strategy:
- Positive (Requirement): "requirement" OR "with_context" in llm_labels
- Negative (Non-requirement): "non_requirement" in llm_labels
- Tests: n=1, n=2, n=3 context windows
- Outputs: Separate prediction files for each configuration

Usage:
    python eval_deberta_comprehensive.py
    python eval_deberta_comprehensive.py --test-data path/to/test.jsonl
"""

from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[init] project root: {PROJECT_ROOT}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using: {DEVICE}")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
DEBERTA_FT_ROOT = Path("classifier/models/deberta_full_finetune")
OUT_ROOT = Path("classifier/outputs/eval")

DEFAULT_TEST_PATHS = [
    Path("classifier/outputs/splits/train_40.jsonl"),
]

# ---------------------------------------------------------------------
# MODEL CLASS
# ---------------------------------------------------------------------
def mean_pooling(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class RequirementClassifier(nn.Module):
    """Full fine-tuning classifier"""
    
    def __init__(self, model_name: str, config_obj, freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config_obj = config_obj
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if hasattr(self.encoder.config, 'hidden_size'):
            emb_dim = self.encoder.config.hidden_size
        else:
            emb_dim = 768
        
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, config_obj.HIDDEN_DIM_1),
            nn.LayerNorm(config_obj.HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(config_obj.DROPOUT_1),
            nn.Linear(config_obj.HIDDEN_DIM_1, config_obj.HIDDEN_DIM_2),
            nn.LayerNorm(config_obj.HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(config_obj.DROPOUT_2),
            nn.Linear(config_obj.HIDDEN_DIM_2, 2),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        logits = self.classifier(pooled)
        return logits


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def load_jsonl(path: Path, limit: int = None) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_context_text(row: Dict[str, Any], n_context: int) -> str:
    """Build context text with n sentences before/after"""
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics"""
    
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        roc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    except:
        roc = float("nan")
    
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except:
        pr_auc = float("nan")
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
def evaluate_with_context(
    model,
    tokenizer,
    test_rows: List[Dict],
    y_true: np.ndarray,
    n_context: int,
    max_length: int,
    batch_size: int,
    device
) -> Dict[str, Any]:
    """Evaluate model with specific context window size"""
    
    print(f"\n{'='*70}")
    print(f"[eval] Context Window: n={n_context}")
    print(f"{'='*70}")
    
    # Build context texts
    print(f"[embed] Building context texts (n={n_context})...")
    ctx_texts = [build_context_text(r, n_context) for r in test_rows]
    
    # Predict
    print(f"[predict] Running inference on {len(ctx_texts)} samples...")
    all_probs = []
    all_preds = []
    all_predictions_detailed = []
    
    model.eval()
    
    for i in range(0, len(ctx_texts), batch_size):
        batch_texts = ctx_texts[i:i + batch_size]
        batch_rows = test_rows[i:i + batch_size]
        
        # Tokenize
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        prob_positive = probs[:, 1]
        preds = (prob_positive >= 0.5).astype(int)
        
        # Store detailed predictions
        for j, (row, pred, prob_pos, prob_neg) in enumerate(zip(batch_rows, preds, prob_positive, probs[:, 0])):
            all_predictions_detailed.append({
                "sent_id": row.get("sent_id", f"sample_{i+j}"),
                "sentence": row.get("sentence", ""),
                "context_before": row.get("context_before", [])[-n_context:] if n_context > 0 else [],
                "context_after": row.get("context_after", [])[:n_context] if n_context > 0 else [],
                "true_label": int(y_true[i+j]),
                "true_labels": row.get("llm_labels", []),
                "predicted_label": int(pred),
                "predicted_class": "requirement" if pred == 1 else "non_requirement",
                "prob_requirement": float(prob_pos),
                "prob_non_requirement": float(prob_neg),
                "confidence": float(max(prob_pos, prob_neg)),
                "correct": bool(pred == y_true[i+j]),
                "n_context": n_context,
            })
        
        all_probs.extend(prob_positive)
        all_preds.extend(preds)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # Compute metrics
    metrics = compute_metrics(y_true, all_preds, all_probs)
    
    print(f"\n[results]")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1']*100:.2f}%")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']:4d}  FP: {metrics['fp']:4d}")
    print(f"  FN: {metrics['fn']:4d}  TN: {metrics['tn']:4d}")
    
    return {
        "n_context": n_context,
        "metrics": metrics,
        "predictions": all_predictions_detailed,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Comprehensive DeBERTa Evaluation")
    ap.add_argument("--test-data", type=str, default=None)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--context-sizes", type=str, default="1,2,3", help="Comma-separated context sizes")
    args = ap.parse_args()
    
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"deberta_comprehensive_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("[eval] DeBERTa Comprehensive Evaluation")
    print(f"{'='*70}\n")
    
    # Parse context sizes
    context_sizes = [int(n.strip()) for n in args.context_sizes.split(",")]
    print(f"[config] Testing context sizes: {context_sizes}")
    
    # ================================================================
    # 1. LOAD TEST DATA
    # ================================================================
    test_path = None
    if args.test_data:
        test_path = Path(args.test_data)
    else:
        for p in DEFAULT_TEST_PATHS:
            if p.exists():
                test_path = p
                break
    
    if not test_path or not test_path.exists():
        print(f"[error] No test data found!")
        return
    
    print(f"[data] Loading test data from: {test_path}")
    test_rows = load_jsonl(test_path, args.limit)
    
    if not test_rows:
        print(f"[error] No data loaded")
        return
    
    # Extract labels
    y_true = []
    filtered_rows = []
    
    for r in test_rows:
        labels = set(r.get("llm_labels", []))
        
        if "requirement" in labels or "with_context" in labels:
            y_true.append(1)
            filtered_rows.append(r)
        elif "non_requirement" in labels:
            y_true.append(0)
            filtered_rows.append(r)
    
    test_rows = filtered_rows
    y_true = np.array(y_true)
    
    print(f"[data] Loaded {len(test_rows)} samples")
    
    # Analyze labels
    req_count = sum(1 for r in test_rows if "requirement" in r.get("llm_labels", []))
    with_ctx_count = sum(1 for r in test_rows if "with_context" in r.get("llm_labels", []))
    non_req_count = sum(1 for r in test_rows if "non_requirement" in r.get("llm_labels", []))
    
    print(f"\nLabel Distribution:")
    print(f"  'requirement':     {req_count:4d}")
    print(f"  'with_context':    {with_ctx_count:4d}")
    print(f"  'non_requirement': {non_req_count:4d}")
    print(f"\nFinal Classification:")
    print(f"  Positive: {y_true.sum()} ({y_true.sum()/len(y_true)*100:.1f}%)")
    print(f"  Negative: {len(y_true) - y_true.sum()} ({(len(y_true)-y_true.sum())/len(y_true)*100:.1f}%)")
    
    # ================================================================
    # 2. LOAD MODEL
    # ================================================================
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        if not DEBERTA_FT_ROOT.exists():
            print(f"[error] No models found")
            return
        
        models = sorted([d for d in DEBERTA_FT_ROOT.iterdir() if d.is_dir() and (d / "best.pt").exists()])
        if not models:
            print(f"[error] No trained models found")
            return
        
        model_path = models[-1]
    
    print(f"\n[model] Using: {model_path.name}")
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    model_name = config.get("MODEL_NAME", "microsoft/deberta-v3-base")
    max_length = config.get("MAX_LENGTH", 256)
    
    print(f"[model] Base: {model_name}")
    print(f"[model] Max Length: {max_length}")
    
    # Load checkpoint
    ckpt_path = model_path / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = RequirementClassifier(
        model_name=model_name,
        config_obj=type('Config', (), {
            'HIDDEN_DIM_1': config.get("HIDDEN_DIM_1", 512),
            'HIDDEN_DIM_2': config.get("HIDDEN_DIM_2", 256),
            'DROPOUT_1': config.get("DROPOUT_1", 0.3),
            'DROPOUT_2': config.get("DROPOUT_2", 0.2),
        })(),
        freeze_encoder=False
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"[OK] Model loaded successfully")
    
    # ================================================================
    # 3. EVALUATE WITH DIFFERENT CONTEXT SIZES
    # ================================================================
    results = []
    
    for n in context_sizes:
        result = evaluate_with_context(
            model=model,
            tokenizer=tokenizer,
            test_rows=test_rows,
            y_true=y_true,
            n_context=n,
            max_length=max_length,
            batch_size=args.batch_size,
            device=DEVICE
        )
        results.append(result)
    
    # ================================================================
    # 4. SUMMARY TABLE
    # ================================================================
    print(f"\n{'='*70}")
    print("[summary] Results Across Context Sizes")
    print(f"{'='*70}\n")
    
    print(f"{'n_ctx':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC':>8}")
    print("-" * 60)
    
    for result in results:
        m = result["metrics"]
        print(f"{result['n_context']:>6} "
              f"{m['accuracy']*100:>7.2f}% "
              f"{m['precision']*100:>7.2f}% "
              f"{m['recall']*100:>7.2f}% "
              f"{m['f1']*100:>7.2f}% "
              f"{m['roc_auc']:>8.4f}")
    
    # ================================================================
    # 5. SAVE RESULTS
    # ================================================================
    print(f"\n{'='*70}")
    print("[save] Saving results...")
    print(f"{'='*70}\n")
    
    # Save overall summary
    summary_data = {
        "metadata": {
            "timestamp": run_id,
            "test_data": str(test_path),
            "model_path": str(model_path),
            "n_samples": len(test_rows),
            "n_positive": int(y_true.sum()),
            "n_negative": int(len(y_true) - y_true.sum()),
            "context_sizes": context_sizes,
        },
        "results": [
            {
                "n_context": r["n_context"],
                "metrics": r["metrics"],
            }
            for r in results
        ]
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"  ✓ summary.json")
    
    # Save detailed predictions for each context size
    for result in results:
        n = result["n_context"]
        
        # Predictions file
        pred_file = out_dir / f"predictions_n{n}.json"
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({
                "n_context": n,
                "total_samples": len(result["predictions"]),
                "correct": sum(1 for p in result["predictions"] if p["correct"]),
                "incorrect": sum(1 for p in result["predictions"] if not p["correct"]),
                "metrics": result["metrics"],
                "predictions": result["predictions"],
            }, f, indent=2)
        
        print(f"  ✓ predictions_n{n}.json ({len(result['predictions'])} samples)")
        
        # Errors file
        errors = [p for p in result["predictions"] if not p["correct"]]
        if errors:
            errors_file = out_dir / f"errors_n{n}.json"
            with open(errors_file, "w", encoding="utf-8") as f:
                json.dump({
                    "n_context": n,
                    "total_errors": len(errors),
                    "false_positives": sum(1 for e in errors if e["predicted_label"] == 1),
                    "false_negatives": sum(1 for e in errors if e["predicted_label"] == 0),
                    "errors": errors,
                }, f, indent=2)
            
            print(f"  ✓ errors_n{n}.json ({len(errors)} errors)")
    
    # Save CSV comparison
    import csv
    csv_file = out_dir / "comparison.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_context", "accuracy", "precision", "recall", "f1", 
            "roc_auc", "tp", "fp", "tn", "fn"
        ])
        writer.writeheader()
        
        for result in results:
            m = result["metrics"]
            writer.writerow({
                "n_context": result["n_context"],
                "accuracy": f"{m['accuracy']*100:.2f}",
                "precision": f"{m['precision']*100:.2f}",
                "recall": f"{m['recall']*100:.2f}",
                "f1": f"{m['f1']*100:.2f}",
                "roc_auc": f"{m['roc_auc']:.4f}",
                "tp": m['tp'],
                "fp": m['fp'],
                "tn": m['tn'],
                "fn": m['fn'],
            })
    
    print(f"  ✓ comparison.csv")
    
    print(f"\n{'='*70}")
    print(f"[done] Evaluation complete!")
    print(f"[done] Results saved to: {out_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()