#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate DeBERTa Full Fine-Tuned Model
---------------------------------------

Evaluates the full fine-tuned DeBERTa model on held-out test set.
Provides comprehensive metrics and comparison with baselines.

Usage:
    python eval_deberta_full_ft.py
    python eval_deberta_full_ft.py --test-data classifier/outputs/splits/test_20.jsonl
    python eval_deberta_full_ft.py --model-path classifier/models/deberta_full_finetune/20260213_010841_deberta-v3-base
"""

from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
BASELINE_PHASE1_ROOT = Path("classifier/models/mpnet_phase1")
BASELINE_PHASE15_ROOT = Path("classifier/models/mpnet_phase1_5_context_v2")

# Default test data paths (will try in order)
DEFAULT_TEST_PATHS = [
    Path("classifier/outputs/splits/train_40.jsonl"),
    # Path("classifier/outputs/splits/holdout_20.jsonl"),
    # Path("classifier/outputs/splits/holdout_60.jsonl"),  # Fallback to dev if no test
]

OUT_ROOT = Path("classifier/outputs/eval")

# ---------------------------------------------------------------------
# MODEL CLASSES
# ---------------------------------------------------------------------
class RequirementClassifier(nn.Module):
    """Full fine-tuning classifier (matches training architecture)"""
    
    def __init__(self, model_name: str, config_obj, freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config_obj = config_obj
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        if hasattr(self.encoder.config, 'hidden_size'):
            emb_dim = self.encoder.config.hidden_size
        else:
            emb_dim = 768
        
        # Classifier head (matches training)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, config_obj.HIDDEN_DIM_1),
            nn.LayerNorm(config_obj.HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(config_obj.DROPOUT_1),
            nn.Linear(config_obj.HIDDEN_DIM_1, config_obj.HIDDEN_DIM_2),
            nn.LayerNorm(config_obj.HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(config_obj.DROPOUT_2),
            nn.Linear(config_obj.HIDDEN_DIM_2, 2),  # Binary classification
        )
    
    def forward(self, input_ids, attention_mask):
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class BaselineHead(nn.Module):
    """Baseline model head (for Phase-1/Phase-1.5 comparison)"""
    
    def __init__(self, state_dict: dict):
        super().__init__()
        
        w0 = state_dict["net.0.weight"]
        in_dim = w0.shape[1]
        hidden1 = w0.shape[0]
        
        # Check architecture
        if "net.6.weight" in state_dict:
            # 3-layer (Phase-1.5)
            hidden2 = state_dict["net.3.weight"].shape[0]
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden2, 1),
            )
        elif "net.3.weight" in state_dict:
            # 2-layer (Phase-1)
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
def load_jsonl(path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Load JSONL file"""
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


def mean_pooling(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for sentence embeddings"""
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def embed_texts(texts: List[str], tokenizer, encoder, max_length: int = 256, batch_size: int = 32) -> torch.Tensor:
    """Embed texts in batches"""
    all_embs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(DEVICE)
        
        outputs = encoder(**enc)
        pooled = mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        
        all_embs.append(pooled.cpu())
    
    return torch.cat(all_embs, dim=0)


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
    
    # Confusion matrix
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
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }


def build_context_text(row: Dict[str, Any], n_context: int) -> str:
    """Build context text from row (same as training)"""
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


# ---------------------------------------------------------------------
# EVALUATION FUNCTIONS
# ---------------------------------------------------------------------
def evaluate_deberta_full_ft(
    model_path: Path,
    test_rows: List[Dict],
    y_true: np.ndarray,
    batch_size: int = 32,
    n_context_override: int = None
) -> Dict[str, Any]:
    """Evaluate DeBERTa full fine-tuned model"""
    
    print(f"\n{'='*70}")
    print(f"[eval] DeBERTa Full Fine-Tuned Model")
    print(f"{'='*70}")
    print(f"Model: {model_path.name}")
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Use override context size if provided, otherwise use model's default
    n_context = n_context_override if n_context_override is not None else config.get('N_CONTEXT', 2)
    
    print(f"Config: {config.get('MODEL_NAME', 'unknown')}")
    print(f"Context: n={n_context}")
    print(f"Max Length: {config.get('MAX_LENGTH', 256)}")
    
    # Load checkpoint
    ckpt_path = model_path / "best.pt"
    if not ckpt_path.exists():
        print(f"[warn] best.pt not found, using last.pt")
        ckpt_path = model_path / "last.pt"
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    # Load encoder
    model_name = config.get("MODEL_NAME", "microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get embedding dimension from config
    if hasattr(AutoModel.from_pretrained(model_name).config, 'hidden_size'):
        emb_dim = AutoModel.from_pretrained(model_name).config.hidden_size
    else:
        emb_dim = 768
    
    # Create full model (encoder + classifier)
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
    
    # Load the entire model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"[OK] Model loaded successfully")
    print(f"Parameters: {emb_dim} embedding dim")
    
    # Use the model's encoder for encoding
    encoder = model.encoder
    classifier = model.classifier
    
    # Build context texts (using n_context parameter)
    max_length = config.get("MAX_LENGTH", 256)
    
    print(f"\n[embed] Building context texts (n={n_context})...")
    ctx_texts = [build_context_text(r, n_context) for r in test_rows]
    
    # Predict
    print(f"[predict] Running inference on {len(ctx_texts)} samples...")
    all_probs = []
    all_preds = []
    all_predictions_detailed = []  # Store detailed predictions
    
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
        ).to(DEVICE)
        
        # Forward pass through full model
        with torch.no_grad():  # Disable gradients for inference
            logits = model(enc["input_ids"], enc["attention_mask"])
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # Both class probs
        prob_positive = probs[:, 1]  # Prob of class 1 (requirement)
        preds = (prob_positive >= 0.5).astype(int)
        
        # Store detailed predictions for each sample
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
        "model": f"DeBERTa-base (Full FT, n={n_context})",
        "model_path": str(model_path),
        "config": config,
        "n_context": n_context,
        "metrics": metrics,
        "predictions": all_predictions_detailed,  # Include detailed predictions
    }


def evaluate_baseline(
    baseline_path: Path,
    test_rows: List[Dict],
    y_true: np.ndarray,
    baseline_name: str,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Evaluate baseline model (Phase-1 or Phase-1.5)"""
    
    print(f"\n{'='*70}")
    print(f"[eval] Baseline: {baseline_name}")
    print(f"{'='*70}")
    print(f"Model: {baseline_path.name}")
    
    # Load checkpoint
    ckpt_path = baseline_path / "best.pt"
    if not ckpt_path.exists():
        print(f"[skip] No checkpoint found")
        return None
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    # Load encoder (MPNet for baselines)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    
    # Load head
    head = BaselineHead(checkpoint["head_state"]).to(DEVICE)
    head.eval()
    
    # Check expected input dimension
    expected_dim = checkpoint["head_state"]["net.0.weight"].shape[1]
    emb_dim = 768  # MPNet embedding dimension
    
    print(f"[model] Expected input dim: {expected_dim}")
    
    # Determine if this is Phase-1 (768) or Phase-1.5 (2305)
    is_phase_15 = expected_dim > 1000  # Phase-1.5 uses context features
    
    if is_phase_15:
        print(f"[model] Detected Phase-1.5 (context features required)")
        
        # Extract sentences and build context
        sentences = [r["sentence"] for r in test_rows]
        
        # Build context texts (n=2, same as Phase-1.5 training)
        n_context = 2
        ctx_texts = []
        for r in test_rows:
            before = (r.get("context_before") or [])[-n_context:]
            after = (r.get("context_after") or [])[:n_context]
            sent = r.get("sentence", "").strip()
            
            parts = []
            if before:
                parts.append(" ".join(b.strip() for b in before if isinstance(b, str) and b.strip()))
            parts.append(sent)
            if after:
                parts.append(" ".join(a.strip() for a in after if isinstance(a, str) and a.strip()))
            
            ctx_texts.append(" ".join([p for p in parts if p]))
        
        print(f"[embed] Encoding sentences and context...")
        E_sent = embed_texts(sentences, tokenizer, encoder, max_length=128, batch_size=batch_size)
        E_ctx = embed_texts(ctx_texts, tokenizer, encoder, max_length=128, batch_size=batch_size)
        
        # Build Phase-1.5 features: [E_sent, E_ctx, abs_diff, cos]
        abs_diff = torch.abs(E_sent - E_ctx)
        cos = nn.functional.cosine_similarity(E_sent, E_ctx, dim=1).unsqueeze(1)
        embeddings = torch.cat([E_sent, E_ctx, abs_diff, cos], dim=1)
        
        print(f"[features] Built Phase-1.5 features: {embeddings.shape[1]} dims")
        
    else:
        print(f"[model] Detected Phase-1 (sentence embeddings only)")
        
        # Extract sentences only (no context for Phase-1)
        sentences = [r["sentence"] for r in test_rows]
        
        print(f"[embed] Encoding {len(sentences)} samples...")
        embeddings = embed_texts(sentences, tokenizer, encoder, max_length=128, batch_size=batch_size)
    
    # Predict
    print(f"[predict] Running inference...")
    with torch.no_grad():
        logits = head(embeddings.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    
    preds = (probs >= 0.5).astype(int)
    
    # Compute metrics
    metrics = compute_metrics(y_true, preds, probs)
    
    print(f"\n[results]")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1']*100:.2f}%")
    
    return {
        "model": baseline_name,
        "model_path": str(baseline_path),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate DeBERTa Full Fine-Tuned Model")
    ap.add_argument("--test-data", type=str, default=None, help="Path to test data JSONL")
    ap.add_argument("--model-path", type=str, default=None, help="Path to specific model checkpoint")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Limit test samples")
    ap.add_argument("--include-baselines", action="store_true", help="Also evaluate baseline models")
    ap.add_argument("--context-sizes", type=str, default="2", help="Comma-separated context sizes (e.g., '1,2,3')")
    ap.add_argument("--comprehensive", action="store_true", help="Run comprehensive evaluation with multiple context sizes")
    args = ap.parse_args()
    
    run_id = time.strftime("%Y%m%d_%H%M%S")
    
    # Use different output directory for comprehensive eval
    if args.comprehensive:
        out_dir = OUT_ROOT / f"deberta_comprehensive_{run_id}"
    else:
        out_dir = OUT_ROOT / f"deberta_ft_eval_{run_id}"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    if args.comprehensive:
        print("[eval] DeBERTa Comprehensive Evaluation (Multiple Context Sizes)")
    else:
        print("[eval] DeBERTa Full Fine-Tuned Model Evaluation")
    print(f"{'='*70}\n")
    
    # ================================================================
    # 1. LOAD TEST DATA
    # ================================================================
    test_path = None
    if args.test_data:
        test_path = Path(args.test_data)
    else:
        # Try default paths in order
        for p in DEFAULT_TEST_PATHS:
            if p.exists():
                test_path = p
                break
    
    if not test_path or not test_path.exists():
        print(f"[error] No test data found!")
        print(f"Tried: {[str(p) for p in DEFAULT_TEST_PATHS]}")
        return
    
    print(f"[data] Loading test data from: {test_path}")
    test_rows = load_jsonl(test_path, args.limit)
    
    if not test_rows:
        print(f"[error] No data loaded from {test_path}")
        return
    
    # Extract labels from llm_labels field
    # Requirement if "requirement" OR "with_context" is in llm_labels
    # Non-requirement if "non_requirement" is in llm_labels
    y_true = []
    for r in test_rows:
        labels = set(r.get("llm_labels", []))
        
        # Check for requirement indicators
        if "requirement" in labels or "with_context" in labels:
            y_true.append(1)  # Requirement
        elif "non_requirement" in labels:
            y_true.append(0)  # Non-requirement
        else:
            # Skip samples without clear labels
            continue
    
    y_true = np.array(y_true)
    
    # Filter test_rows to match y_true length (remove unlabeled samples)
    filtered_rows = []
    for r in test_rows:
        labels = set(r.get("llm_labels", []))
        if "requirement" in labels or "with_context" in labels or "non_requirement" in labels:
            filtered_rows.append(r)
    
    test_rows = filtered_rows
    
    print(f"[data] Loaded {len(test_rows)} samples")
    
    # Analyze label distribution
    req_count = sum(1 for r in test_rows if "requirement" in r.get("llm_labels", []))
    with_ctx_count = sum(1 for r in test_rows if "with_context" in r.get("llm_labels", []))
    non_req_count = sum(1 for r in test_rows if "non_requirement" in r.get("llm_labels", []))
    
    # Count F/NFR labels (for reference, not used in evaluation)
    functional_count = sum(1 for r in test_rows if "functional" in r.get("llm_labels", []))
    non_functional_count = sum(1 for r in test_rows if "non_functional" in r.get("llm_labels", []))
    
    print(f"\nLabel Distribution:")
    print(f"  'requirement':     {req_count:4d}")
    print(f"  'with_context':    {with_ctx_count:4d}")
    print(f"  'non_requirement': {non_req_count:4d}")
    print(f"\nF/NFR Labels (reference only, not evaluated):")
    print(f"  'functional':      {functional_count:4d}")
    print(f"  'non_functional':  {non_functional_count:4d}")
    
    print(f"\nFinal Classification:")
    print(f"  Positive (requirement OR with_context): {y_true.sum()} ({y_true.sum()/len(y_true)*100:.1f}%)")
    print(f"  Negative (non_requirement):             {len(y_true) - y_true.sum()} ({(len(y_true)-y_true.sum())/len(y_true)*100:.1f}%)")
    
    # ================================================================
    # 2. FIND MODEL TO EVALUATE
    # ================================================================
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Find latest DeBERTa model
        if not DEBERTA_FT_ROOT.exists():
            print(f"[error] No models found in {DEBERTA_FT_ROOT}")
            return
        
        models = sorted([d for d in DEBERTA_FT_ROOT.iterdir() if d.is_dir() and (d / "best.pt").exists()])
        if not models:
            print(f"[error] No trained models found in {DEBERTA_FT_ROOT}")
            return
        
        model_path = models[-1]
        print(f"[auto] Using latest model: {model_path.name}")
    
    # ================================================================
    # 3. EVALUATE DEBERTA MODEL
    # ================================================================
    results = []
    
    # Parse context sizes for comprehensive eval
    if args.comprehensive:
        context_sizes = [int(n.strip()) for n in args.context_sizes.split(",")]
        print(f"\n[config] Testing context sizes: {context_sizes}")
        
        # Evaluate with each context size
        for n_ctx in context_sizes:
            deberta_result = evaluate_deberta_full_ft(
                model_path, test_rows, y_true, args.batch_size, n_context_override=n_ctx
            )
            results.append(deberta_result)
    else:
        # Single evaluation with model's default context
        deberta_result = evaluate_deberta_full_ft(
            model_path, test_rows, y_true, args.batch_size
        )
        results.append(deberta_result)
    
    # ================================================================
    # 4. EVALUATE BASELINES (OPTIONAL)
    # ================================================================
    if args.include_baselines:
        # Phase-1
        if BASELINE_PHASE1_ROOT.exists():
            phase1_models = sorted([d for d in BASELINE_PHASE1_ROOT.iterdir() if d.is_dir()])
            if phase1_models:
                phase1_result = evaluate_baseline(
                    phase1_models[-1], test_rows, y_true, "MPNet Phase-1", args.batch_size
                )
                if phase1_result:
                    results.append(phase1_result)
        
        # Phase-1.5
        if BASELINE_PHASE15_ROOT.exists():
            phase15_models = sorted([d for d in BASELINE_PHASE15_ROOT.iterdir() if d.is_dir()])
            if phase15_models:
                phase15_result = evaluate_baseline(
                    phase15_models[-1], test_rows, y_true, "MPNet Phase-1.5", args.batch_size
                )
                if phase15_result:
                    results.append(phase15_result)
    
    # ================================================================
    # 5. SUMMARY TABLE
    # ================================================================
    print(f"\n{'='*70}")
    print("[summary] Evaluation Results Comparison")
    print(f"{'='*70}\n")
    
    # Print table
    print(f"{'Model':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC':>8}")
    print("-" * 80)
    
    for result in results:
        m = result["metrics"]
        print(f"{result['model']:<30} "
              f"{m['accuracy']*100:>7.2f}% "
              f"{m['precision']*100:>7.2f}% "
              f"{m['recall']*100:>7.2f}% "
              f"{m['f1']*100:>7.2f}% "
              f"{m['roc_auc']:>8.4f}")
    
    # Calculate improvements
    if len(results) > 1:
        deberta_f1 = results[0]["metrics"]["f1"]
        baseline_f1 = results[1]["metrics"]["f1"]
        improvement = (deberta_f1 - baseline_f1) * 100
        
        print(f"\n{'='*70}")
        print(f"[improvement] DeBERTa vs Baseline:")
        print(f"  F1 Improvement: +{improvement:.2f} percentage points")
        print(f"  Relative Improvement: +{(deberta_f1/baseline_f1 - 1)*100:.1f}%")
        print(f"{'='*70}")
    
    # ================================================================
    # 6. SAVE RESULTS
    # ================================================================
    output_data = {
        "metadata": {
            "timestamp": run_id,
            "test_data": str(test_path),
            "n_samples": len(test_rows),
            "n_positive": int(y_true.sum()),
            "n_negative": int(len(y_true) - y_true.sum()),
            "comprehensive": args.comprehensive,
            "context_sizes": [r.get("n_context", 2) for r in results if "predictions" in r],
        },
        "results": [
            {
                "model": r["model"],
                "n_context": r.get("n_context", 2),
                "metrics": r["metrics"],
            }
            for r in results
        ],
    }
    
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  - evaluation_results.json (full results)")
    
    # Save individual predictions
    if args.comprehensive:
        # Save predictions for each context size separately
        for result in results:
            if "predictions" not in result:
                continue
            
            n_ctx = result.get("n_context", 2)
            
            # Detailed predictions
            pred_file = out_dir / f"predictions_n{n_ctx}.json"
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump({
                    "n_context": n_ctx,
                    "total_samples": len(result["predictions"]),
                    "correct": sum(1 for p in result["predictions"] if p["correct"]),
                    "incorrect": sum(1 for p in result["predictions"] if not p["correct"]),
                    "metrics": result["metrics"],
                    "predictions": result["predictions"],
                }, f, indent=2)
            
            print(f"  - predictions_n{n_ctx}.json ({len(result['predictions'])} samples)")
            
            # Error analysis
            errors = [p for p in result["predictions"] if not p["correct"]]
            if errors:
                errors_file = out_dir / f"errors_n{n_ctx}.json"
                with open(errors_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "n_context": n_ctx,
                        "total_errors": len(errors),
                        "false_positives": sum(1 for e in errors if e["predicted_label"] == 1),
                        "false_negatives": sum(1 for e in errors if e["predicted_label"] == 0),
                        "errors": errors,
                    }, f, indent=2)
                
                print(f"  - errors_n{n_ctx}.json ({len(errors)} errors)")
    else:
        # Standard single-evaluation saving
        if results and "predictions" in results[0]:
            predictions_output = {
                "metadata": output_data["metadata"],
                "model": results[0]["model"],
                "total_samples": len(results[0]["predictions"]),
                "correct": sum(1 for p in results[0]["predictions"] if p["correct"]),
                "incorrect": sum(1 for p in results[0]["predictions"] if not p["correct"]),
                "predictions": results[0]["predictions"],
            }
            
            with open(out_dir / "predictions_detailed.json", "w", encoding="utf-8") as f:
                json.dump(predictions_output, f, indent=2)
            
            print(f"  - predictions_detailed.json (individual predictions)")
            
            # Save error analysis (incorrect predictions only)
            errors = [p for p in results[0]["predictions"] if not p["correct"]]
            if errors:
                with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "total_errors": len(errors),
                        "false_positives": sum(1 for e in errors if e["predicted_label"] == 1),
                        "false_negatives": sum(1 for e in errors if e["predicted_label"] == 0),
                        "errors": errors,
                    }, f, indent=2)
                print(f"  - errors.json ({len(errors)} incorrect predictions)")
    
    # Save CSV
    import csv
    summary_rows = []
    for result in results:
        m = result["metrics"]
        summary_rows.append({
            "model": result["model"],
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
    
    with open(out_dir / "summary_table.csv", "w", newline="", encoding="utf-8") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    
    print(f"\n{'='*70}")
    print(f"[done] Evaluation complete!")
    print(f"[done] Results saved to: {out_dir}")
    print(f"  - evaluation_results.json (full results)")
    print(f"  - summary_table.csv (comparison table)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()