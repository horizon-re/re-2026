#!/usr/bin/env python3
"""
req_pipeline — Multi-Task DeBERTa Classifier (LoRA)
------------------------------------------------

Multi-task learning for requirement classification:
  Task 1 (Primary): Requirement vs Non-Requirement
  Task 2 (Secondary): Functional vs Non-Functional (for requirements)
  Task 3 (Tertiary): Ambiguity detection

Data Sources:
  1. classifier/outputs/splits/holdout_60.jsonl - contextual requirements
  2. classifier/outputs/splits_frnfr/train.jsonl & dev.jsonl - F/NFR labels
  3. classifier/synthetic/non_requirements/synthetic_non_requirements.jsonl - non-requirements

Models to try:
  - microsoft/deberta-v3-base (primary)
  - sentence-transformers/all-MiniLM-L12-v2
  - microsoft/mpnet-base
  - sentence-transformers/paraphrase-mpnet-base-v2
"""

from __future__ import annotations

import os, sys, json, time, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
print(f"[init] Working directory: {PROJECT_ROOT}")

# Data paths
DATA_CONTEXTUAL = Path("classifier/outputs/splits/holdout_60.jsonl")
DATA_FRNFR_TRAIN = Path("classifier/outputs/splits_frnfr/train.jsonl")
DATA_FRNFR_DEV = Path("classifier/outputs/splits_frnfr/dev.jsonl")
DATA_SYNTH_NR = Path("classifier/synthetic/non_requirements/synthetic_non_requirements.jsonl")

OUT_ROOT = Path("classifier/models/multitask_deberta_lora")

MAX_LENGTH = 128

# Model options
MODEL_OPTIONS = {
    "deberta": "microsoft/deberta-v3-base",
    "minilm": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet": "microsoft/mpnet-base",
    "paraphrase-mpnet": "sentence-transformers/paraphrase-mpnet-base-v2",
}

# LoRA target modules per model type
LORA_TARGETS = {
    "deberta": ["query_proj", "key_proj", "value_proj"],  # DeBERTa v3
    "minilm": ["q", "k", "v"],  # MiniLM (BERT-based)
    "mpnet": ["q", "k", "v"],  # MPNet
    "paraphrase-mpnet": ["q", "k", "v"],  # Paraphrase MPNet
}

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        print(f"[warn] File not found: {p}, returning empty list")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def mean_pooling(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

# ---------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------
@dataclass
class Sample:
    """Unified sample for multi-task learning"""
    text: str
    is_requirement: Optional[int] = None  # Task 1: 0=non-req, 1=req
    is_functional: Optional[int] = None   # Task 2: 0=NFR, 1=FR (only for requirements)
    is_ambiguous: Optional[int] = None    # Task 3: 0=non-ambiguous, 1=ambiguous
    source: str = "unknown"
    
    def __repr__(self):
        return f"Sample(req={self.is_requirement}, func={self.is_functional}, amb={self.is_ambiguous}, src={self.source})"

# ---------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------
def load_contextual_data(path: Path, n_context: int = 1) -> List[Sample]:
    """Load from holdout_60.jsonl with context"""
    rows = load_jsonl(path)
    samples = []
    
    for r in rows:
        labels = set(r.get("llm_labels", []) or [])
        if not labels.intersection({"requirement", "non_requirement", "with_context"}):
            continue
            
        # Build context
        before = (r.get("context_before") or [])[-n_context:]
        after = (r.get("context_after") or [])[:n_context]
        sent = (r.get("sentence") or "").strip()
        
        parts = []
        if before:
            parts.append(" ".join(b.strip() for b in before if isinstance(b, str) and b.strip()))
        parts.append(sent)
        if after:
            parts.append(" ".join(a.strip() for a in after if isinstance(a, str) and a.strip()))
        
        text = " [CTX] ".join([p for p in parts if p])
        
        # Task 1: requirement classification
        is_req = 1 if ("requirement" in labels or "with_context" in labels) else 0
        
        # Task 3: ambiguity (from llm_labels)
        is_amb = 1 if "ambiguous" in labels else 0 if "non_ambiguous" in labels else None
        
        samples.append(Sample(
            text=text,
            is_requirement=is_req,
            is_functional=None,  # No F/NFR info in this dataset
            is_ambiguous=is_amb,
            source="contextual"
        ))
    
    return samples

def load_frnfr_data(path: Path) -> List[Sample]:
    """Load F/NFR labeled data (train.jsonl or dev.jsonl)"""
    rows = load_jsonl(path)
    samples = []
    
    for r in rows:
        text = r.get("text", "").strip()
        label = r.get("label", "").strip()
        
        if not text or not label:
            continue
            
        # These are all requirements, classified as F or NFR
        is_func = 1 if label == "F" else 0 if label in ["NFR", "NF"] else None
        
        if is_func is not None:
            samples.append(Sample(
                text=text,
                is_requirement=1,  # All are requirements
                is_functional=is_func,
                is_ambiguous=None,  # No ambiguity labels here
                source="frnfr"
            ))
    
    return samples

def load_synthetic_nr(path: Path) -> List[Sample]:
    rows = load_jsonl(path)
    samples = []

    for r in rows:
        raw_text = r.get("text", "")

        # Normalize text field
        if isinstance(raw_text, str):
            text = raw_text
        elif isinstance(raw_text, dict):
            text = (
                raw_text.get("text")
                or raw_text.get("sentence")
                or raw_text.get("content")
                or ""
            )
        else:
            text = ""

        text = text.strip()
        label = r.get("label", "").strip()

        if not text or label != "non_requirement":
            continue

        samples.append(Sample(
            text=text,
            is_requirement=0,
            is_functional=None,
            is_ambiguous=None,
            source="synthetic"
        ))

    return samples


# ---------------------------------------------------------------------
# IMPROVED CLASSIFIER HEAD
# ---------------------------------------------------------------------
class MultiTaskHead(nn.Module):
    """
    Multi-task head with improved architecture:
    - Task 1 (Primary): Requirement vs Non-Requirement
    - Task 2 (Secondary): Functional vs Non-Functional
    - Task 3 (Tertiary): Ambiguity detection
    """
    def __init__(self, in_dim: int = 768):
        super().__init__()
        
        # Shared layers (deeper, better regularization)
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
        
        # Task-specific heads
        self.req_head = nn.Linear(256, 1)      # Task 1: Requirement classification
        self.func_head = nn.Linear(256, 1)     # Task 2: F vs NFR
        self.amb_head = nn.Linear(256, 1)      # Task 3: Ambiguity
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_features = self.shared(x)
        
        req_logits = self.req_head(shared_features).squeeze(-1)
        func_logits = self.func_head(shared_features).squeeze(-1)
        amb_logits = self.amb_head(shared_features).squeeze(-1)
        
        return req_logits, func_logits, amb_logits

# ---------------------------------------------------------------------
# COMPUTE METRICS
# ---------------------------------------------------------------------
def compute_metrics_single_task(y_true: np.ndarray, probs: np.ndarray, task_name: str) -> Dict[str, float]:
    """Compute metrics for a single task"""
    # Filter out None values
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
        pr = float(average_precision_score(y_true, probs))
    except:
        pr = float("nan")
    
    return {
        f"{task_name}_n": int(valid_idx.sum()),
        f"{task_name}_acc": float(acc),
        f"{task_name}_p": float(p),
        f"{task_name}_r": float(r),
        f"{task_name}_f1": float(f1),
        f"{task_name}_roc": float(roc),
        f"{task_name}_pr": float(pr),
    }

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------
def train_multitask(
    samples_train: List[Sample],
    samples_dev: List[Sample],
    args,
    out_dir: Path,
    model_type: str,
):
    device = torch.device(args.device)
    ensure_dir(out_dir)
    
    # Model name and LoRA targets
    model_name = MODEL_OPTIONS[model_type]
    lora_targets = LORA_TARGETS[model_type]
    
    print(f"\n{'='*70}")
    print(f"[model] {model_name}")
    print(f"[lora] targets: {lora_targets}")
    print(f"{'='*70}\n")
    
    # Tokenizer + encoder
    # tok = AutoTokenizer.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    base_encoder = AutoModel.from_pretrained(model_name)
    
    # Get embedding dimension
    if hasattr(base_encoder.config, 'hidden_size'):
        emb_dim = base_encoder.config.hidden_size
    else:
        emb_dim = 768  # fallback
    
    print(f"[encoder] Embedding dimension: {emb_dim}")
    
    # LoRA config
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    
    encoder = get_peft_model(base_encoder, lora_cfg)
    encoder.to(device)
    
    head = MultiTaskHead(in_dim=emb_dim).to(device)
    
    # Optimizer
    params = list(p for p in encoder.parameters() if p.requires_grad) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss functions (with class weighting for imbalanced data)
    # Count positive samples for task 1
    n_req = sum(1 for s in samples_train if s.is_requirement == 1)
    n_non_req = sum(1 for s in samples_train if s.is_requirement == 0)
    pos_weight_req = torch.tensor([n_non_req / max(n_req, 1)]).to(device)
    
    loss_fn_req = nn.BCEWithLogitsLoss(pos_weight=pos_weight_req)
    loss_fn_func = nn.BCEWithLogitsLoss()
    loss_fn_amb = nn.BCEWithLogitsLoss()
    
    # Save config
    config = {
        "model": model_name,
        "model_type": model_type,
        "max_length": MAX_LENGTH,
        "n_context": args.n_context,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": lora_targets,
        },
        "n_train": len(samples_train),
        "n_dev": len(samples_dev),
        "class_balance": {
            "train_req": n_req,
            "train_non_req": n_non_req,
            "pos_weight": float(pos_weight_req.item()),
        },
        "tasks": {
            "task1": "requirement vs non-requirement (primary)",
            "task2": "functional vs non-functional (secondary)",
            "task3": "ambiguity detection (tertiary)",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    # Prepare data
    def samples_to_arrays(samples: List[Sample]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        texts = [s.text for s in samples]
        y_req = np.array([s.is_requirement if s.is_requirement is not None else np.nan for s in samples], dtype=np.float32)
        y_func = np.array([s.is_functional if s.is_functional is not None else np.nan for s in samples], dtype=np.float32)
        y_amb = np.array([s.is_ambiguous if s.is_ambiguous is not None else np.nan for s in samples], dtype=np.float32)
        return texts, y_req, y_func, y_amb
    
    tr_texts, ytr_req, ytr_func, ytr_amb = samples_to_arrays(samples_train)
    dv_texts, ydv_req, ydv_func, ydv_amb = samples_to_arrays(samples_dev)
    
    print(f"[data] train={len(tr_texts)} dev={len(dv_texts)}")
    print(f"[task1] req_pos={np.nansum(ytr_req):.0f}/{np.sum(~np.isnan(ytr_req)):.0f}")
    print(f"[task2] func_pos={np.nansum(ytr_func):.0f}/{np.sum(~np.isnan(ytr_func)):.0f}")
    print(f"[task3] amb_pos={np.nansum(ytr_amb):.0f}/{np.sum(~np.isnan(ytr_amb)):.0f}")
    
    def embed_batch(batch_texts: List[str]) -> torch.Tensor:
        enc = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder(**enc).last_hidden_state
        pooled = mean_pooling(out, enc["attention_mask"])
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        return pooled
    
    # Training loop
    best_f1 = -1.0
    bad = 0
    history = []
    
    print(f"\n{'='*70}")
    print(f"[train] Multi-task LoRA fine-tuning")
    print(f"{'='*70}\n")
    
    for ep in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        
        # Shuffle
        order = np.random.permutation(len(tr_texts))
        tr_texts_shuf = [tr_texts[i] for i in order]
        ytr_req_shuf = ytr_req[order]
        ytr_func_shuf = ytr_func[order]
        ytr_amb_shuf = ytr_amb[order]
        
        # Mini-batches
        losses_req, losses_func, losses_amb = [], [], []
        
        for i in range(0, len(tr_texts_shuf), args.batch_size):
            bt = tr_texts_shuf[i:i + args.batch_size]
            by_req = torch.tensor(ytr_req_shuf[i:i + args.batch_size]).to(device)
            by_func = torch.tensor(ytr_func_shuf[i:i + args.batch_size]).to(device)
            by_amb = torch.tensor(ytr_amb_shuf[i:i + args.batch_size]).to(device)
            
            opt.zero_grad()
            emb = embed_batch(bt)
            logits_req, logits_func, logits_amb = head(emb)
            
            # Compute losses only for valid labels
            loss_total = 0.0
            n_tasks = 0
            
            # Task 1: Requirement
            valid_req = ~torch.isnan(by_req)
            if valid_req.sum() > 0:
                loss_req = loss_fn_req(logits_req[valid_req], by_req[valid_req])
                loss_total += args.task1_weight * loss_req
                losses_req.append(loss_req.item())
                n_tasks += 1
            
            # Task 2: Functional
            valid_func = ~torch.isnan(by_func)
            if valid_func.sum() > 0:
                loss_func = loss_fn_func(logits_func[valid_func], by_func[valid_func])
                loss_total += args.task2_weight * loss_func
                losses_func.append(loss_func.item())
                n_tasks += 1
            
            # Task 3: Ambiguity
            valid_amb = ~torch.isnan(by_amb)
            if valid_amb.sum() > 0:
                loss_amb = loss_fn_amb(logits_amb[valid_amb], by_amb[valid_amb])
                loss_total += args.task3_weight * loss_amb
                losses_amb.append(loss_amb.item())
                n_tasks += 1
            
            if n_tasks > 0:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
        
        train_loss_req = float(np.mean(losses_req)) if losses_req else float("nan")
        train_loss_func = float(np.mean(losses_func)) if losses_func else float("nan")
        train_loss_amb = float(np.mean(losses_amb)) if losses_amb else float("nan")
        
        # Evaluation
        encoder.eval()
        head.eval()
        
        with torch.no_grad():
            dv_probs_req_all, dv_probs_func_all, dv_probs_amb_all = [], [], []
            
            for i in range(0, len(dv_texts), args.batch_size):
                bt = dv_texts[i:i + args.batch_size]
                emb = embed_batch(bt)
                logits_req, logits_func, logits_amb = head(emb)
                
                probs_req = torch.sigmoid(logits_req).detach().cpu().numpy()
                probs_func = torch.sigmoid(logits_func).detach().cpu().numpy()
                probs_amb = torch.sigmoid(logits_amb).detach().cpu().numpy()
                
                dv_probs_req_all.append(probs_req)
                dv_probs_func_all.append(probs_func)
                dv_probs_amb_all.append(probs_amb)
            
            dv_probs_req = np.concatenate(dv_probs_req_all, axis=0)
            dv_probs_func = np.concatenate(dv_probs_func_all, axis=0)
            dv_probs_amb = np.concatenate(dv_probs_amb_all, axis=0)
            
            metrics_req = compute_metrics_single_task(ydv_req, dv_probs_req, "req")
            metrics_func = compute_metrics_single_task(ydv_func, dv_probs_func, "func")
            metrics_amb = compute_metrics_single_task(ydv_amb, dv_probs_amb, "amb")
            
            all_metrics = {**metrics_req, **metrics_func, **metrics_amb}
        
        # Primary metric: Task 1 F1 (requirement classification)
        primary_f1 = metrics_req.get("req_f1", 0.0)
        
        print(f"[ep {ep:02d}] "
              f"req_f1={metrics_req.get('req_f1', 0):.3f} "
              f"req_acc={metrics_req.get('req_acc', 0):.3f} | "
              f"func_f1={metrics_func.get('func_f1', 0):.3f} | "
              f"amb_f1={metrics_amb.get('amb_f1', 0):.3f}")
        
        history.append({
            "epoch": ep,
            "train_loss_req": train_loss_req,
            "train_loss_func": train_loss_func,
            "train_loss_amb": train_loss_amb,
            "dev_metrics": all_metrics,
        })
        
        # Save last
        torch.save({
            "epoch": ep,
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "config": config,
        }, out_dir / "last.pt")
        
        # Early stopping on primary task F1
        if primary_f1 > best_f1:
            best_f1 = primary_f1
            bad = 0
            torch.save({
                "epoch": ep,
                "encoder_state": encoder.state_dict(),
                "head_state": head.state_dict(),
                "config": config,
                "best_dev_f1": primary_f1,
                "best_dev_metrics": all_metrics,
            }, out_dir / "best.pt")
            print(f"[best] req_f1={primary_f1:.3f} ")
        else:
            bad += 1
        
        if bad >= args.patience:
            print(f"[stop] early stopping (patience={args.patience})")
            break
    
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[done] saved -> {out_dir}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Multi-task LoRA fine-tuning for requirement classification")
    
    # Data
    ap.add_argument("--n-context", type=int, default=1, help="Context window size for contextual data")
    ap.add_argument("--use-frnfr", action="store_true", help="Include F/NFR training data")
    ap.add_argument("--use-synthetic", action="store_true", help="Include synthetic non-requirements")
    
    # Model
    ap.add_argument("--model", type=str, default="deberta", 
                    choices=list(MODEL_OPTIONS.keys()),
                    help="Base model to use")
    
    # Training
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dev-ratio", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    
    # Multi-task weights
    ap.add_argument("--task1-weight", type=float, default=1.0, help="Weight for requirement task")
    ap.add_argument("--task2-weight", type=float, default=0.5, help="Weight for F/NFR task")
    ap.add_argument("--task3-weight", type=float, default=0.3, help="Weight for ambiguity task")
    
    args = ap.parse_args()
    set_seed(args.seed)
    
    # Load all data
    print(f"\n{'='*70}")
    print("[data] Loading datasets...")
    print(f"{'='*70}\n")
    
    samples_contextual = load_contextual_data(DATA_CONTEXTUAL, n_context=args.n_context)
    print(f"[OK] Contextual: {len(samples_contextual)} samples")
    
    samples_frnfr_train, samples_frnfr_dev = [], []
    if args.use_frnfr:
        samples_frnfr_train = load_frnfr_data(DATA_FRNFR_TRAIN)
        samples_frnfr_dev = load_frnfr_data(DATA_FRNFR_DEV)
        print(f"[OK] F/NFR train: {len(samples_frnfr_train)} samples")
        print(f"[OK] F/NFR dev: {len(samples_frnfr_dev)} samples")
    
    samples_synthetic = []
    if args.use_synthetic:
        samples_synthetic = load_synthetic_nr(DATA_SYNTH_NR)
        print(f"[OK] Synthetic non-req: {len(samples_synthetic)} samples")
    
    # Combine training data
    # Split contextual into train/dev
    idx = np.arange(len(samples_contextual))
    y_contextual = np.array([s.is_requirement for s in samples_contextual])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.dev_ratio, random_state=args.seed)
    tr_idx, dv_idx = next(sss.split(idx, y_contextual))
    
    samples_train = [samples_contextual[i] for i in tr_idx]
    samples_dev = [samples_contextual[i] for i in dv_idx]
    
    # Add F/NFR data
    if args.use_frnfr:
        samples_train.extend(samples_frnfr_train)
        samples_dev.extend(samples_frnfr_dev)
    
    # Add synthetic non-requirements to train
    if args.use_synthetic:
        # Use 80% for train, 20% for dev
        n_synth = len(samples_synthetic)
        n_train_synth = int(0.8 * n_synth)
        samples_train.extend(samples_synthetic[:n_train_synth])
        samples_dev.extend(samples_synthetic[n_train_synth:])
    
    print(f"\n[combined] train={len(samples_train)} dev={len(samples_dev)}")
    
    # Run training
    run_id = now_run_id()
    run_root = OUT_ROOT / run_id / args.model
    
    meta = {
        "run_id": run_id,
        "timestamp": run_id,
        "model_type": args.model,
        "n_train": len(samples_train),
        "n_dev": len(samples_dev),
        "use_frnfr": args.use_frnfr,
        "use_synthetic": args.use_synthetic,
        "n_context": args.n_context,
    }
    ensure_dir(OUT_ROOT / run_id)
    (OUT_ROOT / run_id / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    train_multitask(samples_train, samples_dev, args, run_root, args.model)
    
    print(f"\n{'='*70}")
    print(f"[done] Model saved -> {run_root}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()