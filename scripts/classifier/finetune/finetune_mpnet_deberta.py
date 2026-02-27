#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Context-Aware Training: DeBERTa + MPNet
------------------------------------------------

Trains both DeBERTa and MPNet models with multiple context window sizes (n=1,2,3)
on a clean 4K requirement dataset.

Strategy:
- Models: DeBERTa-base, MPNet-base
- Context sizes: n=1, 2, 3
- Dataset: ~4K requirement sentences only (with context)
- Clean comparison across models and context sizes

Usage:
    python train_context_comparison.py
    python train_context_comparison.py --context-sizes "1,2,3"
    python train_context_comparison.py --max-samples 4000
"""

from __future__ import annotations

import os, sys, json, time, random
import io
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Fix Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =====================================================================
# CONFIGURATION
# =====================================================================
class Config:
    """Unified configuration for context comparison"""
    
    # Models to train
    MODELS = {
        "deberta-base": "microsoft/deberta-v3-base",
        "mpnet-base": "sentence-transformers/all-mpnet-base-v2",
    }
    
    # Context sizes to test
    CONTEXT_SIZES = [1, 2, 3]
    
    # Data configuration
    DATA_CONTEXTUAL = "classifier/outputs/splits/train_4k.jsonl"
    MAX_SAMPLES = 4000  # Target ~4K samples
    DEV_RATIO = 0.15
    
    # Training configuration
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    EPOCHS = 15
    PATIENCE = 5
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-6
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0
    MAX_LENGTH = 256
    
    # Architecture
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 256
    DROPOUT_1 = 0.3
    DROPOUT_2 = 0.2
    
    # Output
    OUT_ROOT = "classifier/models/context_comparison"
    
    # Class weighting
    USE_CLASS_WEIGHTS = True

# =====================================================================
# PROJECT SETUP
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[init] project root: {PROJECT_ROOT}")

# =====================================================================
# UTILITIES
# =====================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def mean_pooling(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

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

# =====================================================================
# DATA STRUCTURES
# =====================================================================
@dataclass
class Sample:
    text: str
    context_text: str
    label: int
    sent_id: str = ""

# =====================================================================
# DATA LOADER
# =====================================================================
def load_contextual_data(path: Path, n_context: int, max_samples: int = None) -> List[Sample]:
    """Load contextual data with proper binary labels.
    
    Labeling:
        requirement OR with_context → 1
        non_requirement → 0
        others → skipped
    """
    rows = load_jsonl(path)
    samples = []

    for r in rows:
        labels = set(r.get("llm_labels", []) or [])

        # Determine label
        if "requirement" in labels or "with_context" in labels:
            label = 1
        elif "non_requirement" in labels:
            label = 0
        else:
            continue  # skip uncertain / ambiguous / other cases

        sent = (r.get("sentence") or "").strip()
        if not sent:
            continue

        ctx_text = build_context_text(r, n_context)

        samples.append(Sample(
            text=sent,
            context_text=ctx_text,
            label=label,
            sent_id=r.get("sent_id", "")
        ))

        if max_samples and len(samples) >= max_samples:
            break

    return samples


# =====================================================================
# DATASET
# =====================================================================
class RequirementDataset(Dataset):
    def __init__(self, samples: List[Sample], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample.context_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample.label, dtype=torch.long)
        }

# =====================================================================
# MODEL
# =====================================================================
class RequirementClassifier(nn.Module):
    """Unified classifier for both DeBERTa and MPNet
    with architecture-aware pooling + normalization.
    """
    
    def __init__(self, model_name: str, config: Config):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config_obj = config
        self.model_name = model_name.lower()

        # Detect model type once
        self.is_deberta = "deberta" in self.model_name
        self.is_mpnet = "mpnet" in self.model_name

        # Get embedding dimension
        if hasattr(self.encoder.config, 'hidden_size'):
            emb_dim = self.encoder.config.hidden_size
        else:
            emb_dim = 768

        # Classifier head (unchanged)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, config.HIDDEN_DIM_1),
            nn.LayerNorm(config.HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_1),
            nn.Linear(config.HIDDEN_DIM_1, config.HIDDEN_DIM_2),
            nn.LayerNorm(config.HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_2),
            nn.Linear(config.HIDDEN_DIM_2, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # ----------------------------
        # Model-aware pooling
        # ----------------------------
        if self.is_deberta:
            # Use CLS token for DeBERTa
            pooled = outputs.last_hidden_state[:, 0]
        else:
            # Mean pooling for MPNet (and others)
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask)

        # ----------------------------
        # Normalize only for MPNet
        # ----------------------------
        if self.is_mpnet:
            pooled = nn.functional.normalize(pooled, p=2, dim=1)

        logits = self.classifier(pooled)
        return logits


# =====================================================================
# METRICS
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        roc = float(roc_auc_score(y_true, probs[:, 1])) if len(set(y_true.tolist())) > 1 else float("nan")
    except:
        roc = float("nan")
    
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
    }

# =====================================================================
# TRAINING
# =====================================================================
def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, config):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss = loss / config.ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % config.ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.ACCUMULATION_STEPS
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
    
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    
    return avg_loss, metrics

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    
    return avg_loss, metrics

# =====================================================================
# MAIN TRAINING FUNCTION
# =====================================================================
def train_model(model_name: str, model_key: str, n_context: int, config: Config, samples: List[Sample]):
    """Train a single model with specific context size"""
    
    print(f"\n{'='*70}")
    print(f"[train] {model_key.upper()} with n={n_context}")
    print(f"{'='*70}\n")
    
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.OUT_ROOT) / f"{run_id}_{model_key}_n{n_context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {
        "model_name": model_name,
        "model_key": model_key,
        "n_context": n_context,
        "n_samples": len(samples),
        "run_id": run_id,
        **{k: v for k, v in vars(config).items() if not k.startswith('_') and k.isupper()}
    }
    
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    
    # Split data
    y = np.array([s.label for s in samples])
    idx = np.arange(len(samples))
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.DEV_RATIO, random_state=config.SEED)
    train_idx, dev_idx = next(sss.split(idx, y))
    
    train_samples = [samples[i] for i in train_idx]
    dev_samples = [samples[i] for i in dev_idx]
    
    print(f"[data] Train: {len(train_samples)}, Dev: {len(dev_samples)}")
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = RequirementDataset(train_samples, tokenizer, config.MAX_LENGTH)
    dev_dataset = RequirementDataset(dev_samples, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Create model
    model = RequirementClassifier(model_name, config).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    num_training_steps = len(train_loader) * config.EPOCHS // config.ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Loss function
    if config.USE_CLASS_WEIGHTS:
        n_pos = sum(1 for s in train_samples if s.label == 1)
        n_neg = sum(1 for s in train_samples if s.label == 0)

        # Avoid degenerate cases
        if n_pos == 0 or n_neg == 0:
            print(f"[warn] Degenerate class distribution in train: n_neg={n_neg}, n_pos={n_pos}. Disabling class weights.")
            criterion = nn.CrossEntropyLoss()
        else:
            # Inverse-frequency weights (balanced)
            # weight[c] ∝ N / (2 * count[c])
            total = n_pos + n_neg
            w0 = total / (2.0 * n_neg)
            w1 = total / (2.0 * n_pos)
            class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=config.DEVICE)
            print(f"[info] Using class weights: w0={w0:.4f}, w1={w1:.4f} | n_neg={n_neg}, n_pos={n_pos}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = -1.0
    patience_counter = 0
    history = []
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{config.EPOCHS}]")
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, config.DEVICE, config
        )
        dev_loss, dev_metrics = evaluate(model, dev_loader, criterion, config.DEVICE)
        
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_metrics['f1']*100:.2f}%, Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"  Dev   - Loss: {dev_loss:.4f}, F1: {dev_metrics['f1']*100:.2f}%, Acc: {dev_metrics['accuracy']*100:.2f}%")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "dev_loss": dev_loss,
            "dev_metrics": dev_metrics,
        })
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_dict,
            "dev_metrics": dev_metrics,
        }
        
        # Save best
        if dev_metrics['f1'] > best_f1:
            best_f1 = dev_metrics['f1']
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"  [BEST] F1={best_f1*100:.2f}% ***")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n[early stop] No improvement for {config.PATIENCE} epochs")
            break
    
    # Save history
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[done] Best F1: {best_f1*100:.2f}%")
    print(f"[done] Model saved to: {out_dir}")
    
    return {
        "model_key": model_key,
        "n_context": n_context,
        "best_f1": best_f1,
        "best_acc": dev_metrics['accuracy'],
        "output_dir": str(out_dir),
    }

# =====================================================================
# MAIN
# =====================================================================
def main():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-sizes", type=str, default="1,2,3")
    ap.add_argument("--models", type=str, default="deberta-base,mpnet-base")
    ap.add_argument("--max-samples", type=int, default=4000)
    args = ap.parse_args()
    
    config = Config()
    set_seed(config.SEED)
    
    # Parse arguments
    context_sizes = [int(n.strip()) for n in args.context_sizes.split(",")]
    models_to_train = [m.strip() for m in args.models.split(",")]
    
    print(f"\n{'='*70}")
    print("[train] Context-Aware Model Comparison")
    print(f"{'='*70}")
    print(f"Models: {models_to_train}")
    print(f"Context sizes: {context_sizes}")
    print(f"Max samples: {args.max_samples}")
    print(f"Device: {config.DEVICE}")
    print(f"{'='*70}\n")
    
    # Load data for each context size
    print(f"[data] Loading contextual data...")
    data_by_context = {}
    
    for n in context_sizes:
        samples = load_contextual_data(
            Path(config.DATA_CONTEXTUAL),
            n_context=n,
            max_samples=args.max_samples
        )
        data_by_context[n] = samples
        labels = np.array([s.label for s in samples], dtype=int)
        dist = np.bincount(labels, minlength=2)
        print(f"  n={n}: {len(samples)} samples | label dist = {dist.tolist()} (0=non_req, 1=req)")

    
    # Train all combinations
    results = []
    total_runs = len(models_to_train) * len(context_sizes)
    current_run = 0
    
    for model_key in models_to_train:
        if model_key not in config.MODELS:
            print(f"[warn] Unknown model: {model_key}, skipping")
            continue
        
        model_name = config.MODELS[model_key]
        
        for n in context_sizes:
            current_run += 1
            print(f"\n{'='*70}")
            print(f"[progress] Run {current_run}/{total_runs}")
            print(f"{'='*70}")
            
            result = train_model(
                model_name=model_name,
                model_key=model_key,
                n_context=n,
                config=config,
                samples=data_by_context[n]
            )
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("[summary] Training Complete")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<15} {'n_ctx':>5} {'Best F1':>10} {'Best Acc':>10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['model_key']:<15} {r['n_context']:>5} {r['best_f1']*100:>9.2f}% {r['best_acc']*100:>9.2f}%")
    
    # Save summary
    summary_file = Path(config.OUT_ROOT) / f"training_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "context_sizes": context_sizes,
                "models": models_to_train,
                "max_samples": args.max_samples,
            },
            "results": results,
        }, f, indent=2)
    
    print(f"\n[done] Summary saved to: {summary_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()