#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Fine-Tuning DeBERTa for Requirement Classification
--------------------------------------------------------

RESEARCH-GRADE MODEL FOR 85-90% ACCURACY

This script does FULL fine-tuning (not LoRA) for maximum performance.
All hyperparameters are configurable via variables at the top.

Key Differences from Multi-Task:
- Single task only (requirement classification)
- Full fine-tuning (all parameters updated)
- Larger models supported (deberta-v3-large)
- Context-aware features (like Phase-1.5)
- Optimized for research-grade accuracy
"""

from __future__ import annotations

import os, sys, json, time, random
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# =====================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# =====================================================================

class Config:
    """All hyperparameters in one place - easy to modify"""
    
    # ============ MODEL CONFIGURATION ============
    MODEL_NAME = "microsoft/deberta-v3-base"  # Options:
    # "microsoft/deberta-v3-base"  (184M params, good for starting)
    # "microsoft/deberta-v3-large" (304M params, best accuracy, slower)
    # "roberta-large"              (355M params, alternative)
    # "microsoft/mpnet-base"       (110M params, faster)
    
    # ============ TRAINING CONFIGURATION ============
    FULL_FINETUNE = True  # True = full fine-tuning, False = freeze encoder
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    EPOCHS = 15  # More epochs for full fine-tuning
    PATIENCE = 5  # Early stopping patience
    BATCH_SIZE = 8  # Smaller batch for large models (adjust based on GPU memory)
    ACCUMULATION_STEPS = 4  # Effective batch = BATCH_SIZE * ACCUMULATION_STEPS = 32
    
    LEARNING_RATE = 5e-6  # Lower LR for full fine-tuning
    WARMUP_RATIO = 0.1  # 10% of training for warmup
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0
    
    # LR scheduler type
    SCHEDULER = "cosine"  # Options: "linear", "cosine"
    
    # ============ DATA CONFIGURATION ============
    DATA_CONTEXTUAL = "classifier/outputs/splits/holdout_60.jsonl"
    DATA_FRNFR_TRAIN = "classifier/outputs/splits_frnfr/train.jsonl"
    DATA_FRNFR_DEV = "classifier/outputs/splits_frnfr/dev.jsonl"
    DATA_SYNTH_NR = "classifier/synthetic/non_requirements/synthetic_non_requirements.jsonl"
    
    USE_FRNFR = True  # Include F/NFR data as additional requirements
    USE_SYNTHETIC = True  # Include synthetic non-requirements
    
    DEV_RATIO = 0.15  # Train/dev split ratio
    
    # ============ CONTEXT CONFIGURATION ============
    N_CONTEXT = 2  # Context window size (1, 2, or 3)
    MAX_LENGTH = 256  # Longer context for better performance
    
    # Context features (like Phase-1.5)
    USE_CONTEXT_FEATURES = True  # Add abs_diff and cosine similarity
    
    # ============ MODEL ARCHITECTURE ============
    # Classifier head configuration
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 256
    DROPOUT_1 = 0.3
    DROPOUT_2 = 0.2
    
    # ============ OUTPUT CONFIGURATION ============
    OUT_ROOT = "classifier/models/deberta_full_finetune"
    SAVE_BEST_ONLY = False  # Save both best and last checkpoint
    
    # ============ CLASS IMBALANCE ============
    USE_CLASS_WEIGHTS = True  # Weight loss by class frequency
    
    # ============ LOGGING ============
    LOG_INTERVAL = 5  # Log every N batches
    EVAL_STEPS = None  # Eval every N steps (None = eval every epoch)

# =====================================================================
# PROJECT SETUP
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[init] project root: {PROJECT_ROOT}")
print(f"[device] using: {Config.DEVICE}")

# =====================================================================
# UTILITIES
# =====================================================================
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

# =====================================================================
# DATA STRUCTURES
# =====================================================================
@dataclass
class Sample:
    """Training sample"""
    text: str
    context_text: str  # Text with context
    label: int  # 0 or 1
    source: str = "unknown"

# =====================================================================
# DATA LOADERS
# =====================================================================
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

def load_contextual_data(path: Path, n_context: int) -> List[Sample]:
    """Load contextual requirement data"""
    rows = load_jsonl(path)
    samples = []
    
    for r in rows:
        labels = set(r.get("llm_labels", []) or [])
        if not labels.intersection({"requirement", "non_requirement", "with_context"}):
            continue
        
        sent = (r.get("sentence") or "").strip()
        if not sent:
            continue
        
        ctx_text = build_context(r, n_context)
        
        is_req = 1 if ("requirement" in labels or "with_context" in labels) else 0
        
        samples.append(Sample(
            text=sent,
            context_text=ctx_text,
            label=is_req,
            source="contextual"
        ))
    
    return samples

def load_frnfr_data(path: Path) -> List[Sample]:
    """Load F/NFR data - treat all as requirements"""
    rows = load_jsonl(path)
    samples = []
    
    for r in rows:
        text = r.get("text", "").strip()
        label = r.get("label", "").strip()
        
        if not text or not label:
            continue
        
        # All F/NFR samples are requirements
        samples.append(Sample(
            text=text,
            context_text=text,  # No context available
            label=1,  # All are requirements
            source="frnfr"
        ))
    
    return samples

def load_synthetic_nr(path: Path) -> List[Sample]:
    """Load synthetic non-requirements"""
    rows = load_jsonl(path)
    samples = []

    for r in rows:
        raw_text = r.get("text", "")

        # If text is a dictionary, extract actual string
        if isinstance(raw_text, dict):
            raw_text = (
                raw_text.get("content")
                or raw_text.get("sentence")
                or raw_text.get("text")
                or ""
            )

        # Ensure string
        if not isinstance(raw_text, str):
            continue

        text = raw_text.strip()
        label = r.get("label", "").strip()

        if not text or label != "non_requirement":
            continue

        samples.append(Sample(
            text=text,
            context_text=text,
            label=0,
            source="synthetic"
        ))

    return samples


# =====================================================================
# DATASET
# =====================================================================
class RequirementDataset(Dataset):
    """PyTorch Dataset for requirements"""
    
    def __init__(self, samples: List[Sample], tokenizer, max_length: int, use_context: bool = True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Use context text if available and enabled
        text = sample.context_text if self.use_context else sample.text
        
        encoding = self.tokenizer(
            text,
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
    """Full fine-tuning classifier with context features"""
    
    def __init__(self, model_name: str, config: Config, freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config_obj = config
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[model] Encoder frozen")
        else:
            print("[model] Full fine-tuning enabled")
        
        # Get embedding dimension
        if hasattr(self.encoder.config, 'hidden_size'):
            emb_dim = self.encoder.config.hidden_size
        else:
            emb_dim = 768
        
        print(f"[model] Embedding dimension: {emb_dim}")
        
        # Classifier head (improved architecture)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, config.HIDDEN_DIM_1),
            nn.LayerNorm(config.HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_1),
            nn.Linear(config.HIDDEN_DIM_1, config.HIDDEN_DIM_2),
            nn.LayerNorm(config.HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_2),
            nn.Linear(config.HIDDEN_DIM_2, 2),  # Binary classification
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

# =====================================================================
# METRICS
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics"""
    
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        roc = float(roc_auc_score(y_true, probs[:, 1])) if len(set(y_true.tolist())) > 1 else float("nan")
    except:
        roc = float("nan")
    
    try:
        pr_auc = float(average_precision_score(y_true, probs[:, 1]))
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
    }

# =====================================================================
# TRAINING
# =====================================================================
def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, config, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward (with gradient accumulation)
        loss = loss / config.ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % config.ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * config.ACCUMULATION_STEPS
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        
        # Logging
        if (step + 1) % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (step + 1)
            print(f"  [step {step+1}/{len(train_loader)}] loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
    
    # Epoch metrics
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return avg_loss, metrics

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
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
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return avg_loss, metrics

# =====================================================================
# MAIN
# =====================================================================
def main():
    config = Config()
    print("MODEL USED:", config.MODEL_NAME)
    set_seed(config.SEED)

    
    # Create output directory
    run_id = now_run_id()
    model_short_name = config.MODEL_NAME.split('/')[-1]
    out_dir = Path(config.OUT_ROOT) / f"{run_id}_{model_short_name}"
    ensure_dir(out_dir)
    
    print(f"\n{'='*70}")
    print(f"[train] Full Fine-Tuning: {config.MODEL_NAME}")
    print(f"{'='*70}\n")
    
    # Save config
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    config_dict['run_id'] = run_id
    config_dict['output_dir'] = str(out_dir)
    
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    
    # Load data
    print(f"{'='*70}")
    print("[data] Loading datasets...")
    print(f"{'='*70}\n")
    
    samples_contextual = load_contextual_data(Path(config.DATA_CONTEXTUAL), config.N_CONTEXT)
    print(f"[OK] Contextual: {len(samples_contextual)} samples")
    
    samples_frnfr = []
    if config.USE_FRNFR:
        samples_frnfr_train = load_frnfr_data(Path(config.DATA_FRNFR_TRAIN))
        samples_frnfr_dev = load_frnfr_data(Path(config.DATA_FRNFR_DEV))
        samples_frnfr = samples_frnfr_train + samples_frnfr_dev
        print(f"[OK] F/NFR: {len(samples_frnfr)} samples")
    
    samples_synthetic = []
    if config.USE_SYNTHETIC:
        samples_synthetic = load_synthetic_nr(Path(config.DATA_SYNTH_NR))
        print(f"[OK] Synthetic: {len(samples_synthetic)} samples")
    
    # Combine and split
    all_samples = samples_contextual + samples_frnfr + samples_synthetic
    
    # ================================================================
    # CLASS BALANCE ENFORCEMENT (60-40 to 65-35 ratio)
    # ================================================================
    print(f"\n{'='*70}")
    print("[data] Balancing dataset...")
    print(f"{'='*70}\n")
    
    # Separate by class
    pos_samples = [s for s in all_samples if s.label == 1]
    neg_samples = [s for s in all_samples if s.label == 0]
    
    print(f"[before] Positive: {len(pos_samples)}, Negative: {len(neg_samples)}")
    print(f"[before] Ratio: {len(pos_samples)/(len(pos_samples)+len(neg_samples))*100:.1f}% pos / {len(neg_samples)/(len(pos_samples)+len(neg_samples))*100:.1f}% neg")
    
    # Target: 60-65% positive, 35-40% negative
    TARGET_POS_RATIO = 0.625  # 62.5% positive (middle of 60-65%)
    
    n_pos = len(pos_samples)
    n_neg_target = int(n_pos * (1 - TARGET_POS_RATIO) / TARGET_POS_RATIO)
    
    # Save backup of discarded samples
    backup_dir = Path(config.OUT_ROOT) / "discarded_samples_backup"
    ensure_dir(backup_dir)
    
    if len(neg_samples) > n_neg_target:
        # Randomly sample negative samples
        np.random.shuffle(neg_samples)
        neg_samples_keep = neg_samples[:n_neg_target]
        neg_samples_discard = neg_samples[n_neg_target:]
        
        # Save discarded samples
        backup_file = backup_dir / f"discarded_negatives_{now_run_id()}.jsonl"
        with open(backup_file, "w", encoding="utf-8") as f:
            for s in neg_samples_discard:
                f.write(json.dumps({
                    "text": s.text,
                    "context_text": s.context_text,
                    "label": s.label,
                    "source": s.source
                }) + "\n")
        
        print(f"[balanced] Kept {len(neg_samples_keep)} negatives, discarded {len(neg_samples_discard)}")
        print(f"[backup] Saved discarded samples to: {backup_file}")
        
        neg_samples = neg_samples_keep
    
    # Combine balanced samples
    all_samples = pos_samples + neg_samples
    np.random.shuffle(all_samples)  # Shuffle combined samples
    
    print(f"[after] Positive: {len(pos_samples)}, Negative: {len(neg_samples)}")
    print(f"[after] Ratio: {len(pos_samples)/(len(pos_samples)+len(neg_samples))*100:.1f}% pos / {len(neg_samples)/(len(pos_samples)+len(neg_samples))*100:.1f}% neg")
    print(f"[after] Total samples: {len(all_samples)}")
    
    # ================================================================
    
    # Split contextual into train/dev, add others to train
    y = np.array([s.label for s in all_samples])
    idx = np.arange(len(all_samples))
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.DEV_RATIO, random_state=config.SEED)
    train_idx, dev_idx = next(sss.split(idx, y))
    
    train_samples = [all_samples[i] for i in train_idx]
    dev_samples = [all_samples[i] for i in dev_idx]
    
    n_pos_train = sum(1 for s in train_samples if s.label == 1)
    n_neg_train = len(train_samples) - n_pos_train
    
    print(f"\n[data] Train: {len(train_samples)} ({n_pos_train} pos, {n_neg_train} neg)")
    print(f"[data] Dev: {len(dev_samples)} ({sum(1 for s in dev_samples if s.label == 1)} pos)")
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_dataset = RequirementDataset(train_samples, tokenizer, config.MAX_LENGTH, use_context=True)
    dev_dataset = RequirementDataset(dev_samples, tokenizer, config.MAX_LENGTH, use_context=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n{'='*70}")
    print("[model] Initializing...")
    print(f"{'='*70}\n")
    
    model = RequirementClassifier(
        config.MODEL_NAME,
        config,
        freeze_encoder=not config.FULL_FINETUNE
    )
    model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total parameters: {total_params:,}")
    print(f"[model] Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * config.EPOCHS // config.ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    
    if config.SCHEDULER == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    print(f"[scheduler] {config.SCHEDULER} with {num_warmup_steps} warmup steps")
    
    # Loss function with class weighting
    if config.USE_CLASS_WEIGHTS:
        pos_weight = n_neg_train / n_pos_train
        class_weights = torch.tensor([1.0, pos_weight]).to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[loss] Class weights: [1.0, {pos_weight:.3f}]")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\n{'='*70}")
    print("[train] Starting training...")
    print(f"{'='*70}\n")
    
    best_f1 = -1.0
    best_acc = -1.0
    patience_counter = 0
    history = []
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            config.DEVICE, config, epoch
        )
        
        # Evaluate
        dev_loss, dev_metrics = evaluate(model, dev_loader, criterion, config.DEVICE)
        
        # Log
        print(f"\n{'='*70}")
        print(f"[EPOCH {epoch} RESULTS]")
        print(f"{'='*70}")
        print(f"\nTRAINING METRICS:")
        print(f"  Loss:      {train_loss:.4f}")
        print(f"  Accuracy:  {train_metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {train_metrics['precision']*100:.2f}%")
        print(f"  Recall:    {train_metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {train_metrics['f1']*100:.2f}%")
        print(f"  ROC-AUC:   {train_metrics['roc_auc']:.4f}")
        
        print(f"\nDEV METRICS:")
        print(f"  Loss:      {dev_loss:.4f}")
        print(f"  Accuracy:  {dev_metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {dev_metrics['precision']*100:.2f}%")
        print(f"  Recall:    {dev_metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {dev_metrics['f1']*100:.2f}%")
        print(f"  ROC-AUC:   {dev_metrics['roc_auc']:.4f}")
        
        print(f"\nCONFUSION MATRIX (Dev):")
        print(f"  True Positives:  {dev_metrics['tp']:4d}")
        print(f"  False Positives: {dev_metrics['fp']:4d}")
        print(f"  True Negatives:  {dev_metrics['tn']:4d}")
        print(f"  False Negatives: {dev_metrics['fn']:4d}")
        
        print(f"\nPROGRESS:")
        print(f"  Best F1 so far: {max(best_f1, dev_metrics['f1'])*100:.2f}%")
        print(f"  Patience: {patience_counter}/{config.PATIENCE}")
        print(f"{'='*70}")
        
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
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config_dict,
            "dev_metrics": dev_metrics,
        }
        
        # Save last
        if not config.SAVE_BEST_ONLY:
            torch.save(checkpoint, out_dir / "last.pt")
        
        # Save best (by F1, then accuracy)
        is_best = False
        if dev_metrics['f1'] > best_f1:
            is_best = True
            best_f1 = dev_metrics['f1']
            best_acc = dev_metrics['accuracy']
        elif dev_metrics['f1'] == best_f1 and dev_metrics['accuracy'] > best_acc:
            is_best = True
            best_acc = dev_metrics['accuracy']
        
        if is_best:
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"\n  [BEST] F1={best_f1:.4f}, Acc={best_acc:.4f} ***")
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
    
    print(f"\n{'='*70}")
    print(f"[done] Training complete!")
    print(f"[done] Best F1: {best_f1:.4f}, Best Acc: {best_acc:.4f}")
    print(f"[done] Model saved to: {out_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()