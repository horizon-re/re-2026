# Scripts Reference

This guide documents the training methodology used in the study. Full replication may require GPUs, local models, external APIs, and environment configurations beyond the scope of anonymous review. The documentation is provided for transparency rather than out-of-the-box execution.

Complete documentation for all scripts in the repository, organized by category.

## Table of Contents

- [Data Processing Scripts](#data-processing-scripts)
- [Annotation Scripts](#annotation-scripts)
- [Training Scripts](#training-scripts)
- [Evaluation Scripts](#evaluation-scripts)
- [Utility Scripts](#utility-scripts)
- [Service Integration](#service-integration)

---

## Data Processing Scripts

### `scripts/requirement-unitizer/scripts/step_0_ingest.py`

**Purpose**: Document ingestion and canonicalization

**Input**: Raw requirement documents from `02_raw_requirements/`

**Output**: `classifier/step_0/<domain>/<prompt>/<req_id>/`

**Usage**:
```bash
# Process all documents
python scripts/requirement-unitizer/scripts/step_0_ingest.py

# Filter by domain
python scripts/requirement-unitizer/scripts/step_0_ingest.py --domain fintech

# Filter by prompt
python scripts/requirement-unitizer/scripts/step_0_ingest.py --prompt P-FT-001

# Limit for testing
python scripts/requirement-unitizer/scripts/step_0_ingest.py --limit 10

# Overwrite existing outputs
python scripts/requirement-unitizer/scripts/step_0_ingest.py --overwrite
```

**Key Functions**:
- `safe_read_text(path)`: Multi-encoding file reader
- `extract_req_id_from_filename(filename)`: ID extraction
- `uuid_v5_for_req(req_key)`: Stable UUID generation
- `parse_rel(rel_path)`: Path parsing

**Output Files**:
- `raw.txt`: Original text
- `hlj.json`: High-level JSON packet
- `meta.json`: Metadata and checksums

---

### `scripts/classifier/dataset_crawler.py`

**Purpose**: Raw text extraction for classifier pipeline

**Input**: `02_raw_requirements/` and `03_refined_json_normalized/`

**Output**: `classifier/step_1/<domain>/<prompt>/`

**Usage**:
```bash
# Generate all outputs
python scripts/classifier/dataset_crawler.py

# Filter by domain
python scripts/classifier/dataset_crawler.py --domain fintech

# Filter by prompt
python scripts/classifier/dataset_crawler.py --prompt P-FT-001

# Dry run (preview only)
python scripts/classifier/dataset_crawler.py --dry-run
```

**Key Functions**:
- `find_source_files(root, domain, prompt)`: File discovery
- `extract_req_id(filename)`: ID extraction
- `generate_output_files(source_path, domain, prompt_id, root)`: File generation

**Output Files**:
- `<req_id>_raw.txt`: Raw requirement text
- `<req_id>_hlj.json`: Structured JSON data

---

### `scripts/classifier/extract_sentences_spacy_stanza.py`

**Purpose**: Sentence segmentation with dual-parser validation

**Input**: `classifier/step_1/<domain>/<prompt>/<req_id>_raw.txt`

**Output**: `classifier/step_2/<domain>/<prompt>/<req_id>/`

**Usage**:
```bash
python scripts/classifier/extract_sentences_spacy_stanza.py
```

**Key Functions**:
- `split_spacy(text, nlp)`: spaCy segmentation
- `split_stanza(text, nlp)`: Stanza segmentation
- `check_alignment(a, b)`: Agreement metrics
- `clean_text(txt)`: Text normalization

**Output Files**:
- `sentences_spacy.jsonl`: spaCy output
- `sentences_stanza.jsonl`: Stanza output
- `merged_sentences.jsonl`: Validated merge
- `_manifest.jsonl`: Statistics

**Configuration**:
```python
LANG = "en"
NLP_SPACY = spacy.load("en_core_web_sm")
NLP_STANZA = stanza.Pipeline("en", processors="tokenize")
```

---

### `scripts/requirement-unitizer/scripts/step_1_sentence_segmentation.py`

**Purpose**: Alternative sentence segmentation (linguistic only)

**Input**: `classifier/step_0/<domain>/<prompt>/<req_id>/raw.txt`

**Output**: `classifier/step_1/<domain>/<prompt>/<req_id>/`

**Usage**:
```bash
python scripts/requirement-unitizer/scripts/step_1_sentence_segmentation.py
```

**Key Functions**:
- `split_spacy(text)`: spaCy segmentation
- `split_stanza(text)`: Stanza segmentation
- `merge_splits(spacy_sents, stanza_sents)`: Conservative merge
- `make_sent_id(req_id, idx)`: ID generation

**Output Files**:
- `sentences_spacy.jsonl`
- `sentences_stanza.jsonl`
- `sentences_merged.jsonl`
- `segmentation_manifest.json`

---

### `scripts/requirement-unitizer/scripts/step_2_req_candidate.py`

**Purpose**: Requirement candidate detection with signal analysis

**Input**: `classifier/step_1/<domain>/<prompt>/<req_id>/sentences_merged.jsonl`

**Output**: `classifier/step_2/<domain>/<prompt>/<req_id>/`

**Usage**:
```bash
python scripts/requirement-unitizer/scripts/step_2_req_candidate.py
```

**Detection Signals**:
```python
has_action_verb(doc) -> bool
has_modal(text) -> bool
has_system_subject(doc) -> bool
has_constraint(text) -> bool
```

**Acceptance Rule**:
```python
is_candidate = action_verb AND (modal OR system_subject OR constraint)
```

**Output Files**:
- `candidates.jsonl`: Accepted sentences
- `non_candidates.jsonl`: Rejected sentences
- `signal_matrix.json`: Per-requirement metrics
- `candidate_manifest.json`: Summary
- `_global_metrics.json`: Global statistics
- `_signal_cooccurrence.json`: Pattern analysis
- `_domain_matrix.json`: Domain-level metrics

---

### `scripts/classifier/embedding_aggregator.py`

**Purpose**: Semantic quality filtering and embedding generation

**Input**: `classifier/step_2/<domain>/<prompt>/<req_id>/merged_sentences.jsonl`

**Output**: `classifier/step_3/<domain>/<prompt>/<req_id>/`

**Usage**:
```bash
python scripts/classifier/embedding_aggregator.py
```

**Quality Scoring**:
```python
keep_score = (0.35 × norm) + (0.20 × noun_ratio) + 
             (0.20 × unique_ratio) + (0.15 × entropy) + 
             (0.10 × context_sim)
```

**Key Functions**:
- `evaluate_sentence_quality(sent, emb, nlp, context_embs)`: Quality scoring
- `build_word_freqs(sent_texts, nlp)`: Frequency counting
- `compute_hybrid_weights(sent_texts, freq, embs)`: SIF weighting
- `passes_basic_filters(text)`: Rule-based filtering

**Output Files**:
- `kept_sentences.jsonl`: High-quality sentences
- `dropped_sentences.jsonl`: Filtered sentences
- `sentence_embeddings.npy`: MPNet embeddings
- `doc_embedding.npy`: SIF-weighted document vector
- `top_sentences.json`: Top-5 by weight
- `_pca_meta.json`: PCA de-biasing info
- `_manifest.jsonl`: Processing statistics

**Configuration**:
```python
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MINI_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIF_A = 1e-3
```

---

### `scripts/requirement-unitizer/scripts/step_3_atomic_splitter.py`

**Purpose**: Atomic requirement unit extraction (dual mode)

**Input**: `classifier/step_2/<domain>/<prompt>/<req_id>/candidates.jsonl`

**Output**: `classifier/step_3/<conservative|aggressive>/<domain>/<prompt>/<req_id>/`

**Usage**:
```bash
python scripts/requirement-unitizer/scripts/step_3_atomic_splitter.py
```

**Splitting Strategies**:
```python
# Conservative: Only split on main verbs + sequential markers
conservative_split(doc) -> List[str]

# Aggressive: Split on all verbs
aggressive_split(doc) -> List[str]
```

**Output Files**:
- `atomic_units.jsonl`: Extracted units
- `split_trace.jsonl`: Splitting provenance
- `atomic_matrix.json`: Per-requirement metrics
- `_atomicity_comparison.json`: Global comparison

---

## Annotation Scripts

### `scripts/requirement-unitizer/llm_labeling/step_2_llm_reviewer.py`

**Purpose**: LLM-assisted annotation with context-aware prompting

**Input**: Sentence candidates from step 2

**Output**: `classifier/llm_reviews/`

**Usage**:
```bash
# Requires API key
export DEEPSEEK_API_KEY="your-key"
python scripts/requirement-unitizer/llm_labeling/step_2_llm_reviewer.py
```

**Annotation Labels**:
- `requirement`: Clear standalone requirement
- `non_requirement`: Definitively not a requirement
- `with_context`: Requires context for identification
- `ambiguous`: Unclear even with context
- `non_ambiguous`: Clear classification

**Output Files**:
- `deepseek_reviews.jsonl`: All annotations
- `_deepseek_review_metrics.json`: Statistics
- `by_req/<domain>/<prompt>/<req_id>.jsonl`: Per-requirement

**Key Features**:
- Context-aware prompting
- 50 few-shot examples
- Confidence scoring
- Reasoning generation

---

### `scripts/requirement-unitizer/annotation_ui/annotate_requirements.py`

**Purpose**: Manual annotation interface

**Usage**:
```bash
python scripts/requirement-unitizer/annotation_ui/annotate_requirements.py
```

**Features**:
- Interactive CLI interface
- Context display
- Label validation
- Progress tracking
- Export to JSONL

---

### `scripts/requirement-unitizer/annotation_ui/llm_annotate_step2.py`

**Purpose**: LLM annotation with manual verification

**Usage**:
```bash
python scripts/requirement-unitizer/annotation_ui/llm_annotate_step2.py
```

**Workflow**:
1. LLM generates initial labels
2. Display for manual review
3. Accept/reject/modify
4. Export verified annotations

---

## Training Scripts

### `scripts/classifier/finetune/finetune_deberta.py`

**Purpose**: Full fine-tuning of DeBERTa-v3-base

**Usage**:
```bash
# Track A (domain-only, 4K)
python scripts/classifier/finetune/finetune_deberta.py \
  --track A \
  --context_size 2 \
  --epochs 15 \
  --lr 5e-6 \
  --batch_size 8

# Track B (mixed, 15K)
python scripts/classifier/finetune/finetune_deberta.py \
  --track B \
  --context_size 2 \
  --epochs 15 \
  --lr 5e-6
```

**Arguments**:
- `--track`: A (domain-only) or B (mixed)
- `--context_size`: 0, 1, 2, or 3
- `--epochs`: Training epochs (default: 15)
- `--lr`: Learning rate (default: 5e-6)
- `--batch_size`: Batch size (default: 8)
- `--grad_accum`: Gradient accumulation steps (default: 4)
- `--max_len`: Max sequence length (default: 256)
- `--seed`: Random seed (default: 42)

**Model Configuration**:
```python
model = "microsoft/deberta-v3-base"  # 184M parameters
pooling = "cls"  # [CLS] token
optimizer = "AdamW"
weight_decay = 1e-2
warmup_ratio = 0.1
scheduler = "cosine"
```

**Output**:
- `classifier/models/deberta_track{A|B}_k{size}/`
  - `best_model.pt`: Best checkpoint
  - `config.json`: Training configuration
  - `metrics.json`: Training metrics
  - `predictions.jsonl`: Test predictions

---

### `scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py`

**Purpose**: Frozen MPNet with structured context features

**Usage**:
```bash
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --context_size 1 \
  --epochs 50 \
  --lr 1e-3
```

**Feature Engineering**:
```python
features = [
    sentence_embedding,      # E_s (768-dim)
    context_embedding,       # E_c (768-dim)
    absolute_difference,     # |E_s - E_c| (768-dim)
    cosine_similarity        # cos(E_s, E_c) (1-dim)
]
# Total: 2305-dim
```

**Classifier Head**:
```python
h1 = Dropout(0.3)(ReLU(LayerNorm(W1 @ x)))  # 512-dim
h2 = Dropout(0.2)(ReLU(LayerNorm(W2 @ h1))) # 256-dim
y = W3 @ h2                                  # 2-dim
```

**Output**:
- `classifier/models/mpnet_phase1_5_k{size}/`
  - `classifier_head.pt`: Trained head
  - `feature_config.json`: Feature configuration
  - `metrics.json`: Evaluation metrics

---

### `scripts/classifier/finetune/check_deberta_large.py`

**Purpose**: Verify DeBERTa-large model availability

**Usage**:
```bash
python scripts/classifier/finetune/check_deberta_large.py
```

**Checks**:
- Model download status
- GPU availability
- Memory requirements
- Tokenizer compatibility

---

## Evaluation Scripts

### `scripts/classifier/eval/eval_deberta_comp.py`

**Purpose**: Comprehensive DeBERTa evaluation

**Usage**:
```bash
python scripts/classifier/eval/eval_deberta_comp.py \
  --model_path classifier/models/deberta_trackA_k2/best_model.pt \
  --test_data classifier/outputs/splits/test_rest.jsonl
```

**Metrics Computed**:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Per-class metrics

**Output**:
- `classifier/outputs/eval/deberta_results.json`
- `classifier/outputs/eval/confusion_matrix.png`
- `classifier/outputs/eval/roc_curve.png`
- `classifier/outputs/eval/pr_curve.png`

---

### `scripts/classifier/eval/eval_mpnet_phase1_5_holdout.py`

**Purpose**: Evaluate frozen MPNet baseline

**Usage**:
```bash
python scripts/classifier/eval/eval_mpnet_phase1_5_holdout.py \
  --model_path classifier/models/mpnet_phase1_5_k1/classifier_head.pt \
  --test_data classifier/outputs/splits/test_rest.jsonl
```

**Output**:
- `classifier/outputs/eval/mpnet_phase1_5_results.json`
- Feature importance analysis
- Error analysis

---

### `scripts/classifier/eval/eval_domain_fullft.py`

**Purpose**: Domain-specific full fine-tuning evaluation

**Usage**:
```bash
python scripts/classifier/eval/eval_domain_fullft.py \
  --track A \
  --context_size 2
```

**Compares**:
- Track A vs Track B
- Different context sizes
- Domain-specific performance

**Output**:
- `classifier/outputs/eval/domain_comparison.json`
- `classifier/outputs/eval/context_ablation.json`

---

### `scripts/classifier/eval/metrics.py`

**Purpose**: Shared metrics computation utilities

**Functions**:
```python
compute_metrics(y_true, y_pred, y_prob) -> Dict
compute_confusion_matrix(y_true, y_pred) -> np.ndarray
compute_roc_auc(y_true, y_prob) -> float
compute_pr_auc(y_true, y_prob) -> float
plot_roc_curve(y_true, y_prob, save_path)
plot_pr_curve(y_true, y_prob, save_path)
plot_confusion_matrix(cm, save_path)
```

**Usage**:
```python
from scripts.classifier.eval.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred, y_prob)
print(f"F1: {metrics['f1']:.3f}")
```

---

## Utility Scripts

### `scripts/utils/paths.py`

**Purpose**: Centralized path management

**Functions**:
```python
get_project_root() -> Path
get_raw_requirements_dir() -> Path
get_classifier_dir() -> Path
get_step_dir(step: int) -> Path
get_output_dir() -> Path
```

**Usage**:
```python
from scripts.utils.paths import get_project_root, get_step_dir

root = get_project_root()
step2_dir = get_step_dir(2)
```

---

### `scripts/utils/config_log.py`

**Purpose**: Configuration and logging utilities

**Functions**:
```python
setup_logging(log_file: str, level: str = "INFO")
load_config(config_path: str) -> Dict
save_config(config: Dict, save_path: str)
```

**Usage**:
```python
from scripts.utils.config_log import setup_logging, load_config

setup_logging("pipeline.log")
config = load_config("scripts/classifier/config.yaml")
```

---

### `scripts/utils/artifact_registry.py`

**Purpose**: Artifact tracking and versioning

**Functions**:
```python
register_artifact(artifact_id: str, metadata: Dict)
get_artifact(artifact_id: str) -> Dict
list_artifacts(filter: Dict = None) -> List[Dict]
```

**Usage**:
```python
from scripts.utils.artifact_registry import register_artifact

register_artifact(
    "step2_output",
    {
        "path": "classifier/step_2",
        "timestamp": time.time(),
        "num_files": 110
    }
)
```

---

### `scripts/utils/example_utils.py`

**Purpose**: Example data and utilities

**Functions**:
```python
load_seed_examples() -> List[Dict]
get_example_by_category(category: str) -> List[Dict]
validate_example(example: Dict) -> bool
```

---

### `scripts/dataset_normalization/dataset_normalization.py`

**Purpose**: Dataset format normalization

**Usage**:
```bash
python scripts/dataset_normalization/dataset_normalization.py \
  --input classifier/outputs/all_annotations.jsonl \
  --output classifier/datasets/normalized/
```

**Normalization Steps**:
1. Standardize field names
2. Validate labels
3. Remove duplicates
4. Balance classes (optional)
5. Export to standard format

---

### `scripts/requirement-unitizer/merge_and_resplit.py`

**Purpose**: Dataset merging and splitting

**Usage**:
```bash
python scripts/requirement-unitizer/merge_and_resplit.py
```

**Configuration**:
```python
TRAIN_TARGET_SIZE = 4000
RANDOM_SEED = 42
HOLDOUT_FILE = "classifier/outputs/splits/holdout_60.jsonl"
TRAIN_FILE = "classifier/outputs/splits/train_40.jsonl"
```

**Output**:
- `train_4k.jsonl`: 4K training set
- `test_rest.jsonl`: Remaining test set

---

### `scripts/requirement-unitizer/classifier_dataset_unitization/select_main_dataset_subset.py`

**Purpose**: Select representative subset from main dataset

**Usage**:
```bash
python scripts/requirement-unitizer/classifier_dataset_unitization/select_main_dataset_subset.py \
  --size 4000 \
  --strategy stratified
```

**Strategies**:
- `random`: Random sampling
- `stratified`: Stratified by label
- `balanced`: Balance classes
- `diverse`: Maximize diversity

---

## Service Integration

### `services/llm_clients.py`

**Purpose**: LLM API client integrations

**Supported Models**:
- OpenAI GPT-4.1
- OpenAI GPT-5 Mini
- Claude Opus 4
- DeepSeek R1
- GLM-Z1
- LLaMA 3.3 70B

**Functions**:
```python
# Summarization
call_glm_z1_summarizer(prompt_text: str) -> str

# Generation
call_gpt41_model(prompt_text: str) -> str
call_gpt41_model_with_retry(prompt_text: str, retries=3) -> str
call_gpt5_mini_model(prompt_text: str) -> str
call_claude_opus4_model(prompt_text: str) -> str

# Validation
call_deepseek_validator(prompt_text: str) -> dict
call_glm_z1_validator(prompt_text: str) -> str

# Fix Agent
call_llama_70b_fix_agent(prompt_text: str) -> str
```

**Usage**:
```python
from services.llm_clients import call_deepseek_validator

response = call_deepseek_validator(
    "Classify this sentence: The system shall process payments."
)
print(response["verdict"])
```

**Configuration**:
```python
# API endpoints (local services)
GPT41_ENDPOINT = "http://localhost:5003/gpt-41"
DEEPSEEK_ENDPOINT = "http://localhost:5001/validate"
CLAUDE_ENDPOINT = "http://localhost:5008/claude-opus-4"
```

---

### `scripts/requirement-unitizer/synthetic_data_generation/generate_synthetic_non_requirements.py`

**Purpose**: Generate synthetic non-requirement sentences

**Usage**:
```bash
export DEEPSEEK_API_KEY="your-key"
python scripts/requirement-unitizer/synthetic_data_generation/generate_synthetic_non_requirements.py
```

**Categories**:
- BACKGROUND_CONTEXT
- CAPABILITY_DESCRIPTION
- ARCHITECTURE_EXPLANATION
- RATIONALE_JUSTIFICATION
- FEASIBILITY_ANALYSIS
- ROADMAP_PLANNING
- STAKEHOLDER_DESCRIPTION
- MARKETING_LANGUAGE
- AMBIGUOUS_HYBRID
- RANDOM_SINGLE_WORDS
- BROKEN_PHRASES
- MARKDOWN_ARTIFACTS
- BULLET_FRAGMENTS
- HEADINGS_TITLES
- LINKS_REFERENCES
- CODE_SNIPPET_NOISE
- TABLE_ROW_FRAGMENTS
- SEPARATORS_DIVIDERS

**Configuration**:
```python
MODEL_NAME = "deepseek-r1:7b"
TOTAL_SAMPLES = 5000
CHUNK_SIZE = 10
MAX_RETRIES = 3
```

**Output**:
- `classifier/synthetic/non_requirements/synthetic_non_requirements.jsonl`

---

## Script Dependencies

### Common Dependencies
```python
# Core
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# NLP
import spacy
import stanza
from sentence_transformers import SentenceTransformer

# ML
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

# Utils
import yaml
from tqdm import tqdm
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Execution Order

### Complete Pipeline
```bash
# 1. Ingestion
python scripts/requirement-unitizer/scripts/step_0_ingest.py

# 2. Segmentation
python scripts/classifier/extract_sentences_spacy_stanza.py

# 3. Candidate Detection
python scripts/requirement-unitizer/scripts/step_2_req_candidate.py

# 4. Semantic Filtering
python scripts/classifier/embedding_aggregator.py

# 5. LLM Annotation
python scripts/requirement-unitizer/llm_labeling/step_2_llm_reviewer.py

# 6. Dataset Preparation
python scripts/requirement-unitizer/merge_and_resplit.py

# 7. Training
python scripts/classifier/finetune/finetune_deberta.py --track A --context_size 2

# 8. Evaluation
python scripts/classifier/eval/eval_deberta_comp.py
```

---

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Model Download Failures
```bash
# Set cache directory
export TRANSFORMERS_CACHE=.cache/transformers
export SENTENCE_TRANSFORMERS_HOME=.cache/sentence_transformers
```

#### GPU Memory Issues
```bash
# Reduce batch size
python script.py --batch_size 4 --grad_accum 8
```

#### API Rate Limits
```bash
# Add delays between requests
time.sleep(0.5)  # In script
```

---

## Performance Tips

### Parallel Processing
```bash
# Use GNU parallel
find classifier/step_0 -name "raw.txt" | parallel -j 8 process_file {}
```

### Batch Processing
```python
# Process in batches
for batch in tqdm(batched(items, batch_size=32)):
    process_batch(batch)
```

### Caching
```python
# Cache embeddings
@lru_cache(maxsize=10000)
def get_embedding(text):
    return model.encode(text)
```

---
