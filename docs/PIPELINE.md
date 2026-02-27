# Data Processing Pipeline

This document provides a comprehensive overview of the complete data processing pipeline used in the research.

## Pipeline Overview

The pipeline consists of 6 main stages that transform raw requirement documents into training-ready datasets:

```
Raw Documents → Ingestion → Segmentation → Candidate Detection → 
Semantic Filtering → LLM Annotation → Dataset Preparation
```

## Stage 0: Document Ingestion & Canonicalization

**Script**: `scripts/requirement-unitizer/scripts/step_0_ingest.py`

### Purpose
Creates lossless, traceable per-requirement packets with stable identifiers.

### Input
- Raw requirement documents from `02_raw_requirements/`
- Format: `<domain>/<prompt>/<req_id>_raw.txt`

### Process
1. Reads raw requirement text files
2. Generates stable document IDs using UUID v5
3. Creates canonical directory structure
4. Attaches metadata and checksums

### Output Structure
```
classifier/step_0/<domain>/<prompt>/<req_id>/
├── raw.txt           # Original raw text
├── hlj.json          # High-level JSON packet
└── meta.json         # Metadata (checksums, counts, timestamps)
```

### HLJ (High-Level JSON) Format
```json
{
  "doc_id": "uuid-v5-hash",
  "req_key": "domain::prompt::req_id",
  "req_id": "REQ-001",
  "domain": "fintech",
  "prompt_id": "P-FT-001",
  "source": {
    "raw_path": "02_raw_requirements/fintech/P-FT-001/req-010_raw.txt",
    "refined_path": null,
    "origin": "raw_only"
  },
  "text": "The system shall...",
  "structured_data": null
}
```

### Usage
```bash
# Process all documents
python scripts/requirement-unitizer/scripts/step_0_ingest.py

# Filter by domain
python scripts/requirement-unitizer/scripts/step_0_ingest.py --domain fintech

# Filter by prompt
python scripts/requirement-unitizer/scripts/step_0_ingest.py --prompt P-FT-001

# Limit for testing
python scripts/requirement-unitizer/scripts/step_0_ingest.py --limit 10
```

### Key Features
- **Stable IDs**: UUID v5 ensures reproducible identifiers
- **Lossless**: Preserves original text exactly
- **Traceable**: Full provenance tracking
- **Checksums**: SHA-256 for integrity verification

---

## Stage 1: Sentence Segmentation

**Script**: `scripts/classifier/extract_sentences_spacy_stanza.py`

### Purpose
Splits requirement text into clean, validated sentences using dual-parser approach.

### Input
- `classifier/step_1/<domain>/<prompt>/<req_id>_raw.txt`

### Process
1. **spaCy Segmentation**: Uses dependency parsing
2. **Stanza Segmentation**: Uses neural tokenization
3. **Cross-Validation**: Compares outputs for agreement
4. **Merge**: Creates validated sentence list

### Output Structure
```
classifier/step_2/<domain>/<prompt>/<req_id>/
├── sentences_spacy.jsonl      # spaCy output
├── sentences_stanza.jsonl     # Stanza output
├── merged_sentences.jsonl     # Validated merge
└── _manifest.jsonl            # Segmentation statistics
```

### Sentence Record Format
```json
{
  "req_id": "req-010",
  "sent_id": "req-010::s001",
  "index": 1,
  "text": "The system shall process payments within 2 seconds.",
  "splitter": "spacy",
  "verified_by_stanza": true,
  "alignment_offset": 2,
  "token_count": 9
}
```

### Agreement Metrics
- **Diff Ratio**: `|len(spacy) - len(stanza)| / max(len(spacy), len(stanza))`
- **Agreement**: Diff ratio < 0.1 and < 10% misaligned sentences
- **Misalignment**: Character length difference > 15

### Usage
```bash
python scripts/classifier/extract_sentences_spacy_stanza.py
```

### Key Features
- **Dual Validation**: Two independent parsers
- **Stable IDs**: Format `<req_id>::s<index>`
- **Quality Metrics**: Agreement tracking
- **Robust**: Handles encoding issues gracefully

---

## Stage 2: Requirement Candidate Detection

**Script**: `scripts/requirement-unitizer/scripts/step_2_req_candidate.py`

### Purpose
Identifies sentences that are likely to be requirements using rule-based signals.

### Input
- `classifier/step_2/<domain>/<prompt>/<req_id>/merged_sentences.jsonl`

### Detection Signals

#### 1. Action Verb
```python
has_action_verb(doc) -> bool
# Checks for VERB with ROOT or conj dependency
```

#### 2. Modal Verb
```python
has_modal(text) -> bool
# Regex: shall|must|should|will|may|required to
```

#### 3. System Subject
```python
has_system_subject(doc) -> bool
# Checks for nsubj in: system, application, service, api, platform
```

#### 4. Constraint Marker
```python
has_constraint(text) -> bool
# Regex: <=|>=|<|>|at least|no more than|ms|sec|%|tps
```

### Acceptance Rule
```python
is_candidate = action_verb AND (modal OR system_subject OR constraint)
```

### Output Structure
```
classifier/step_2/<domain>/<prompt>/<req_id>/
├── candidates.jsonl           # Accepted sentences
├── non_candidates.jsonl       # Rejected sentences
├── signal_matrix.json         # Per-requirement metrics
└── candidate_manifest.json    # Summary statistics
```

### Candidate Record Format
```json
{
  "doc_id": "uuid",
  "req_id": "req-010",
  "sent_id": "req-010::s001",
  "order": 1,
  "text": "The system shall process payments.",
  "is_candidate": true,
  "signals": {
    "action": true,
    "modal": true,
    "system": true,
    "constraint": false
  }
}
```

### Global Metrics
```json
{
  "total_requirements": 110,
  "total_sentences": 5501,
  "total_candidates": 3247,
  "global_acceptance_rate": 0.590,
  "signal_acceptance_rates": {
    "action": 0.892,
    "modal": 0.673,
    "system": 0.445,
    "constraint": 0.234
  }
}
```

### Usage
```bash
python scripts/requirement-unitizer/scripts/step_2_req_candidate.py
```

### Key Features
- **Permissive Gate**: Avoids discarding valid requirements
- **Signal Analysis**: Tracks co-occurrence patterns
- **Domain Metrics**: Per-domain acceptance rates
- **Audit Trail**: Full signal matrix for each sentence

---

## Stage 3: Semantic Filtering & Embedding

**Script**: `scripts/classifier/embedding_aggregator.py`

### Purpose
Filters low-quality sentences and generates semantic embeddings.

### Input
- `classifier/step_2/<domain>/<prompt>/<req_id>/merged_sentences.jsonl`

### Quality Evaluation

#### Composite Keep Score
```python
keep_score = (0.35 × norm) + (0.20 × noun_ratio) + 
             (0.20 × unique_ratio) + (0.15 × entropy) + 
             (0.10 × context_sim)
```

#### Components
1. **Semantic Norm**: Embedding magnitude (content density)
2. **Noun Ratio**: Proportion of nouns/proper nouns
3. **Unique Ratio**: Lexical diversity
4. **Entropy**: Information content
5. **Context Similarity**: Deviation from surrounding sentences

#### Adaptive Thresholds
- **Long sentences** (≥6 words): threshold = 0.55
- **Short sentences** (<6 words): threshold = 0.65

### Smart Cleaning Rules
Filters out:
- Markdown headers (`### 1.`)
- Symbol-only lines (`***`)
- Link-only lines
- Placeholders (`[user query]`)
- Numeric islands (`42`)
- URL lines

### Embedding Generation

#### Models Used
- **Primary**: `all-mpnet-base-v2` (768-dim)
- **Context**: `all-MiniLM-L6-v2` (384-dim)

#### SIF Weighting
```python
weight = (a / (a + freq)) × log(1 + length) × (norm + ε)
```
Where `a = 1e-3` (smoothing parameter)

#### Document Embedding
```python
doc_emb = Σ(sentence_emb × normalized_weight)
```

### PCA De-biasing
Removes top-1 principal component to reduce corpus-level bias:
```python
X_debiased = X - X @ u[:, None] * u[None, :]
```

### Output Structure
```
classifier/step_3/<domain>/<prompt>/<req_id>/
├── kept_sentences.jsonl       # High-quality sentences
├── dropped_sentences.jsonl    # Filtered sentences
├── sentence_embeddings.npy    # MPNet embeddings (N×768)
├── doc_embedding.npy          # SIF-weighted doc vector (768)
├── top_sentences.json         # Top-5 by SIF weight
└── _pca_meta.json            # PCA component info
```

### Kept Sentence Format
```json
{
  "text": "The system shall process payments.",
  "keep_score": 0.782,
  "semantic_norm": 0.891,
  "noun_ratio": 0.444,
  "context_sim": 0.623
}
```

### Dropped Sentence Format
```json
{
  "text": "### 1.",
  "reason": "markdown_number_header",
  "semantic_norm": 0.123,
  "noun_ratio": 0.000,
  "context_sim": 0.234
}
```

### Usage
```bash
python scripts/classifier/embedding_aggregator.py
```

### Key Features
- **Context-Aware**: Uses surrounding sentences for quality assessment
- **Adaptive**: Different thresholds for different sentence lengths
- **Explainable**: Detailed scores for each decision
- **Efficient**: Batch processing with caching

---

## Stage 4: LLM-Assisted Annotation

**Script**: `scripts/requirement-unitizer/llm_labeling/step_2_llm_reviewer.py`

### Purpose
Labels sentences using LLM with context-aware few-shot prompting.

### Annotation Protocol

#### Stage 1: Context-Free Labeling
Initial attempt with sentence-only prompts (revealed instability).

#### Stage 2: Context-Aware Few-Shot
Uses 50 manually curated seed examples with structured prompts.

#### Stage 3: Manual Verification
Random sampling of 25 outputs (~90% agreement).

### Prompt Structure
```
You are an expert requirements engineer. Given a sentence and its context,
classify it as one of:
- requirement: Clear standalone requirement
- non_requirement: Definitively not a requirement
- with_context: Requirement only identifiable with context
- ambiguous: Unclear even with context
- non_ambiguous: Clear classification

Examples:
[50 seed examples]

Sentence: {target_sentence}
Previous: {previous_sentence}
Next: {next_sentence}

Provide:
1. Label
2. Confidence (0-1)
3. Reasoning
```

### LLM Output Format
```json
{
  "sent_id": "req-010::s001",
  "text": "The system shall process payments.",
  "label": "requirement",
  "confidence": 0.95,
  "reasoning": "Clear obligation with modal verb and system subject.",
  "context_used": true,
  "model": "deepseek-r1:7b"
}
```

### Output Structure
```
classifier/llm_reviews/
├── deepseek_reviews.jsonl              # All LLM annotations
├── _deepseek_review_metrics.json       # Aggregate statistics
└── by_req/
    └── <domain>/<prompt>/<req_id>.jsonl
```

### Quality Metrics
```json
{
  "total_reviewed": 5501,
  "label_distribution": {
    "requirement": 2847,
    "non_requirement": 1923,
    "with_context": 542,
    "ambiguous": 189
  },
  "avg_confidence": 0.847,
  "manual_agreement": 0.902
}
```

### Usage
```bash
# Requires API keys
export DEEPSEEK_API_KEY="your-key"
python scripts/requirement-unitizer/llm_labeling/step_2_llm_reviewer.py
```

### Key Features
- **Context-Aware**: Includes neighboring sentences
- **Few-Shot**: 50 curated examples
- **Confidence Scores**: Enables filtering
- **Reasoning**: Explainable decisions
- **Batch Processing**: Efficient API usage

---

## Stage 5: Dataset Preparation & Splitting

**Script**: `scripts/requirement-unitizer/merge_and_resplit.py`

### Purpose
Creates experimental tracks with proper train/dev/test splits.

### Track Definitions

#### Track A: Domain-Only (4K samples)
- **Source**: Primary corpus only
- **Size**: ~4,000 instances
- **Purpose**: Evaluate domain-aligned training

#### Track B: Mixed-Source (15K samples)
- **Sources**:
  - Primary corpus: ~5,500
  - External datasets: ~9,000
  - Synthetic non-requirements: ~5,000
- **Purpose**: Evaluate scale vs. distribution

### Split Strategy

#### Stratified Sentence-Level Splitting
```python
train: 85% (stratified by label)
dev: 15% (stratified by label)
test: held-out corpus subset (~1,500-2,200)
```

#### Label Mapping
```python
positive (1): requirement + with_context
negative (0): non_requirement
excluded: ambiguous, non_ambiguous
```

### Output Structure
```
classifier/outputs/splits/
├── train_4k.jsonl          # Track A training
├── test_rest.jsonl         # Track A test
├── train_40.jsonl          # Track B training
├── holdout_60.jsonl        # Track B holdout
└── splits_frnfr/           # FR/NFR splits (multi-task)
```

### Split Record Format
```json
{
  "sent_id": "req-010::s001",
  "text": "The system shall process payments.",
  "label": 1,
  "original_label": "requirement",
  "domain": "fintech",
  "prompt_id": "P-FT-001",
  "source": "primary_corpus"
}
```

### Usage
```bash
python scripts/requirement-unitizer/merge_and_resplit.py
```

### Key Features
- **Stratified**: Preserves label distribution
- **Reproducible**: Fixed random seed (42)
- **Flexible**: Easy to create new splits
- **Documented**: Full provenance tracking

---

## Stage 6: Context Window Construction

### Purpose
Augments sentences with neighboring context for training.

### Window Definition
```
s̃ᵢ = [s_{i-k}, ..., s_{i-1}] [CTX] [sᵢ] [CTX] [s_{i+1}, ..., s_{i+k}]
```

Where:
- `k ∈ {1, 2, 3}`: Context window size
- `[CTX]`: Separator token
- `sᵢ`: Target sentence to classify

### Implementation
Context construction is integrated into training scripts:
- `scripts/classifier/finetune/finetune_deberta.py`
- `scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py`

### Example
```python
# k=2 example
context_window = [
    "Payment processing is critical.",      # s_{i-2}
    "The system must be reliable.",         # s_{i-1}
    "[CTX]",
    "The system shall process payments within 2 seconds.",  # sᵢ (target)
    "[CTX]",
    "This ensures user satisfaction.",      # s_{i+1}
    "Performance metrics will be tracked."  # s_{i+2}
]
```

### Truncation Strategy
- **Max length**: 256 tokens
- **Truncation**: From outer boundaries
- **Preservation**: Target sentence always included

---

## Pipeline Execution

### Full Pipeline
```bash
# Run all stages sequentially
./run_pipeline.sh
```

### Individual Stages
```bash
# Stage 0
python scripts/requirement-unitizer/scripts/step_0_ingest.py

# Stage 1 (alternative to extract_sentences_spacy_stanza.py)
python scripts/requirement-unitizer/scripts/step_1_sentence_segmentation.py

# Stage 2
python scripts/requirement-unitizer/scripts/step_2_req_candidate.py

# Stage 3
python scripts/classifier/embedding_aggregator.py

# Stage 4
python scripts/requirement-unitizer/llm_labeling/step_2_llm_reviewer.py

# Stage 5
python scripts/requirement-unitizer/merge_and_resplit.py
```

### Monitoring Progress
Each stage produces:
- **Console output**: Real-time progress
- **Manifest files**: Summary statistics
- **Log files**: Detailed execution logs

---

## Data Flow Diagram

```
02_raw_requirements/
    ↓ [Step 0: Ingestion]
classifier/step_0/
    ↓ [Step 1: Segmentation]
classifier/step_2/
    ↓ [Step 2: Candidate Detection]
classifier/step_2/ (with candidates)
    ↓ [Step 3: Semantic Filtering]
classifier/step_3/
    ↓ [Step 4: LLM Annotation]
classifier/llm_reviews/
    ↓ [Step 5: Dataset Preparation]
classifier/outputs/splits/
    ↓ [Step 6: Context Construction]
Training-Ready Datasets
```

---

## Quality Assurance

### Validation Checks
1. **Segmentation Agreement**: spaCy vs. Stanza
2. **Signal Consistency**: Co-occurrence patterns
3. **Embedding Quality**: Norm distribution
4. **Label Agreement**: LLM vs. manual
5. **Split Balance**: Label distribution

### Audit Trails
- **Checksums**: SHA-256 for all files
- **Provenance**: Full source tracking
- **Metrics**: Per-stage statistics
- **Manifests**: Comprehensive metadata

---

## Troubleshooting

### Common Issues

#### Segmentation Disagreement
```bash
# Check manifest for agreement metrics
cat classifier/step_2/_manifest.jsonl | jq '.agreement'
```

#### Low Candidate Rate
```bash
# Review signal matrix
cat classifier/step_2/_global_metrics.json
```

#### Embedding Errors
```bash
# Check for NaN values
python -c "import numpy as np; print(np.isnan(np.load('path/to/embeddings.npy')).any())"
```

#### LLM API Failures
```bash
# Check API key
echo $DEEPSEEK_API_KEY

# Review error logs
tail -f classifier/llm_reviews/errors.log
```

---

## Performance Optimization

### Parallel Processing
```bash
# Use GNU parallel for stage 0-3
find 02_raw_requirements -name "*_raw.txt" | parallel -j 8 process_file {}
```

### Batch Size Tuning
```python
# In embedding_aggregator.py
BATCH_SIZE = 32  # Adjust based on GPU memory
```

### Caching
```bash
# Enable model caching
export TRANSFORMERS_CACHE=.cache/transformers
export SENTENCE_TRANSFORMERS_HOME=.cache/sentence_transformers
```

---

## Next Steps

After completing the pipeline:
1. Review quality metrics in manifest files
2. Inspect sample outputs for correctness
3. Proceed to model training (see [TRAINING.md](TRAINING.md))
4. Evaluate results (see [EVALUATION.md](EVALUATION.md))
