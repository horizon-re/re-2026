# Context-Aware Requirements Identification Research Artifact

This guide documents the training methodology used in the study. Full replication may require GPUs, local models, external APIs, and environment configurations beyond the scope of anonymous review. The documentation is provided for transparency rather than out-of-the-box execution.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

This repository contains the complete research artifact for **"Towards Improving Sentence-Level Requirements Identification through Explicit Local Context Modeling"** submitted to IEEE RE 2026.

## 🎯 What is This Artifact?

This is a comprehensive research artifact that demonstrates how incorporating local context (neighboring sentences) dramatically improves automated requirements identification. The artifact includes:

- **Complete Dataset**: ~15,000 labeled requirement sentences from 110 real-world FinTech and SaaS documents
- **Full Pipeline**: End-to-end processing from raw documents to trained models
- **Reproducible Experiments**: All code, configurations, and scripts to reproduce paper results
- **Multiple Approaches**: Frozen encoders, LoRA adaptation, and full fine-tuning implementations
- **Evaluation Framework**: Comprehensive metrics and analysis tools

## 📊 Key Research Findings

| Metric | Sentence-Only | With Context (k=1) | Improvement |
|--------|---------------|-------------------|-------------|
| F1 Score | 0.664 | 0.824 | **+16.0 points** |
| ROC-AUC | 0.645 | 0.876 | **+23.1 points** |
| Accuracy | 0.621 | 0.794 | **+17.3 points** |

**Best Overall Performance**: DeBERTa-v3-base with k=2 context achieves **F1=0.894** (ROC-AUC=0.933)

### Critical Insights

1. **Context is Essential**: Adding just one neighboring sentence improves F1 by 16 points
2. **Structure Matters**: Naïve concatenation degrades performance; structured divergence features (cosine similarity) are critical
3. **Distribution > Scale**: 4K domain-aligned samples outperform 15K mixed samples
4. **Optimal Window**: k=2 (two sentences before and after) consistently performs best

## 🏗️ Repository Structure

```
re-2026/
├── 02_raw_requirements/          # Raw requirement documents (110 docs)
│   ├── fintech/                  # 44 FinTech documents (P-FT-001 to P-FT-044)
│   └── saas/                     # 66 SaaS documents (P-SAAS-001 to P-SAAS-081)
│
├── classifier/                   # Processing pipeline outputs
│   ├── step_0/                   # Document ingestion & canonicalization
│   ├── step_1/                   # Sentence segmentation (spaCy + Stanza)
│   ├── step_2/                   # Requirement candidate detection
│   ├── step_3/                   # Semantic filtering & embeddings
│   ├── datasets/                 # Processed training datasets
│   ├── llm_reviews/             # LLM-assisted annotations
│   ├── manual_annotations/      # Human-verified labels
│   ├── outputs/                 # Training splits & evaluation results
│   ├── models/                  # Trained model checkpoints
│   └── synthetic/               # Synthetic non-requirements
│
├── scripts/                     # All processing & training scripts
│   ├── classifier/              # Core classification pipeline
│   │   ├── dataset_crawler.py           # Stage 1: Raw text extraction
│   │   ├── extract_sentences_spacy_stanza.py  # Stage 2: Segmentation
│   │   ├── embedding_aggregator.py      # Stage 3: Semantic filtering
│   │   ├── finetune/                    # Model training scripts
│   │   └── eval/                        # Evaluation scripts
│   │
│   ├── requirement-unitizer/    # Annotation & labeling tools
│   │   ├── scripts/                     # Step 0-3 processing
│   │   ├── llm_labeling/               # LLM-assisted annotation
│   │   ├── annotation_ui/              # Manual annotation interface
│   │   ├── synthetic_data_generation/  # Synthetic data generation
│   │   └── rules/                      # Rule-based filters
│   │
│   ├── config/                  # Configuration files
│   └── utils/                   # Shared utilities
│
├── services/                    # External service integrations
│   └── llm_clients.py          # LLM API clients (OpenAI, DeepSeek, etc.)
│
├── docs/                        # Detailed documentation (see below)
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install torch transformers sentence-transformers
pip install spacy stanza scikit-learn scipy numpy pandas
pip install tqdm pyyaml jsonlines

# Download NLP models
python -m spacy download en_core_web_sm
# Stanza will auto-download on first use
```

### Running the Pipeline

```bash
# 1. Clone the repository
git clone https://github.com/horizon-re/re-2026.git
cd re-2026

# 2. Run the complete pipeline
./run_pipeline.sh

# Or run individual stages:
python scripts/requirement-unitizer/scripts/step_0_ingest.py
python scripts/classifier/dataset_crawler.py
python scripts/classifier/extract_sentences_spacy_stanza.py
python scripts/classifier/embedding_aggregator.py
```

### Training Models

```bash
# Frozen encoder baseline (Phase 1)
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py

# Full fine-tuning (DeBERTa)
python scripts/classifier/finetune/finetune_deberta.py --context_size 2

# Evaluation
python scripts/classifier/eval/eval_deberta_comp.py
```

## 📖 Documentation

Detailed documentation is available in the `docs/` folder:

- **[Pipeline Overview](docs/PIPELINE.md)**: Complete data processing workflow
- **[Scripts Reference](docs/SCRIPTS.md)**: Detailed documentation for each script
- **[Dataset Guide](docs/DATASET.md)**: Dataset structure and format specifications
- **[Model Training](docs/TRAINING.md)**: Training configurations and hyperparameters
- **[Evaluation Guide](docs/EVALUATION.md)**: Metrics and evaluation protocols

## 🔬 Research Methodology

This artifact implements the methodology described in Section 3 of the paper:

### Data Processing Pipeline

1. **Document Ingestion** (Step 0): Canonicalize raw documents with stable IDs
2. **Sentence Segmentation** (Step 1): Cross-validate with spaCy and Stanza
3. **Candidate Detection** (Step 2): Rule-based filtering with signal analysis
4. **Semantic Filtering** (Step 3): Quality scoring and embedding generation
5. **LLM Annotation**: Context-aware few-shot labeling with manual verification
6. **Dataset Preparation**: Track A (domain-only) and Track B (mixed-source)

### Context Window Construction

```
s̃ᵢ = [s_{i-k}, ..., s_{i-1}] [CTX] [sᵢ] [CTX] [s_{i+1}, ..., s_{i+k}]
```

Where k ∈ {1, 2, 3} represents the number of neighboring sentences.

### Model Architectures

- **MPNet** (all-mpnet-base-v2): 110M parameters, mean pooling
- **DeBERTa-v3-base**: 184M parameters, [CLS] token pooling

### Training Configurations

- **Frozen Encoder**: Feature-based with structured divergence
- **LoRA Adaptation**: Rank r ∈ {8, 16, 32}
- **Full Fine-Tuning**: End-to-end with context-aware classification head

## 📊 Dataset

### Primary Corpus
- **110 documents**: 44 FinTech + 66 SaaS
- **~5,500 sentences**: Real-world stakeholder-style requirements
- **Domains**: Payment processing, accounting, CRM, project management

### Label Taxonomy
- `requirement`: Clear standalone requirement
- `non_requirement`: Definitively not a requirement
- `with_context`: Requires surrounding context for identification
- `ambiguous`: Unclear even with context
- `non_ambiguous`: Clear classification

### External Datasets
- DOSSPRE corpus
- HuggingFace FR/NFR dataset
- Kaggle requirements dataset
- **Total**: ~9,000 additional sentences

### Synthetic Data
- **5,000 non-requirements**: Generated using deepseek-r1:7b
- **9 categories**: Background, rationale, noise, artifacts, etc.
- **Controlled generation**: Filtered to avoid modal verbs and constraints

## 🎯 Reproducing Paper Results

### Track A (Domain-Only, 4K samples)
```bash
python scripts/classifier/finetune/finetune_deberta.py \
  --track A \
  --context_size 2 \
  --epochs 15 \
  --lr 5e-6
```

**Expected Results**: F1=0.894, Acc=0.875, ROC-AUC=0.933

### Track B (Mixed-Source, 15K samples)
```bash
python scripts/classifier/finetune/finetune_deberta.py \
  --track B \
  --context_size 2 \
  --epochs 15 \
  --lr 5e-6
```

**Expected Results**: F1=0.883, Acc=0.868, ROC-AUC=0.933

### Frozen Encoder Baseline
```bash
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --context_size 1
```

**Expected Results**: F1=0.824, Acc=0.794, ROC-AUC=0.876

## 📈 Evaluation Metrics

All experiments report:
- **Accuracy**: Overall correctness
- **Precision**: Fraction of predicted requirements that are true
- **Recall**: Fraction of true requirements identified
- **F1-score**: Harmonic mean (primary metric)
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

## 🔧 Configuration

### Key Configuration Files

- `scripts/classifier/config.yaml`: Classifier pipeline settings
- `scripts/config/paths.yaml`: Path definitions
- `scripts/config/artifacts.yaml`: Artifact registry
- `scripts/requirement-unitizer/rules/*.yaml`: Rule-based filters

### Environment Variables

```bash
# For LLM-assisted annotation (optional)
export OPENAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: DOSSPRE, HuggingFace, Kaggle communities
- **Models**: Hugging Face Transformers, Sentence-Transformers
- **NLP Tools**: spaCy, Stanza
- **LLM Services**: OpenAI, DeepSeek

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Dataset**: Available in this repository
- **Permanent Archive**: Will be provided upon acceptance

---

**Last Updated**: February 2026  
**Status**: Research Artifact for RE 2026 Submission
