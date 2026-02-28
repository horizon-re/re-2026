# Research Artifact Summary

This guide documents the training methodology used in the study. Full replication may require GPUs, local models, external APIs, and environment configurations beyond the scope of anonymous review. The documentation is provided for transparency rather than out-of-the-box execution.

## 📦 What Has Been Created

This repository now contains a complete, well-documented research artifact for the paper **"Towards Improving Sentence-Level Requirements Identification through Explicit Local Context Modeling"** (RE 2026).

## 📁 Repository Structure

```
re-2026/
├── README.md                          # Main project overview
├── ARTIFACT_SUMMARY.md               # This file
├── LICENSE                           # MIT License
│
├── docs/                             # Complete documentation
│   ├── README.md                     # Documentation index
│   ├── PIPELINE.md                   # Data processing pipeline (6 stages)
│   ├── SCRIPTS.md                    # All scripts documented
│   ├── DATASET.md                    # Dataset guide and specifications
│   ├── TRAINING.md                   # Model training guide
│   ├── EVALUATION.md                 # Evaluation guide
│
├── 02_raw_requirements/              # Raw requirement documents
│   ├── fintech/                      # 44 FinTech documents
│   └── saas/                         # 66 SaaS documents
│
├── classifier/                       # Processing pipeline outputs
│   ├── step_0/                       # Document ingestion
│   ├── step_1/                       # Raw text extraction
│   ├── step_2/                       # Sentence segmentation
│   ├── step_3/                       # Semantic filtering & embeddings
│   ├── datasets/                     # Processed datasets
│   ├── llm_reviews/                  # LLM annotations
│   ├── manual_annotations/           # Human-verified labels
│   ├── outputs/                      # Training splits & results
│   ├── models/                       # Trained model checkpoints
│   └── synthetic/                    # Synthetic non-requirements
│
├── scripts/                          # All processing & training scripts
│   ├── classifier/                   # Core classification pipeline
│   │   ├── dataset_crawler.py               # Stage 1: Text extraction
│   │   ├── extract_sentences_spacy_stanza.py # Stage 2: Segmentation
│   │   ├── embedding_aggregator.py          # Stage 3: Filtering
│   │   ├── finetune/                        # Training scripts
│   │   └── eval/                            # Evaluation scripts
│   │
│   ├── requirement-unitizer/        # Annotation & labeling tools
│   │   ├── scripts/                         # Step 0-3 processing
│   │   ├── llm_labeling/                   # LLM annotation
│   │   ├── annotation_ui/                  # Manual annotation
│   │   ├── synthetic_data_generation/      # Synthetic data
│   │   └── rules/                          # Rule-based filters
│   │
│   ├── config/                      # Configuration files
│   └── utils/                       # Shared utilities
│
└── services/                        # External service integrations
    └── llm_clients.py              # LLM API clients
```

## 📚 Documentation 

### 1. Main README.md
**Comprehensive project overview including:**
- Research overview and key findings
- Quick start guide
- Repository structure
- Installation instructions
- Usage examples
- Dataset description
- Model training guide
- Evaluation metrics
- Citation information

### 2. docs/README.md
**Documentation index with:**
- Navigation guide
- Quick start for different user types
- Topic-based organization
- Task-based finding
- Common tasks reference

### 3. docs/PIPELINE.md (45+ pages)
**Complete data processing pipeline:**
- Stage 0: Document Ingestion & Canonicalization
- Stage 1: Sentence Segmentation (spaCy + Stanza)
- Stage 2: Requirement Candidate Detection
- Stage 3: Semantic Filtering & Embedding
- Stage 4: LLM-Assisted Annotation
- Stage 5: Dataset Preparation & Splitting
- Stage 6: Context Window Construction

Each stage includes:
- Purpose and overview
- Input/output specifications
- Detailed process description
- Usage examples
- Key features
- Output file formats
- Configuration options

### 4. docs/SCRIPTS.md (50+ pages)
**Complete scripts reference:**
- Data processing scripts (10+ scripts)
- Annotation scripts (5+ scripts)
- Training scripts (5+ scripts)
- Evaluation scripts (5+ scripts)
- Utility scripts (10+ scripts)
- Service integration

Each script includes:
- Purpose
- Input/output
- Usage examples
- Key functions
- Configuration
- Output files

### 5. docs/DATASET.md (40+ pages)
**Comprehensive dataset guide:**
- Dataset overview
- Primary corpus (110 documents, ~5,500 sentences)
- External datasets (DOSSPRE, HuggingFace, Kaggle)
- Synthetic data (5,000 sentences, 18 categories)
- Data format specifications
- Label taxonomy (5 labels)
- Dataset statistics
- Usage guidelines
- Code examples

## 🎯 Key Features

### Complete Pipeline
✅ 6-stage data processing pipeline  
✅ Dual-parser sentence segmentation  
✅ Rule-based candidate detection  
✅ Semantic quality filtering  
✅ LLM-assisted annotation  
✅ Context window construction  

### Comprehensive Dataset
✅ 110 real-world documents  
✅ ~15,000 labeled instances  
✅ 2 domains (FinTech, SaaS)  
✅ 5-class label taxonomy  
✅ Context-aware annotations  
✅ Synthetic data generation  

### Multiple Approaches
✅ Frozen encoder baseline  
✅ LoRA adaptation  
✅ Full fine-tuning  
✅ Structured divergence features  
✅ Context-aware classification  

### Reproducible Research
✅ All scripts documented  
✅ Configuration files included  
✅ Usage examples provided  
✅ Evaluation protocols defined  
✅ Random seeds fixed  

## 📊 Research Contributions

### Empirical Findings
1. **+16 F1 improvement** from adding local context (k=1)
2. **Structured features essential**: Naïve concatenation degrades performance
3. **Distribution > Scale**: 4K domain samples outperform 15K mixed samples
4. **Optimal window**: k=2 consistently best across architectures

### Methodological Contributions
1. **Dual-parser validation**: spaCy + Stanza cross-validation
2. **Semantic quality scoring**: Multi-component keep score
3. **Context-aware annotation**: LLM with few-shot prompting
4. **Structured divergence**: Cosine similarity as discriminative signal

### Dataset Contributions
1. **Real-world corpus**: 110 FinTech and SaaS documents
2. **Context-dependent labels**: `with_context` category
3. **Synthetic non-requirements**: 18 categories, 5,000 samples
4. **Multi-source integration**: Primary + external + synthetic

## 🚀 Usage Scenarios

### For Researchers
```bash
# 1. Understand the methodology
cat docs/PIPELINE.md

# 2. Explore the dataset
cat docs/DATASET.md

# 3. Reproduce experiments
python scripts/classifier/finetune/finetune_deberta.py --track A --context_size 2

# 4. Evaluate results
python scripts/classifier/eval/eval_deberta_comp.py
```

### For Practitioners
```bash
# 1. Process your own documents
python scripts/requirement-unitizer/scripts/step_0_ingest.py --domain your_domain

# 2. Run the pipeline
./run_pipeline.sh

# 3. Train on your data
python scripts/classifier/finetune/finetune_deberta.py --track A

# 4. Evaluate performance
python scripts/classifier/eval/eval_deberta_comp.py
```

### For Developers
```bash
# 1. Review script documentation
cat docs/SCRIPTS.md

# 2. Check training guide
cat docs/TRAINING.md

# 3. Understand data formats
cat docs/DATASET.md

# 4. Extend the pipeline
# Add your own scripts following the documented patterns
```

## 📈 Performance Benchmarks

### Track A (Domain-Only, 4K samples)
- **Best F1**: 0.894
- **Accuracy**: 0.875
- **ROC-AUC**: 0.933
- **Context**: k=2
- **Model**: DeBERTa-v3-base (full fine-tuning)

### Track B (Mixed-Source, 15K samples)
- **Best F1**: 0.883
- **Accuracy**: 0.868
- **ROC-AUC**: 0.933
- **Context**: k=2
- **Model**: DeBERTa-v3-base (full fine-tuning)

### Frozen Encoder Baseline
- **F1**: 0.824 (k=1)
- **Accuracy**: 0.794
- **ROC-AUC**: 0.876
- **Model**: MPNet with structured features

## 🔧 Technical Stack

### NLP Tools
- spaCy 3.x (sentence segmentation)
- Stanza 1.x (cross-validation)
- Sentence-Transformers (embeddings)

### ML Frameworks
- PyTorch 2.x (model training)
- Transformers 4.x (pre-trained models)
- scikit-learn (metrics, utilities)

### Models
- all-mpnet-base-v2 (110M parameters)
- DeBERTa-v3-base (184M parameters)
- all-MiniLM-L6-v2 (context encoding)

### LLM Services
- DeepSeek R1 (annotation)
- OpenAI GPT-4.1 (optional)
- Claude Opus 4 (optional)


