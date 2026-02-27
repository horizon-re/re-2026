# Dataset Guide

Complete documentation for the dataset structure, format specifications, and usage guidelines.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Primary Corpus](#primary-corpus)
- [External Datasets](#external-datasets)
- [Synthetic Data](#synthetic-data)
- [Data Format Specifications](#data-format-specifications)
- [Label Taxonomy](#label-taxonomy)
- [Dataset Statistics](#dataset-statistics)
- [Usage Guidelines](#usage-guidelines)

---

## Dataset Overview

The complete dataset comprises approximately **15,000 labeled instances** from three sources:

| Source | Size | Description |
|--------|------|-------------|
| Primary Corpus | ~5,500 | Real-world FinTech and SaaS documents |
| External Datasets | ~9,000 | DOSSPRE, HuggingFace, Kaggle |
| Synthetic Data | ~5,000 | LLM-generated non-requirements |
| **Total** | **~15,000** | **Balanced training set** |

### Experimental Tracks

**Track A (Domain-Only)**:
- Size: ~4,000 instances
- Source: Primary corpus only
- Purpose: Evaluate domain-aligned training
- Best F1: 0.894

**Track B (Mixed-Source)**:
- Size: ~15,000 instances
- Sources: All three
- Purpose: Evaluate scale vs. distribution
- Best F1: 0.883

---

## Primary Corpus

### Document Collection

**Total Documents**: 110
- **FinTech**: 44 documents (P-FT-001 to P-FT-044)
- **SaaS**: 66 documents (P-SAAS-001 to P-SAAS-081)

### Domain Breakdown

#### FinTech Domain (44 documents)
**Topics**:
- Payment processing systems
- Banking applications
- Financial reporting tools
- Compliance and regulatory systems
- Fraud detection
- Transaction management
- Account management
- Credit scoring systems

**Example Documents**:
- P-FT-001: Payment gateway requirements
- P-FT-010: Mobile banking app
- P-FT-025: Fraud detection system
- P-FT-040: Regulatory compliance platform

#### SaaS Domain (66 documents)
**Topics**:
- Project management tools
- CRM systems
- Collaboration platforms
- Analytics dashboards
- Workflow automation
- Customer support systems
- Marketing automation
- HR management systems

**Example Documents**:
- P-SAAS-001: Project tracking tool
- P-SAAS-020: CRM platform
- P-SAAS-045: Team collaboration suite
- P-SAAS-070: Analytics dashboard

### Document Structure

```
02_raw_requirements/
├── fintech/
│   ├── P-FT-001/
│   │   ├── req-010_raw.txt
│   │   └── example_raw_requirement.txt
│   ├── P-FT-002/
│   │   └── req-011_raw.txt
│   └── ...
└── saas/
    ├── P-SAAS-001/
    │   └── req-090_raw.txt
    ├── P-SAAS-002/
    │   └── req-091_raw.txt
    └── ...
```

### Sentence Statistics

| Metric | Value |
|--------|-------|
| Total Sentences | ~5,500 |
| Avg Sentences/Doc | 50 |
| Avg Tokens/Sentence | 18.3 |
| Min Sentence Length | 3 tokens |
| Max Sentence Length | 127 tokens |

### Content Characteristics

**Stakeholder-Style Prose**:
- Mixed descriptive and prescriptive content
- Business rationale interleaved with requirements
- Informal language with domain terminology
- Varying levels of structure

**Requirement Types**:
- Functional requirements (60%)
- Non-functional requirements (25%)
- Constraints (10%)
- Context-dependent (5%)

---

## External Datasets

### DOSSPRE Corpus

**Source**: [DOSSPRE Dataset](https://github.com/dosspre/dataset)

**Size**: ~3,000 sentences

**Content**:
- Software requirements from open-source projects
- Functional and non-functional requirements
- Well-structured, formal language
- Explicit modal verbs (shall, must, should)

**Label Distribution**:
- Functional: 65%
- Non-functional: 35%

**Usage in This Work**:
- Included in Track B (mixed-source)
- Provides canonical requirement examples
- Helps with FR/NFR classification

### HuggingFace FR/NFR Dataset

**Source**: HuggingFace Datasets Hub

**Size**: ~4,000 sentences

**Content**:
- Requirements from various domains
- Pre-labeled as functional or non-functional
- Mix of formal and informal language

**Label Distribution**:
- Functional: 70%
- Non-functional: 30%

**Quality Notes**:
- Some label noise (~5-10%)
- Varying sentence quality
- Useful for scale but not distribution alignment

### Kaggle Requirements Dataset

**Source**: Kaggle Datasets

**Size**: ~2,000 sentences

**Content**:
- Software requirements from academic projects
- Student-written requirements
- Varying quality and structure

**Label Distribution**:
- Requirements: 80%
- Non-requirements: 20%

**Quality Notes**:
- Higher noise level (~15%)
- Less domain-specific
- Useful for robustness testing

---

## Synthetic Data

### Generation Process

**Model**: deepseek-r1:7b

**Prompt Strategy**:
```
Generate {count} non-requirement sentences in the category {category}.
These should resemble requirement document language but NOT express
system obligations or constraints.

Examples:
[seed examples]

Output format: JSON array of strings
```

**Total Generated**: 5,000 sentences

### Categories (9 types)

#### Semantic Non-Requirements (9 categories)

1. **BACKGROUND_CONTEXT** (~550 samples)
   - Industry statistics
   - Market analysis
   - Historical context
   - Example: "In a recent UK study, 87% of SMB users report better visibility."

2. **CAPABILITY_DESCRIPTION** (~550 samples)
   - General system capabilities
   - Feature descriptions without obligations
   - Example: "The platform emphasizes ease-of-use for non-technical users."

3. **ARCHITECTURE_EXPLANATION** (~550 samples)
   - System architecture descriptions
   - Technology stack explanations
   - Example: "The system uses a microservices architecture."

4. **RATIONALE_JUSTIFICATION** (~550 samples)
   - Business justifications
   - Design rationale
   - Example: "This approach reduces development time."

5. **FEASIBILITY_ANALYSIS** (~550 samples)
   - Technical feasibility discussions
   - Risk assessments
   - Example: "Implementation is feasible within 6 months."

6. **ROADMAP_PLANNING** (~550 samples)
   - Future plans
   - Roadmap items
   - Example: "Phase 2 will include mobile support."

7. **STAKEHOLDER_DESCRIPTION** (~550 samples)
   - User personas
   - Stakeholder descriptions
   - Example: "Target users are small business owners."

8. **MARKETING_LANGUAGE** (~550 samples)
   - Marketing copy
   - Value propositions
   - Example: "Revolutionary platform for modern businesses."

9. **AMBIGUOUS_HYBRID** (~550 samples)
   - Borderline cases
   - Mixed content
   - Example: "The system should ideally support multiple languages."

#### Noise/Artifacts (9 categories)

10. **RANDOM_SINGLE_WORDS** (~280 samples)
    - Single words
    - Example: "Payment"

11. **BROKEN_PHRASES** (~280 samples)
    - Incomplete sentences
    - Example: "when the user clicks"

12. **MARKDOWN_ARTIFACTS** (~280 samples)
    - Markdown syntax
    - Example: "### 1.2.3"

13. **BULLET_FRAGMENTS** (~280 samples)
    - Bullet point fragments
    - Example: "- and then"

14. **HEADINGS_TITLES** (~280 samples)
    - Section headings
    - Example: "User Authentication"

15. **LINKS_REFERENCES** (~280 samples)
    - URLs and references
    - Example: "[See documentation](https://...)"

16. **CODE_SNIPPET_NOISE** (~280 samples)
    - Code fragments
    - Example: "function() { return; }"

17. **TABLE_ROW_FRAGMENTS** (~280 samples)
    - Table cells
    - Example: "| Column 1 | Column 2 |"

18. **SEPARATORS_DIVIDERS** (~280 samples)
    - Divider lines
    - Example: "---"

### Quality Control

**Filtering**:
- Remove sentences with modal verbs (shall, must, should)
- Remove sentences with constraint markers (<=, >=, at least)
- Remove duplicates
- Validate JSON format

**Post-Processing**:
- Strip `<think>` blocks from DeepSeek output
- Clean markdown artifacts
- Normalize whitespace

**Quality Metrics**:
- Avg confidence: 0.89
- Manual verification: 25 samples (~92% acceptable)
- Diversity score: 0.78 (cosine similarity)

---

## Data Format Specifications

### Raw Document Format

**File**: `<req_id>_raw.txt`

**Encoding**: UTF-8

**Structure**: Plain text, may contain:
- Multiple paragraphs
- Bullet points
- Numbered lists
- Markdown formatting
- Mixed content types

**Example**:
```
Payment Processing Requirements

The system shall process credit card payments within 2 seconds.

Background: Current systems take 5-10 seconds, causing user frustration.

The platform will support Visa, Mastercard, and American Express.

Performance requirements:
- 99.9% uptime
- < 2 second response time
- Support 1000 TPS
```

### Sentence Record Format

**File**: `sentences_merged.jsonl`

**Format**: JSON Lines (one JSON object per line)

**Schema**:
```json
{
  "req_id": "req-010",
  "sent_id": "req-010::s001",
  "index": 1,
  "text": "The system shall process payments within 2 seconds.",
  "splitter": "spacy",
  "verified_by_stanza": true,
  "alignment_offset": 2,
  "token_count": 9,
  "char_start": 0,
  "char_end": 52
}
```

**Field Descriptions**:
- `req_id`: Requirement document ID
- `sent_id`: Unique sentence ID (format: `<req_id>::s<index>`)
- `index`: Sentence position in document (1-indexed)
- `text`: Sentence text
- `splitter`: Segmentation tool used (spacy/stanza/merged)
- `verified_by_stanza`: Cross-validation flag
- `alignment_offset`: Character difference between parsers
- `token_count`: Number of tokens
- `char_start`: Start position in original text
- `char_end`: End position in original text

### Candidate Record Format

**File**: `candidates.jsonl`

**Schema**:
```json
{
  "doc_id": "a8b0d6f2-41ab-43f9-9a8c-1234abcd5678",
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

**Signal Descriptions**:
- `action`: Has action verb (VERB with ROOT/conj dependency)
- `modal`: Contains modal verb (shall, must, should, will, may)
- `system`: Has system subject (system, application, service, etc.)
- `constraint`: Contains constraint marker (<=, >=, at least, etc.)

### Annotated Record Format

**File**: `deepseek_reviews.jsonl`

**Schema**:
```json
{
  "sent_id": "req-010::s001",
  "text": "The system shall process payments.",
  "label": "requirement",
  "confidence": 0.95,
  "reasoning": "Clear obligation with modal verb and system subject.",
  "context_used": true,
  "previous_sentence": "Payment processing is critical.",
  "next_sentence": "This ensures user satisfaction.",
  "model": "deepseek-r1:7b",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Training Split Format

**File**: `train_4k.jsonl` / `test_rest.jsonl`

**Schema**:
```json
{
  "sent_id": "req-010::s001",
  "text": "The system shall process payments.",
  "label": 1,
  "original_label": "requirement",
  "domain": "fintech",
  "prompt_id": "P-FT-001",
  "source": "primary_corpus",
  "context_k1_prev": "Payment processing is critical.",
  "context_k1_next": "This ensures user satisfaction.",
  "context_k2_prev": ["Background context.", "Payment processing is critical."],
  "context_k2_next": ["This ensures user satisfaction.", "Performance is key."]
}
```

**Label Mapping**:
- `1`: Requirement (includes `requirement` + `with_context`)
- `0`: Non-requirement
- Excluded: `ambiguous`, `non_ambiguous`

---

## Label Taxonomy

### Primary Labels

#### 1. `requirement`
**Definition**: Clear, standalone requirement statement

**Characteristics**:
- Expresses system obligation or constraint
- Contains modal verbs (shall, must, should)
- Specifies measurable criteria
- Can be verified independently

**Examples**:
```
✓ "The system shall process payments within 2 seconds."
✓ "Users must authenticate before accessing data."
✓ "The API should return JSON responses."
```

#### 2. `non_requirement`
**Definition**: Definitively not a requirement

**Characteristics**:
- Background information
- Rationale or justification
- Descriptive prose
- No system obligation

**Examples**:
```
✓ "In a recent study, 87% of users reported satisfaction."
✓ "This approach reduces development time."
✓ "The market is growing rapidly."
```

#### 3. `with_context`
**Definition**: Requirement identifiable only with surrounding context

**Characteristics**:
- Ambiguous in isolation
- Depends on previous/next sentences
- Implicit obligation
- Requires discourse understanding

**Examples**:
```
Context: "The system shall support multiple payment methods."
Target: "This includes credit cards and PayPal." ← with_context

Context: "Users can customize their dashboard."
Target: "The system shall save these preferences." ← requirement
```

#### 4. `ambiguous`
**Definition**: Unclear classification even with context

**Characteristics**:
- Vague language
- Missing specifics
- Borderline cases
- Subjective interpretation

**Examples**:
```
? "The system should be user-friendly."
? "Performance must be acceptable."
? "The interface will be intuitive."
```

#### 5. `non_ambiguous`
**Definition**: Clear classification (used for quality control)

**Characteristics**:
- Unambiguous requirement or non-requirement
- High annotator agreement
- Used for validation

---

## Dataset Statistics

### Primary Corpus Statistics

```
Total Documents: 110
Total Sentences: 5,501
Total Tokens: 100,718

Domain Distribution:
- FinTech: 44 docs (40%), 2,420 sentences (44%)
- SaaS: 66 docs (60%), 3,081 sentences (56%)

Label Distribution (Primary Corpus):
- requirement: 2,847 (51.8%)
- non_requirement: 1,923 (35.0%)
- with_context: 542 (9.8%)
- ambiguous: 189 (3.4%)

Sentence Length Distribution:
- Min: 3 tokens
- Max: 127 tokens
- Mean: 18.3 tokens
- Median: 16 tokens
- Std Dev: 9.7 tokens

Token Distribution:
- Nouns: 28,345 (28.1%)
- Verbs: 18,127 (18.0%)
- Adjectives: 9,064 (9.0%)
- Adverbs: 4,532 (4.5%)
- Other: 40,650 (40.4%)
```

### Track A Statistics

```
Total Instances: 4,000
Source: Primary corpus only

Label Distribution:
- Positive (requirement + with_context): 2,400 (60%)
- Negative (non_requirement): 1,600 (40%)

Domain Distribution:
- FinTech: 1,760 (44%)
- SaaS: 2,240 (56%)

Split:
- Train: 3,400 (85%)
- Dev: 600 (15%)
- Test: 1,500 (held-out)
```

### Track B Statistics

```
Total Instances: 15,000
Sources: Primary + External + Synthetic

Source Distribution:
- Primary corpus: 5,000 (33.3%)
- DOSSPRE: 3,000 (20.0%)
- HuggingFace: 4,000 (26.7%)
- Kaggle: 2,000 (13.3%)
- Synthetic: 1,000 (6.7%)

Label Distribution:
- Positive: 9,000 (60%)
- Negative: 6,000 (40%)

Split:
- Train: 12,750 (85%)
- Dev: 2,250 (15%)
- Test: 2,200 (held-out from primary corpus)
```

### Context Window Statistics

```
Context Size k=1:
- Avg context length: 36.6 tokens
- Max context length: 254 tokens
- Truncation rate: 2.3%

Context Size k=2:
- Avg context length: 73.2 tokens
- Max context length: 508 tokens
- Truncation rate: 8.7%

Context Size k=3:
- Avg context length: 109.8 tokens
- Max context length: 762 tokens
- Truncation rate: 18.4%
```

---

## Usage Guidelines

### Loading Data

#### Python
```python
import json
from pathlib import Path

# Load JSONL file
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Load training data
train_data = load_jsonl('classifier/outputs/splits/train_4k.jsonl')

# Access fields
for item in train_data:
    print(f"Text: {item['text']}")
    print(f"Label: {item['label']}")
    print(f"Domain: {item['domain']}")
```

#### Pandas
```python
import pandas as pd

# Load as DataFrame
df = pd.read_json('classifier/outputs/splits/train_4k.jsonl', lines=True)

# Basic statistics
print(df['label'].value_counts())
print(df['domain'].value_counts())
print(df['text'].str.len().describe())
```

### Filtering Data

```python
# Filter by domain
fintech_data = [item for item in train_data if item['domain'] == 'fintech']

# Filter by label
requirements = [item for item in train_data if item['label'] == 1]

# Filter by source
primary_only = [item for item in train_data if item['source'] == 'primary_corpus']
```

### Creating Custom Splits

```python
from sklearn.model_selection import train_test_split

# Load data
data = load_jsonl('classifier/outputs/all_annotations.jsonl')

# Extract features and labels
texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, 
    test_size=0.2, 
    stratify=labels, 
    random_seed=42
)
```

### Context Window Construction

```python
def construct_context_window(sentences, target_idx, k=2):
    """
    Construct context window around target sentence.
    
    Args:
        sentences: List of sentence texts
        target_idx: Index of target sentence
        k: Context window size
    
    Returns:
        Context window string with [CTX] separators
    """
    start = max(0, target_idx - k)
    end = min(len(sentences), target_idx + k + 1)
    
    prev_context = sentences[start:target_idx]
    target = sentences[target_idx]
    next_context = sentences[target_idx+1:end]
    
    window = prev_context + ['[CTX]', target, '[CTX]'] + next_context
    return ' '.join(window)

# Example usage
sentences = [
    "Payment processing is critical.",
    "The system shall process payments within 2 seconds.",
    "This ensures user satisfaction."
]
context = construct_context_window(sentences, target_idx=1, k=1)
print(context)
# Output: "Payment processing is critical. [CTX] The system shall process payments within 2 seconds. [CTX] This ensures user satisfaction."
```

### Data Augmentation

```python
# Back-translation augmentation
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src='en', pivot='de'):
    # Translate to pivot language
    model_name = f'Helsinki-NLP/opus-mt-{src}-{pivot}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    translated = model.generate(**tokenizer(text, return_tensors="pt"))
    pivot_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Translate back
    model_name = f'Helsinki-NLP/opus-mt-{pivot}-{src}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    back_translated = model.generate(**tokenizer(pivot_text, return_tensors="pt"))
    return tokenizer.decode(back_translated[0], skip_special_tokens=True)

# Augment data
augmented = [back_translate(item['text']) for item in train_data[:100]]
```

### Quality Checks

```python
def validate_dataset(data):
    """Validate dataset quality."""
    issues = []
    
    # Check for duplicates
    texts = [item['text'] for item in data]
    if len(texts) != len(set(texts)):
        issues.append("Duplicate sentences found")
    
    # Check label distribution
    labels = [item['label'] for item in data]
    pos_ratio = sum(labels) / len(labels)
    if pos_ratio < 0.3 or pos_ratio > 0.7:
        issues.append(f"Imbalanced labels: {pos_ratio:.2%} positive")
    
    # Check sentence length
    lengths = [len(item['text'].split()) for item in data]
    if max(lengths) > 200:
        issues.append(f"Very long sentences: max {max(lengths)} tokens")
    
    # Check missing fields
    required_fields = ['text', 'label', 'sent_id']
    for item in data:
        if not all(field in item for field in required_fields):
            issues.append(f"Missing fields in {item.get('sent_id', 'unknown')}")
            break
    
    return issues

# Validate
issues = validate_dataset(train_data)
if issues:
    print("Dataset issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Dataset validation passed!")
```

---

## Data Access

### Repository Structure
```
classifier/
├── outputs/
│   ├── splits/
│   │   ├── train_4k.jsonl          # Track A training
│   │   ├── test_rest.jsonl         # Track A test
│   │   ├── train_40.jsonl          # Track B training
│   │   └── holdout_60.jsonl        # Track B holdout
│   └── all_annotations.jsonl       # Complete annotated dataset
├── llm_reviews/
│   └── deepseek_reviews.jsonl      # LLM annotations
└── manual_annotations/
    └── all_annotations.jsonl        # Human-verified labels
```

### Download
```bash
# Clone repository
git clone https://github.com/horizon-re/re-2026.git
cd re-2026

# Data is included in repository
ls classifier/outputs/splits/
```

---

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{re2026context,
  title={Towards Improving Sentence-Level Requirements Identification through Explicit Local Context Modeling},
  author={[Authors]},
  booktitle={Proceedings of the 2026 IEEE International Requirements Engineering Conference},
  year={2026}
}
```

---

## License

Dataset is released under MIT License. See [LICENSE](../LICENSE) for details.

---

## Contact

For questions about the dataset:
- Open a GitHub issue
- Contact authors through paper submission system
