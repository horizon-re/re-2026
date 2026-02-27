#!/usr/bin/env python3
"""
Step 3.5 — Semantic Quality Aggregator 
---------------------------------------------------------------------
Consumes Step 2 sentence splits, encodes with MPNet + MiniLM,
evaluates structural and semantic quality, and produces refined embeddings.

Adds:
- Context-aware filtering (semantic + grammatical + contextual)
- Adaptive thresholding for short/long sentences
- Detailed logging (keep_score, semantic_norm, noun_ratio, etc.)
"""

import os, json, time, re, hashlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
STEP2_DIR = ROOT / "classifier" / "step_2"
STEP3_DIR = ROOT / "classifier" / "step_3"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SIF_A = 1e-3
os.makedirs(STEP3_DIR, exist_ok=True)

# Embedding models
MAIN_MODEL = SentenceTransformer(MODEL_NAME)
MINI_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def l2_normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / norm

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def strip_links(text):
    return re.sub(r"\(https?://[^\s)]+\)", "", text).strip()

# ---------------------------------------------------------------------
# SMART CLEANING RULES
# ---------------------------------------------------------------------
SMART_RULES = [
    (r"^\s*#+\s*\d+\.?\s*$", "markdown_number_header"),
    (r"^\s*\*{1,2}\.?$", "markdown_symbol_only"),
    (r"^\s*[-=*]{2,}\s*$", "markdown_divider"),
    (r"^\s*\|?\s*\[.*\]\(https?://.*\)\)?$", "link_only_line"),
    (r"^\s*\[user query\]\s*$", "placeholder_userquery"),
    (r"^\s*\[.*\]$", "placeholder_bracket"),
    (r"^[\W_]+$", "symbol_only"),
    (r"^\W*\d{1,2}\W*$", "numeric_island"),
    (r"https?://\S{10,}", "url_line"),
]

def passes_basic_filters(text):
    t = text.strip()
    if not t:
        return False, "empty"
    for pattern, reason in SMART_RULES:
        if re.search(pattern, t, re.IGNORECASE):
            return False, reason
    return True, "ok_basic"

# ---------------------------------------------------------------------
# SEMANTIC QUALITY EVALUATOR
# ---------------------------------------------------------------------
def evaluate_sentence_quality(sent, emb, nlp, context_embs):
    """Return composite keep score from multiple signals."""
    doc = nlp(sent)
    tokens = [t for t in doc if t.is_alpha]
    len_tokens = len(tokens)
    if len_tokens == 0:
        return 0.0, 0, 0, 0

    norm = np.linalg.norm(emb)
    noun_ratio = sum(1 for t in tokens if t.pos_ in {"NOUN", "PROPN"}) / len_tokens
    unique_ratio = len(set(t.lemma_.lower() for t in tokens)) / len_tokens
    entropy = -sum((tokens.count(t)/len_tokens) * np.log(tokens.count(t)/len_tokens)
                   for t in set(tokens))
    entropy = entropy / np.log(len_tokens + 1)

    # Contextual similarity
    if context_embs is not None and len(context_embs) > 0:
        sims = [1 - cosine(emb, c) for c in context_embs if np.any(c)]
        context_sim = np.mean(sims) if sims else 0.5
    else:
        context_sim = 0.5

    # Weighted composite score
    keep_score = (0.35 * norm) + (0.2 * noun_ratio) + (0.2 * unique_ratio) + \
                 (0.15 * entropy) + (0.1 * context_sim)
    return keep_score, norm, noun_ratio, context_sim

# ---------------------------------------------------------------------
# WEIGHTING + EMBEDDINGS
# ---------------------------------------------------------------------
def build_word_freqs(sent_texts, nlp):
    freq = Counter()
    for txt in sent_texts:
        doc = nlp(txt)
        for t in doc:
            if t.is_alpha and not t.is_stop:
                freq[t.lemma_.lower()] += 1
    return freq

def compute_hybrid_weights(sent_texts, freq, embs, a=SIF_A):
    sif = []
    for txt in sent_texts:
        toks = [t.lower() for t in txt.split()]
        fs = [freq.get(t, 1) for t in toks]
        invs = [a / (a + f) for f in fs] if toks else [1.0]
        sif.append(np.mean(invs) / max(len(toks), 1))
    sif = np.array(sif, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1)
    lens = np.array([len(s.split()) for s in sent_texts])
    hybrid = sif * np.log1p(lens) * (norms + 1e-6)
    return hybrid

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[init] Loading MPNet + spaCy for semantic analysis…")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    doc_vecs, manifest = [], []

    for domain_path in tqdm(list(STEP2_DIR.iterdir()), desc="Domains"):
        if not domain_path.is_dir(): continue
        for prompt_path in domain_path.iterdir():
            if not prompt_path.is_dir(): continue
            for req_dir in prompt_path.iterdir():
                merged_path = req_dir / "merged_sentences.jsonl"
                if not merged_path.exists(): continue

                req_id = req_dir.name
                out_dir = STEP3_DIR / domain_path.name / prompt_path.name / req_id
                out_dir.mkdir(parents=True, exist_ok=True)

                raw_sents = [r["text"] for r in safe_read_jsonl(merged_path)]
                if not raw_sents: continue

                context_embs = MINI_MODEL.encode(raw_sents[:20], normalize_embeddings=True)
                kept, dropped = [], []

                for s in raw_sents:
                    s = strip_links(s)
                    passes, reason = passes_basic_filters(s)
                    if not passes:
                        dropped.append({"text": s, "reason": reason})
                        continue

                    emb = MINI_MODEL.encode(s, normalize_embeddings=True)
                    keep_score, norm, noun_ratio, context_sim = evaluate_sentence_quality(s, emb, nlp, context_embs)

                    # Adaptive thresholds
                    word_count = len(s.split())
                    min_thresh = 0.55 if word_count >= 6 else 0.65

                    if keep_score < min_thresh:
                        dropped.append({
                            "text": s,
                            "reason": f"low_semantic_score_{keep_score:.2f}",
                            "semantic_norm": float(round(float(norm), 3)),
                            "noun_ratio": float(round(float(noun_ratio), 3)),
                            "context_sim": float(round(float(context_sim), 3))
                        })
                    else:
                        kept.append({
                            "text": s,
                            "keep_score": float(round(float(keep_score), 3)),
                            "semantic_norm": float(round(float(norm), 3)),
                            "noun_ratio": float(round(float(noun_ratio), 3)),
                            "context_sim": float(round(float(context_sim), 3))
                        })


                if not kept:
                    continue

                dropped_count, total = len(dropped), len(raw_sents)
                ratio = (dropped_count / total) * 100 if total else 0

                with open(out_dir / "kept_sentences.jsonl", "w", encoding="utf-8") as kf:
                    for v in kept: kf.write(json.dumps(v) + "\n")
                if dropped:
                    with open(out_dir / "dropped_sentences.jsonl", "w", encoding="utf-8") as df:
                        for d in dropped: df.write(json.dumps(d) + "\n")

                print(f"[filter] {req_id} → kept {len(kept)}/{total} ({100 - ratio:.1f}%)")

                # --- Embedding + SIF
                sentences = [v["text"] for v in kept]
                embs = MAIN_MODEL.encode(sentences, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
                freqs = build_word_freqs(sentences, nlp)
                sif_w = compute_hybrid_weights(sentences, freqs, embs)
                w_norm = sif_w / (sif_w.sum() + 1e-8)
                doc_emb = np.sum(embs * w_norm[:, None], axis=0)
                doc_vecs.append(doc_emb)

                np.save(out_dir / "sentence_embeddings.npy", embs.astype(np.float32))
                np.save(out_dir / "doc_embedding.npy", doc_emb.astype(np.float32))

                # Top-5 by weight
                top_idx = [i for i in np.argsort(-sif_w) if len(sentences[i].split()) > 3][:5]
                top_records = [
                    {"rank": i+1, "sentence": sentences[idx], "weight": float(sif_w[idx])}
                    for i, idx in enumerate(top_idx)
                ]
                json.dump(top_records, open(out_dir / "top_sentences.json", "w", encoding="utf-8"), indent=2)

                mean_score = np.mean([v["keep_score"] for v in kept])
                manifest.append({
                    "req_id": req_id,
                    "domain": domain_path.name,
                    "prompt_id": prompt_path.name,
                    "num_sentences": len(sentences),
                    "num_dropped": dropped_count,
                    "drop_ratio": ratio,
                    "mean_keep_score": round(mean_score, 3),
                    "embedding_dim": int(doc_emb.shape[0]),
                    "path": str(out_dir.relative_to(ROOT)),
                    "checksum": hash_text(req_id + domain_path.name)
                })

    # -----------------------------------------------------------------
    # PCA DE-BIASING
    # -----------------------------------------------------------------
    if not doc_vecs: return
    print(f"[pca] Removing top-1 component across {len(doc_vecs)} docs…")
    X = np.stack(doc_vecs)
    pca = PCA(n_components=1).fit(X)
    u = pca.components_[0]
    X_debiased = X - X @ u[:, None] * u[None, :]
    X_norm = l2_normalize(X_debiased)

    pca_meta = {
        "component": u.tolist(),
        "explained_variance": float(pca.explained_variance_ratio_[0]),
        "model": MODEL_NAME,
        "notes": "Top-1 component removed (SIF de-bias)"
    }
    (STEP3_DIR / "_pca_meta.json").write_text(json.dumps(pca_meta, indent=2))

    with open(STEP3_DIR / "_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest: f.write(json.dumps(r) + "\n")

    print(f"[done] Step 3.5 complete in {time.time()-t0:.1f}s → {STEP3_DIR}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
