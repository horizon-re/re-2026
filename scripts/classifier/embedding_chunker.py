#!/usr/bin/env python3
"""
Step 2 — Embedding Chunker
---------------------------------------------------------
Generates sentence-level and document-level embeddings
for each requirement under classifier/step_1/.

Output directory:
    classifier/step_2/<domain>/<prompt>/<req_id>/

Files generated per requirement:
    - sentences.jsonl             → per-sentence records (id, text, weights)
    - sentence_embeddings.npy     → N × 768 float32 array
    - doc_embedding.npy           → 1 × 768 float32 array
    - step_2_manifest.jsonl       → run summary (all requirements)
"""

import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np
import spacy, stanza
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import Counter
from typing import List, Dict, Any

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
STEP1_DIR = ROOT / "classifier" / "step_1"
STEP2_DIR = ROOT / "classifier" / "step_2"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SIF_A = 1e-3
LANG = "en"
MAX_SENT_WARNING = 200
os.makedirs(STEP2_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_read(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return path.read_bytes().decode(errors="ignore")

def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / norm

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# ---------------------------------------------------------------------
# SENTENCE SPLITTING (spaCy + Stanza cross-check)
# ---------------------------------------------------------------------
def split_sentences_spacy_stanza(text: str, nlp_spacy, nlp_stanza) -> List[str]:
    spacy_sents = [s.text.strip() for s in nlp_spacy(text).sents if s.text.strip()]
    stanza_doc = nlp_stanza(text)
    stanza_sents = [s.text.strip() for s in stanza_doc.sentences if s.text.strip()]

    # Agreement check
    agree = True
    if abs(len(spacy_sents) - len(stanza_sents)) / max(1, len(spacy_sents)) > 0.1:
        agree = False
    return spacy_sents, stanza_sents, agree

# ---------------------------------------------------------------------
# SIF WEIGHT COMPUTATION
# ---------------------------------------------------------------------
def build_word_freqs(texts: List[str], nlp_spacy) -> Counter:
    """Rough word frequency across corpus"""
    freq = Counter()
    for t in texts:
        doc = nlp_spacy(t)
        for tok in doc:
            if tok.is_alpha and not tok.is_stop:
                freq[tok.lemma_.lower()] += 1
    return freq

def compute_sif_weights(sentences: List[str], word_freq: Counter, a=SIF_A) -> np.ndarray:
    weights = []
    for sent in sentences:
        doc_tokens = [t.lower() for t in sent.split()]
        freqs = [word_freq.get(tok, 1) for tok in doc_tokens]
        if freqs:
            invs = [a / (a + f) for f in freqs]
            weights.append(np.mean(invs))
        else:
            weights.append(1.0)
    return np.array(weights, dtype=np.float32)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"[init] Loading models...")
    model = SentenceTransformer(MODEL_NAME)
    nlp_spacy = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp_stanza = stanza.Pipeline(LANG, processors="tokenize", verbose=False)

    manifest_records = []
    all_sentence_vecs = []

    domains = [p for p in STEP1_DIR.iterdir() if p.is_dir()]
    for domain_path in domains:
        for prompt_path in domain_path.iterdir():
            if not prompt_path.is_dir():
                continue
            for req_file in prompt_path.glob("*_raw.txt"):
                req_id = req_file.stem.replace("_raw", "")
                out_dir = STEP2_DIR / domain_path.name / prompt_path.name / req_id
                out_dir.mkdir(parents=True, exist_ok=True)

                raw_text = safe_read(req_file)
                spacy_sents, stanza_sents, agree = split_sentences_spacy_stanza(raw_text, nlp_spacy, nlp_stanza)
                sents = spacy_sents

                if not sents:
                    print(f"[warn] No sentences → {req_id}")
                    continue

                # Build word freqs locally (you can prebuild global one)
                freq = build_word_freqs(sents, nlp_spacy)
                sif_weights = compute_sif_weights(sents, freq, a=SIF_A)

                # Encode all sentences
                embeddings = model.encode(sents, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
                weights_norm = sif_weights / (sif_weights.sum() + 1e-8)

                # Weighted pooling
                doc_emb = np.sum(embeddings * weights_norm[:, None], axis=0)

                # Collect for PCA later
                all_sentence_vecs.append(doc_emb)

                # Write per-sentence data
                sent_records = []
                for i, (txt, w) in enumerate(zip(sents, weights_norm)):
                    sent_id = f"{req_id}::s{str(i+1).zfill(3)}"
                    sent_records.append({
                        "req_id": req_id,
                        "sent_id": sent_id,
                        "index": i+1,
                        "text": txt,
                        "token_count": len(txt.split()),
                        "sif_weight": float(w),
                        "splitter_agree": agree
                    })

                with open(out_dir / "sentences.jsonl", "w", encoding="utf-8") as f:
                    for r in sent_records:
                        f.write(json.dumps(r) + "\n")

                np.save(out_dir / "sentence_embeddings.npy", embeddings.astype(np.float32))
                np.save(out_dir / "doc_embedding_raw.npy", doc_emb.astype(np.float32))

    # PCA DE-BIAS
    print(f"[pca] Computing top-1 component removal over {len(all_sentence_vecs)} documents...")
    X = np.stack(all_sentence_vecs)
    pca = PCA(n_components=1)
    pca.fit(X)
    u = pca.components_[0]
    X_denoised = X - X @ u[:, None] * u[None, :]
    X_norm = l2_normalize(X_denoised)

    # Save PCA meta
    pca_meta = {
        "component": u.tolist(),
        "explained_variance": float(pca.explained_variance_ratio_[0]),
        "model": MODEL_NAME
    }
    (STEP2_DIR / "_pca_meta.json").write_text(json.dumps(pca_meta, indent=2))

    # Rewrite final doc embeddings
    print("[save] Writing final PCA-cleaned doc embeddings...")
    idx = 0
    for domain_path in STEP1_DIR.iterdir():
        for prompt_path in domain_path.iterdir():
            for req_file in prompt_path.glob("*_raw.txt"):
                req_id = req_file.stem.replace("_raw", "")
                out_dir = STEP2_DIR / domain_path.name / prompt_path.name / req_id
                if not out_dir.exists():
                    continue
                doc_emb = X_norm[idx]
                np.save(out_dir / "doc_embedding.npy", doc_emb.astype(np.float32))
                idx += 1

                manifest_records.append({
                    "req_id": req_id,
                    "domain": domain_path.name,
                    "prompt_id": prompt_path.name,
                    "embedding_dim": int(doc_emb.shape[0]),
                    "num_sentences": int(len(np.load(out_dir / "sentence_embeddings.npy"))),
                    "pca_removed": True,
                    "checksum": hash_text(req_id + domain_path.name),
                    "path": str(out_dir.relative_to(ROOT))
                })

    with open(STEP2_DIR / "_manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    print(f"[done] Step 2 complete in {time.time()-t0:.1f}s → {STEP2_DIR}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
