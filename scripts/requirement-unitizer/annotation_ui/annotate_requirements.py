#!/usr/bin/env python3
import json
import random
from pathlib import Path
from datetime import datetime

import streamlit as st
import spacy

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]

STEP2_DIR = ROOT / "classifier" / "step_2"
RAW_DIR = ROOT / "02_raw_requirements"
OUTPUT_DIR = ROOT / "classifier" / "manual_annotations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_OUT = OUTPUT_DIR / "all_annotations.jsonl"

NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
WINDOW = 5

LABEL_BUTTONS = [
    "requirement",
    "non_requirement",
    "functional",
    "non_functional",
    "with_context",
    "ambiguous",
    "non_ambiguous",
    "review_later",
]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def split_raw_sentences(raw_text: str):
    doc = NLP(raw_text)
    return [
        {
            "order": i,
            "text": s.text.strip(),
            "char_start": s.start_char,
            "char_end": s.end_char,
        }
        for i, s in enumerate(doc.sents, start=1)
    ]

def get_raw_context(item):
    raw_path = RAW_DIR / item["domain"] / item["prompt_id"] / f"{item['req_id']}_raw.txt"
    if not raw_path.exists():
        return [], None

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    raw_sents = split_raw_sentences(raw_text)

    idx = next((i for i, s in enumerate(raw_sents) if s["order"] == item["order"]), None)
    if idx is None:
        return [], None

    start = max(0, idx - WINDOW)
    end = min(len(raw_sents), idx + WINDOW + 1)
    return raw_sents[start:end], raw_sents[idx]

def save_annotation(record):
    record["review_timestamp"] = datetime.utcnow().isoformat() + "Z"
    with MASTER_OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
@st.cache_data
def load_all_items():
    items = []
    for domain_dir in STEP2_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name

        for prompt_dir in domain_dir.iterdir():
            if not prompt_dir.is_dir():
                continue
            prompt_id = prompt_dir.name

            for req_dir in prompt_dir.iterdir():
                if not req_dir.is_dir():
                    continue
                req_id = req_dir.name

                for fname in ("candidates.jsonl", "non_candidates.jsonl"):
                    path = req_dir / fname
                    if not path.exists():
                        continue

                    rows = load_jsonl(path)
                    for r in rows:
                        r.update(
                            {
                                "domain": domain,
                                "prompt_id": prompt_id,
                                "req_id": req_id,
                                "pipeline_label": (
                                    "candidate"
                                    if fname == "candidates.jsonl"
                                    else "non_candidate"
                                ),
                            }
                        )
                        items.append(r)

    random.shuffle(items)
    return items

# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("🧪 Manual Requirement Annotation")

items = load_all_items()

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = set()

if st.session_state.idx >= len(items):
    st.success("All items reviewed.")
    st.stop()

item = items[st.session_state.idx]
item_key = f"{item['req_id']}::{item['sent_id']}"

if st.session_state.get("current_item") != item_key:
    st.session_state.selected_labels = set()
    st.session_state.current_item = item_key

context, target = get_raw_context(item)

# ---------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------
st.markdown(f"### `{item['req_id']}` | `{item['sent_id']}`")
st.markdown(f"Pipeline label: **{item['pipeline_label']}**")
st.markdown("---")

if not context:
    st.warning("⚠️ Could not resolve raw context.")
else:
    for s in context:
        if target and s["order"] == target["order"]:
            st.markdown(
                f"""
                <div style="padding:12px;
                            background:#1e293b;
                            color:white;
                            border-radius:8px;
                            font-weight:600">
                👉 {s['text']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='opacity:0.55'>{s['text']}</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ---------------------------------------------------------------------
# LABEL TOGGLE BUTTONS
# ---------------------------------------------------------------------
st.markdown("### 🏷️ Labels (click to toggle)")

def toggle_label(label):
    if label in st.session_state.selected_labels:
        st.session_state.selected_labels.remove(label)
    else:
        st.session_state.selected_labels.add(label)

rows = [
    ["requirement", "non_requirement", "review_later"],
    ["functional", "non_functional"],
    ["with_context"],
    ["ambiguous", "non_ambiguous"],
]

for row in rows:
    cols = st.columns(len(row))
    for col, label in zip(cols, row):
        active = label in st.session_state.selected_labels
        text = f"✅ {label.replace('_', ' ')}" if active else label.replace("_", " ")
        col.button(
            text,
            key=f"{item_key}_{label}",
            on_click=toggle_label,
            args=(label,),
        )

# ---------------------------------------------------------------------
# COMMENT BOX
# ---------------------------------------------------------------------
comment = st.text_area(
    "📝 Comment (optional)",
    placeholder="Why ambiguous? Why context-heavy? Anything future-you should know?",
)

st.markdown("---")

# ---------------------------------------------------------------------
# ACTIONS
# ---------------------------------------------------------------------
c1, c2 = st.columns(2)

def save_and_next():
    if not st.session_state.selected_labels:
        st.warning("Select at least one label before saving.")
        return

    save_annotation(
        {
            **item,
            "human_labels": sorted(st.session_state.selected_labels),
            "comment": comment.strip() or None,
        }
    )
    st.session_state.idx += 1
    st.session_state.selected_labels = set()
    st.rerun()

with c1:
    st.button("💾 Save Annotation", on_click=save_and_next)

with c2:
    if st.button("⏭ Skip"):
        st.session_state.idx += 1
        st.session_state.selected_labels = set()
        st.rerun()
