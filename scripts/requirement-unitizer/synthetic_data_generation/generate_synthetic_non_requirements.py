#!/usr/bin/env python3
import json
import pathlib
import sys
import time
import logging
from datetime import datetime
import re

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from services.llm_clients import call_deepseek_validator 

PROMPT_PATH = ROOT_DIR / "prompts/requirement-unitization/non_requirements_prompt.md"
OUTPUT_DIR  = ROOT_DIR / "classifier/synthetic/non_requirements"

MODEL_NAME = "deepseek-r1:7b"
TOTAL_SAMPLES = 5000
CHUNK_SIZE = 10
MAX_RETRIES = 3


CATEGORIES = [
    # semantic non-requirements
    "BACKGROUND_CONTEXT",
    "CAPABILITY_DESCRIPTION",
    "ARCHITECTURE_EXPLANATION",
    "RATIONALE_JUSTIFICATION",
    "FEASIBILITY_ANALYSIS",
    "ROADMAP_PLANNING",
    "STAKEHOLDER_DESCRIPTION",
    "MARKETING_LANGUAGE",
    "AMBIGUOUS_HYBRID",

    # noise / artifacts
    "RANDOM_SINGLE_WORDS",
    "BROKEN_PHRASES",
    "MARKDOWN_ARTIFACTS",
    "BULLET_FRAGMENTS",
    "HEADINGS_TITLES",
    "LINKS_REFERENCES",
    "CODE_SNIPPET_NOISE",
    "TABLE_ROW_FRAGMENTS",
    "SEPARATORS_DIVIDERS",
]


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8")

def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()
    return text

def build_prompt(template: str, category: str, count: int) -> str:
    return (
        template
        .replace("{{ category }}", category)
        .replace("{{ count }}", str(count))
    )

def main():
    template = load_prompt()
    per_category = TOTAL_SAMPLES // len(CATEGORIES)

    output_path = OUTPUT_DIR / "synthetic_non_requirements.jsonl"
    total_written = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for category in CATEGORIES:
            logging.info(f"Generating {category}")
            generated = 0

            while generated < per_category:
                batch_count = min(CHUNK_SIZE, per_category - generated)
                prompt = build_prompt(template, category, batch_count)

                response = call_deepseek_validator(prompt).get("response", "")
                if not response:
                    logging.warning("Empty response, retrying")
                    continue

                try:
                    cleaned = strip_think_blocks(response)
                    parsed = json.loads(cleaned)

                    if isinstance(parsed, dict) and "items" in parsed:
                        items = parsed["items"]
                    elif isinstance(parsed, list):
                        items = parsed
                    else:
                        raise ValueError("Unexpected JSON shape from LLM")
                except Exception as e:
                    logging.warning(f"Parse error: {e}")
                    continue

                for text in items:
                    record = {
                        "id": f"synthetic-nr-{total_written:06d}",
                        "text": text,
                        "label": "non_requirement",
                        "category": category,
                        "source": "synthetic_llm",
                        "model": MODEL_NAME,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_written += 1
                    generated += 1

                time.sleep(0.3)

    logging.info(f"DONE generated {total_written} synthetic non-requirements")

if __name__ == "__main__":
    main()
