#!/usr/bin/env python3
"""
Select 40% of main (req_pipeline) requirements by req_id
---------------------------------------------------
- Samples req_ids, not sentences
- Keeps all other datasets intact
"""

import pandas as pd
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV  = ROOT / "classifier" / "llm_reviews" / "requirements_all.csv"
OUTPUT_CSV = ROOT / "classifier" / "classifier_dataset_unitization" / "requirements_all_filtered.csv"
STATS_JSON = ROOT / "classifier" / "classifier_dataset_unitization" / "subset_stats.json"

MAIN_SOURCE = "req_pipeline_real"
SELECTION_RATIO = 0.40
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_CSV)

    assert "req_id" in df.columns, "req_id column missing"
    assert "source" in df.columns, "source column missing"

    # Split main vs others
    df_main   = df[df["source"] == MAIN_SOURCE]
    df_other  = df[df["source"] != MAIN_SOURCE]

    # Unique req_ids in main dataset
    req_ids = df_main["req_id"].dropna().unique().tolist()
    total_req_ids = len(req_ids)

    selected_count = int(total_req_ids * SELECTION_RATIO)
    selected_req_ids = set(random.sample(req_ids, selected_count))

    # Filter main dataset
    df_main_selected = df_main[df_main["req_id"].isin(selected_req_ids)]

    # Combine back
    df_final = pd.concat([df_main_selected, df_other], ignore_index=True)

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------
    stats = {
        "main_dataset": {
            "total_req_ids": total_req_ids,
            "selected_req_ids": selected_count,
            "selection_ratio": SELECTION_RATIO,
            "total_rows_before": len(df_main),
            "total_rows_after": len(df_main_selected)
        },
        "final_dataset": {
            "total_rows": len(df_final),
            "label_distribution": df_final["label"].value_counts().to_dict(),
            "source_distribution": df_final["source"].value_counts().to_dict()
        }
    }

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("[done] Selected 40% of main dataset by req_id")
    print(json.dumps(stats, indent=2))

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
