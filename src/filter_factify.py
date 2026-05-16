"""
FACTIFY Dataset Filter — Extract Telugu-only rows
==================================================
Usage:
    python -m src.filter_factify <input_csv> <output_csv>

Example:
    python -m src.filter_factify data/factify_train.csv data/telugu_factify.csv

The FACTIFY dataset (IIT Jodhpur) contains multi-lingual claim-verification pairs.
This script extracts only rows where the 'text' or 'claim' column contains
Telugu characters (U+0C00–U+0C7F) and maps FACTIFY labels to our 3-class schema:
  - 'Fake'   / 'Refuted'  / label 0  → 1 (Fake)
  - 'True'   / 'Supported' / label 1  → 0 (Real)
  - 'Other'  / 'NEI'                  → 2 (Unverifiable)
"""

import pandas as pd
import argparse
import sys
from .preprocess import is_telugu


FACTIFY_LABEL_MAP = {
    # FACTIFY uses string labels
    "fake":          1,
    "refuted":       1,
    "false":         1,
    "true":          0,
    "real":          0,
    "supported":     0,
    "verified":      0,
    "unverifiable":  2,
    "nei":           2,
    "not enough info": 2,
    "other":         2,
}




def map_label(raw_label) -> int:
    """
    Map a raw FACTIFY label string/int to our 3-class int schema.
    Returns 2 (Unverifiable) for any unrecognised value.
    """
    if isinstance(raw_label, int):
        # Already numeric — keep if in range, else fallback
        return raw_label if raw_label in (0, 1, 2) else 2
    return FACTIFY_LABEL_MAP.get(str(raw_label).strip().lower(), 2)


def filter_csv(input_csv: str, output_csv: str) -> None:
    """Filter FACTIFY CSV to Telugu rows only and remap labels."""
    print(f"📂 Loading {input_csv} …")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}", file=sys.stderr)
        return

    original_count = len(df)
    print(f"   Original rows : {original_count:,}")

    # Detect text column
    if "text" in df.columns:
        text_col = "text"
    elif "claim" in df.columns:
        text_col = "claim"
    else:
        print("❌ CSV must contain a 'text' or 'claim' column.", file=sys.stderr)
        return

    # Filter to Telugu rows
    telugu_mask = df[text_col].apply(is_telugu)
    telugu_df = df[telugu_mask].copy()
    print(f"   Telugu rows   : {len(telugu_df):,}")
    print(f"   Removed       : {original_count - len(telugu_df):,} non-Telugu rows")

    # Normalise label column
    label_col = next(
        (c for c in ("label", "Label", "verdict", "Verdict") if c in telugu_df.columns),
        None,
    )
    if label_col:
        telugu_df["label"] = telugu_df[label_col].apply(map_label)
        if label_col != "label":
            telugu_df = telugu_df.drop(columns=[label_col])
        print(f"\n📊 Class distribution after mapping:")
        for label_id, label_name in {0: "Real", 1: "Fake", 2: "Unverifiable"}.items():
            n = (telugu_df["label"] == label_id).sum()
            print(f"   {label_name:15s}: {n:,}")
    else:
        print("⚠️  No label column found — saving without label remapping.")

    # Keep only essential columns
    keep_cols = [text_col, "label"] if "label" in telugu_df.columns else [text_col]
    if text_col != "text":
        telugu_df = telugu_df.rename(columns={text_col: "text"})
        keep_cols = ["text", "label"] if "label" in telugu_df.columns else ["text"]

    telugu_df[keep_cols].to_csv(output_csv, index=False)
    print(f"\n✅ Saved {len(telugu_df):,} Telugu rows → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter FACTIFY dataset: keep only Telugu rows and remap labels."
    )
    parser.add_argument("input",  help="Path to input FACTIFY CSV (e.g., factify_train.csv)")
    parser.add_argument("output", help="Path to save filtered output (e.g., telugu_factify.csv)")
    args = parser.parse_args()
    filter_csv(args.input, args.output)
