#!/usr/bin/env python3
from __future__ import annotations
"""Create a rerun subset without using any target answers.

This script joins test rows with answer-free support signals, then keeps only
the question types and low-confidence / low-margin samples that should be
reprocessed by the heavier multistage pipeline.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from multistage_vqa import build_choices, classify_question_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a rerun subset CSV from test rows and a signal CSV without using any target answers."
    )
    parser.add_argument("--test-csv", type=Path, required=True, help="Path to test.csv.")
    parser.add_argument(
        "--signal-csv",
        type=Path,
        required=True,
        help="CSV with at least id, support_conf_mean, support_margin_mean. qtype is optional.",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Subset CSV to write.")
    parser.add_argument("--output-ids", type=Path, default=None, help="Optional text file with one id per line.")
    parser.add_argument(
        "--qtypes",
        type=str,
        default="count",
        help="Comma-separated qtypes to include. Default: count",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.88, help="Use rerun when support_conf_mean is below this.")
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.50,
        help="Use rerun when support_margin_mean is below this.",
    )
    return parser.parse_args()


def infer_qtype(row: pd.Series) -> str:
    """Infer question type directly from the original question and choices."""
    choice_map, _ = build_choices(row, shuffle=False)
    return classify_question_type(str(row.get("question", "")), choice_map)


def normalize_qtypes(raw: str) -> List[str]:
    """Parse a comma-separated qtype list into normalized items."""
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> None:
    args = parse_args()

    test_df = pd.read_csv(args.test_csv)
    signal_df = pd.read_csv(args.signal_csv)

    required_cols = {"id", "support_conf_mean", "support_margin_mean"}
    missing = required_cols - set(signal_df.columns)
    if missing:
        raise ValueError(f"signal csv missing required columns: {sorted(missing)}")

    # Only rows that exist in both files are eligible for rerun selection.
    merged = test_df.merge(signal_df, on="id", how="inner")
    if "qtype" not in merged.columns:
        merged["qtype"] = merged.apply(infer_qtype, axis=1)

    target_qtypes = set(normalize_qtypes(args.qtypes))
    # A row is selected when its qtype is allowed and at least one stability
    # signal falls below the configured threshold.
    selected = merged[
        merged["qtype"].isin(target_qtypes)
        & (
            (merged["support_conf_mean"] < args.conf_threshold)
            | (merged["support_margin_mean"] < args.margin_threshold)
        )
    ].copy()

    # Write back the original test rows so downstream rerun code sees the same
    # schema as the base test.csv.
    subset_df = test_df[test_df["id"].isin(selected["id"])].copy()
    subset_df.to_csv(args.output_csv, index=False)

    if args.output_ids:
        args.output_ids.write_text(
            "\n".join(subset_df["id"].astype(str).tolist()) + "\n",
            encoding="utf-8",
        )

    print(f"saved subset csv : {args.output_csv}")
    if args.output_ids:
        print(f"saved subset ids : {args.output_ids}")
    print(f"selected rows     : {len(subset_df)}")
    if not selected.empty:
        print(selected["qtype"].value_counts().to_string())


if __name__ == "__main__":
    main()
