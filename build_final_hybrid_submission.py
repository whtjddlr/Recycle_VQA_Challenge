#!/usr/bin/env python3
from __future__ import annotations
"""Build the final submission by selectively replacing baseline answers.

The baseline submission stays untouched by default. Only rows that belong to
the allowed question types and satisfy the low-confidence / low-margin rule
switch over to the rerun answer.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final hybrid submission from a baseline submission, rerun predictions, and answer-free signals."
    )
    parser.add_argument("--baseline-csv", type=Path, required=True, help="Baseline submission CSV with columns id,answer.")
    parser.add_argument(
        "--rerun-predictions-csv",
        type=Path,
        required=True,
        help="Rerun predictions CSV with at least id,answer. question_type is optional.",
    )
    parser.add_argument(
        "--signal-csv",
        type=Path,
        required=True,
        help="Signal CSV with id, support_conf_mean, support_margin_mean. qtype is optional.",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Final hybrid submission CSV.")
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=None,
        help="Optional metadata CSV. Does not include any target answers.",
    )
    parser.add_argument(
        "--qtypes",
        type=str,
        default="count",
        help="Comma-separated qtypes allowed to switch to rerun. Default: count",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.88)
    parser.add_argument("--margin-threshold", type=float, default=0.50)
    return parser.parse_args()


def normalize_qtypes(raw: str) -> List[str]:
    """Parse a comma-separated qtype list into normalized items."""
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> None:
    args = parse_args()

    baseline_df = pd.read_csv(args.baseline_csv).astype({"id": str})
    rerun_df = pd.read_csv(args.rerun_predictions_csv).astype({"id": str})
    signal_df = pd.read_csv(args.signal_csv).astype({"id": str})

    for df_name, df in [("baseline", baseline_df), ("rerun", rerun_df), ("signal", signal_df)]:
        if "id" not in df.columns:
            raise ValueError(f"{df_name} csv must contain an id column")

    # Keep baseline as the left side so every original submission row survives.
    merged = baseline_df.rename(columns={"answer": "baseline_answer"}).merge(
        rerun_df.rename(columns={"answer": "rerun_answer"}),
        on="id",
        how="left",
    )
    merged = merged.merge(signal_df, on="id", how="left")

    if "qtype" not in merged.columns:
        if "question_type" in merged.columns:
            merged["qtype"] = merged["question_type"].astype(str)
        else:
            raise ValueError("signal csv must contain qtype, or rerun predictions must contain question_type")

    required_cols = {"support_conf_mean", "support_margin_mean"}
    missing = required_cols - set(merged.columns)
    if missing:
        raise ValueError(f"signal csv missing required columns: {sorted(missing)}")

    target_qtypes = set(normalize_qtypes(args.qtypes))
    # Switching is deliberately narrow: the row must belong to one of the
    # configured qtypes, have a rerun answer, and look uncertain by signals.
    use_rerun = (
        merged["qtype"].isin(target_qtypes)
        & merged["rerun_answer"].notna()
        & (
            (merged["support_conf_mean"] < args.conf_threshold)
            | (merged["support_margin_mean"] < args.margin_threshold)
        )
    )
    merged["use_rerun"] = use_rerun
    merged["final_answer"] = np.where(use_rerun, merged["rerun_answer"], merged["baseline_answer"])

    output_df = merged[["id", "final_answer"]].rename(columns={"final_answer": "answer"})
    output_df.to_csv(args.output_csv, index=False)

    if args.meta_csv:
        # The meta CSV records which branch was selected and why, without using
        # any hidden target answer column.
        meta_cols = [
            "id",
            "qtype",
            "support_conf_mean",
            "support_margin_mean",
            "baseline_answer",
            "rerun_answer",
            "final_answer",
            "use_rerun",
        ]
        merged[meta_cols].to_csv(args.meta_csv, index=False)

    changed_rows = int((merged["final_answer"] != merged["baseline_answer"]).sum())
    print(f"saved hybrid csv : {args.output_csv}")
    print(f"changed rows     : {changed_rows}")
    if args.meta_csv:
        print(f"saved meta csv   : {args.meta_csv}")


if __name__ == "__main__":
    main()
