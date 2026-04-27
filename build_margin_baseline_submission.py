#!/usr/bin/env python3
from __future__ import annotations
"""Build a conservative baseline submission from two prediction CSV files.

The intended flow is:
1. Run two independent prediction branches.
2. Compare answer, confidence, and margin_top2 per sample.
3. Switch away from the preferred branch only when the alternate branch is
   clearly stronger by the configured thresholds.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


# Only four multiple-choice labels are considered valid outputs.
CHOICE_LABELS = {"a", "b", "c", "d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a baseline submission by conservatively merging two prediction CSVs using margin/confidence."
    )
    parser.add_argument("--primary-csv", type=Path, required=True, help="Default prediction CSV.")
    parser.add_argument("--secondary-csv", type=Path, required=True, help="Alternative prediction CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Merged baseline submission CSV.")
    parser.add_argument("--meta-csv", type=Path, default=None, help="Optional metadata CSV without any target answers.")
    parser.add_argument(
        "--prefer-source",
        choices=["primary", "secondary"],
        default="primary",
        help="Fallback source when signals are tied or unavailable.",
    )
    parser.add_argument(
        "--min-margin-top2",
        type=float,
        default=0.0,
        help="Secondary margin must be at least this value to override the preferred source.",
    )
    parser.add_argument(
        "--min-margin-gap",
        type=float,
        default=0.0,
        help="Secondary margin - preferred margin must be at least this value to override.",
    )
    parser.add_argument(
        "--min-confidence-gap",
        type=float,
        default=0.0,
        help="Used as a tiebreak when margins are tied or missing.",
    )
    return parser.parse_args()


def normalize_answer(value: object) -> str:
    """Normalize any answer-like value into a lower-case choice label."""
    answer = str(value or "").strip().lower()
    return answer if answer in CHOICE_LABELS else ""


def load_prediction_csv(path: Path, prefix: str) -> pd.DataFrame:
    """Load one branch prediction file and namespace its columns."""
    df = pd.read_csv(path).astype({"id": str}).copy()
    if "id" not in df.columns or "answer" not in df.columns:
        raise ValueError(f"{path} must contain id and answer columns")

    renamed = df.rename(columns={"answer": f"{prefix}_answer"})
    for signal in ["confidence", "margin_top2"]:
        if signal in renamed.columns:
            renamed = renamed.rename(columns={signal: f"{prefix}_{signal}"})
    return renamed


def safe_float(value: object) -> float:
    """Convert optional numeric signals without failing on missing values."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def choose_answer(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    """Pick the final answer for one row using margin-first conservative rules."""
    prefer_primary = args.prefer_source == "primary"

    primary_answer = normalize_answer(row.get("primary_answer"))
    secondary_answer = normalize_answer(row.get("secondary_answer"))
    primary_margin = safe_float(row.get("primary_margin_top2"))
    secondary_margin = safe_float(row.get("secondary_margin_top2"))
    primary_conf = safe_float(row.get("primary_confidence"))
    secondary_conf = safe_float(row.get("secondary_confidence"))

    default_answer = primary_answer if prefer_primary else secondary_answer
    default_source = "primary" if prefer_primary else "secondary"
    alt_answer = secondary_answer if prefer_primary else primary_answer
    alt_source = "secondary" if prefer_primary else "primary"
    default_margin = primary_margin if prefer_primary else secondary_margin
    alt_margin = secondary_margin if prefer_primary else primary_margin
    default_conf = primary_conf if prefer_primary else secondary_conf
    alt_conf = secondary_conf if prefer_primary else primary_conf

    if not default_answer and alt_answer:
        return alt_answer, alt_source
    if default_answer and not alt_answer:
        return default_answer, default_source
    if not default_answer and not alt_answer:
        return "", "missing"

    # Margin is the primary switching signal because it is usually more stable
    # than comparing raw confidence across different branches.
    if pd.notna(alt_margin) and pd.notna(default_margin):
        if alt_margin >= args.min_margin_top2 and (alt_margin - default_margin) >= args.min_margin_gap:
            return alt_answer, alt_source
        if default_margin >= args.min_margin_top2 and (default_margin - alt_margin) >= args.min_margin_gap:
            return default_answer, default_source

    # Confidence is only used as a fallback tiebreak when margin is missing or close.
    if pd.notna(alt_conf) and pd.notna(default_conf):
        if (alt_conf - default_conf) >= args.min_confidence_gap:
            return alt_answer, alt_source
        if (default_conf - alt_conf) >= args.min_confidence_gap:
            return default_answer, default_source

    return default_answer, default_source


def main() -> None:
    args = parse_args()

    # Merge by id so the builder can operate even when one branch is missing rows.
    primary_df = load_prediction_csv(args.primary_csv, "primary")
    secondary_df = load_prediction_csv(args.secondary_csv, "secondary")
    merged = primary_df.merge(secondary_df, on="id", how="outer")

    decisions: List[dict] = []
    for row in merged.to_dict(orient="records"):
        answer, picked_source = choose_answer(pd.Series(row), args)
        decisions.append(
            {
                "id": str(row["id"]),
                "answer": answer,
                "picked_source": picked_source,
                "primary_answer": normalize_answer(row.get("primary_answer")),
                "secondary_answer": normalize_answer(row.get("secondary_answer")),
                "primary_confidence": safe_float(row.get("primary_confidence")),
                "secondary_confidence": safe_float(row.get("secondary_confidence")),
                "primary_margin_top2": safe_float(row.get("primary_margin_top2")),
                "secondary_margin_top2": safe_float(row.get("secondary_margin_top2")),
            }
        )

    # The main output remains a plain submission file; extra debug columns only
    # go to the optional meta CSV.
    decision_df = pd.DataFrame(decisions)
    decision_df[["id", "answer"]].to_csv(args.output_csv, index=False)
    if args.meta_csv:
        decision_df.to_csv(args.meta_csv, index=False)

    changed = int(
        (
            decision_df["picked_source"]
            == ("secondary" if args.prefer_source == "primary" else "primary")
        ).sum()
    )
    print(f"saved baseline csv : {args.output_csv}")
    print(f"total rows         : {len(decision_df)}")
    print(f"switched rows      : {changed}")
    if args.meta_csv:
        print(f"saved meta csv     : {args.meta_csv}")


if __name__ == "__main__":
    main()
