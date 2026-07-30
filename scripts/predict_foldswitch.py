#!/usr/bin/env python3
"""
predict_foldswitch.py
======================
Predict fold-switching status, confidence (probability), and a single
consensus fold-switch region per protein, from the output of
diversity_profiler.py.

This combines:

  1. SVM classification on `combined_diversity_summary.csv`
     (Max/Mean Entropy_3di, diversity, ss_entropy) -> Prediction + Probability.

  2. Region-level fold-switch calling from the per-protein `*_metrics.csv`
     detail files:
       a. Rolling-average hotspot detection on Entropy_3di and ss_entropy
          separately (same fixed thresholds/window as fold_switch_annot.py).
       b. The two hotspot sets are reconciled into one consensus region set
          (overlap-and-merge logic ported from
          proteome_foldswitch_disorder_pipeline.py — only the region-calling
          part; the IUPred/MetaPred disorder-overlap analysis from that
          script is not used here):
            - both entropy & SS hotspots present -> intersect the two, then
              merge adjacent/overlapping intervals
            - only one of the two present         -> use it as-is (merged)
            - neither present                     -> no region called

Output
------
  {out}/all_predictions.xlsx      — every protein: features, Prediction,
                                     Probability, raw hotspots, and the
                                     final Fold_Switching_Regions_Predicted
  {out}/positive_predictions.xlsx — subset where Prediction == 1
                                     (fold-switch candidates)

Usage
-----
  python predict_foldswitch.py \\
      --summary  diversity_results/combined_diversity_summary.csv \\
      --detail_dir diversity_results/detail \\
      --model    /path/to/svm_pipeline_model.pkl \\
      --out      predictions/

Omit --detail_dir to skip region prediction (classification only).
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

# ── Classifier features (must match training) ────────────────────────────────
FEATURES = [
    "Max_Entropy_3di",
    "Mean_Entropy_3di",
    "Max_diversity",
    "Mean_diversity",
    "Max_ss_entropy",
    "Mean_ss_entropy",
]

# ── Hotspot-region parameters (same as fold_switch_annot.py) ─────────────────
ROLLING_WINDOW = 20
ENTROPY_THRESHOLD = 0.95
SS_ENTROPY_THRESHOLD = 0.40


# ── Rolling-average hotspot detection (ported from fold_switch_annot.py) ────

def rolling_avg(series: pd.Series, window: int) -> pd.Series:
    """Centered rolling mean, used to smooth out single-residue noise before thresholding."""
    return series.rolling(window=window, center=True).mean()


def find_hotspot_regions(values: pd.Series, threshold: float) -> list[tuple[int, int]]:
    """
    Contiguous-run detection on a rolling-averaged series above `threshold`.
    Returns a list of (start, end) 1-based position tuples.
    """
    rolled = rolling_avg(values, ROLLING_WINDOW)
    hotspot_idx = rolled[rolled > threshold].index.tolist()

    if not hotspot_idx:
        return []

    hotspot_idx = sorted(int(x) for x in hotspot_idx)

    regions = []
    start = hotspot_idx[0]
    end = hotspot_idx[0]

    for pos in hotspot_idx[1:]:
        if pos == end + 1:
            end = pos
        else:
            regions.append((start, end))
            start = pos
            end = pos

    regions.append((start, end))
    return regions


# ── Region reconciliation (ported from proteome_foldswitch_disorder_pipeline.py) ─
# NOTE: only the fold-switch region-calling logic is used here — the
# IUPred/MetaPred disorder-overlap analysis from that script is intentionally
# left out.

def find_overlaps(
    regions1: list[tuple[int, int]],
    regions2: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Pairwise-intersect every region in `regions1` against every region in `regions2`."""
    overlaps = []
    for s1, e1 in regions1:
        for s2, e2 in regions2:
            start = max(s1, s2)
            end = min(e1, e2)
            if start <= end:
                overlaps.append((start, end))
    return overlaps


def merge_regions(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or directly adjacent (start <= prev_end + 1) intervals."""
    if not regions:
        return []

    regions = sorted(regions)
    merged = [list(regions[0])]

    for start, end in regions[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [tuple(r) for r in merged]


def format_regions(regions: list[tuple[int, int]]) -> str:
    """Render a list of (start, end) tuples as a compact "(s,e),(s,e)" string for spreadsheets."""
    if not regions:
        return ""
    return ",".join(f"({s},{e})" for s, e in regions)


def regions_to_set(regions: list[tuple[int, int]]) -> set:
    """Expand (start, end) intervals into the full set of residue positions they cover."""
    residues = set()
    for start, end in regions:
        residues.update(range(start, end + 1))
    return residues


def call_foldswitch_regions(
    entropy_regions: list[tuple[int, int]],
    ss_regions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Reconcile 3Di-entropy hotspots and SS-entropy hotspots into one
    consensus fold-switch region set.
    """
    if entropy_regions and ss_regions:
        return merge_regions(find_overlaps(entropy_regions, ss_regions))
    if entropy_regions:
        return merge_regions(entropy_regions)
    if ss_regions:
        return merge_regions(ss_regions)
    return []


def get_protein_regions(protein_name: str, detail_dir: Path) -> dict:
    """
    Load {protein}_metrics.csv from detail_dir and compute:
      - raw Entropy_3di hotspots
      - raw ss_entropy hotspots
      - consensus Fold_Switching_Regions_Predicted
      - residue count of the consensus region

    Returns a dict with formatted-string / int values; all blank/zero if the
    detail file is missing or unreadable, so a single bad file never crashes
    the whole batch.
    """
    fp = detail_dir / f"{protein_name}_metrics.csv"

    empty = {
        "Entropy_3di_roll20_hotspots": "",
        "SS_Entropy_roll20_hotspots": "",
        "Fold_Switching_Regions_Predicted": "",
        "FoldSwitch_Residues": 0,
    }

    if not fp.exists():
        return empty

    try:
        metrics_df = pd.read_csv(fp, index_col="ResiduePosition")
    except Exception:
        return empty

    entropy_regions = (
        find_hotspot_regions(metrics_df["Entropy_3di"], ENTROPY_THRESHOLD)
        if "Entropy_3di" in metrics_df.columns else []
    )
    ss_regions = (
        find_hotspot_regions(metrics_df["ss_entropy"], SS_ENTROPY_THRESHOLD)
        if "ss_entropy" in metrics_df.columns else []
    )

    final_regions = call_foldswitch_regions(entropy_regions, ss_regions)

    return {
        "Entropy_3di_roll20_hotspots": format_regions(entropy_regions),
        "SS_Entropy_roll20_hotspots": format_regions(ss_regions),
        "Fold_Switching_Regions_Predicted": format_regions(final_regions),
        "FoldSwitch_Residues": len(regions_to_set(final_regions)),
    }


# ── Core prediction routine ──────────────────────────────────────────────────

def run_predictions(
    summary_csv: str,
    model_path: str,
    out_dir: str,
    detail_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the summary features, run the trained SVM, attach region calls, and save spreadsheets."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    missing_feats = [c for c in FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"Summary CSV is missing required feature column(s): {missing_feats}\n"
            f"Found columns: {list(df.columns)}"
        )

    df_out = df.copy()
    X = df[FEATURES]

    # Drop rows with any missing feature values — the classifier can't score
    # a protein with incomplete features, so we skip it rather than error out.
    mask = X.notna().all(axis=1)
    n_dropped = (~mask).sum()
    X = X[mask]
    df_out = df_out[mask].copy()

    if n_dropped:
        print(f"  Skipped {n_dropped} protein(s) with missing feature values.")

    if X.empty:
        raise ValueError("No proteins with complete feature values to predict on.")

    # ── Load model & predict ─────────────────────────────────────────────────
    model = joblib.load(model_path)
    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        probs = model.decision_function(X)
    else:
        probs = [float("nan")] * len(X)

    df_out["Prediction"] = preds
    df_out["Probability"] = probs

    # ── Fold-switch region prediction (optional) ─────────────────────────────
    if detail_dir is not None:
        detail_path = Path(detail_dir)
        if not detail_path.exists():
            print(f"  WARNING: detail_dir not found ({detail_path}); "
                  f"fold-switch regions will be left blank.")
            detail_path = None
    else:
        detail_path = None

    protein_col = "Protein" if "Protein" in df_out.columns else df_out.columns[0]

    region_cols = {
        "Entropy_3di_roll20_hotspots": [],
        "SS_Entropy_roll20_hotspots": [],
        "Fold_Switching_Regions_Predicted": [],
        "FoldSwitch_Residues": [],
    }

    for name in df_out[protein_col]:
        if detail_path is not None:
            r = get_protein_regions(str(name), detail_path)
        else:
            r = {
                "Entropy_3di_roll20_hotspots": "",
                "SS_Entropy_roll20_hotspots": "",
                "Fold_Switching_Regions_Predicted": "",
                "FoldSwitch_Residues": 0,
            }
        for col, val in r.items():
            region_cols[col].append(val)

    for col, vals in region_cols.items():
        df_out[col] = vals

    # ── Save outputs ──────────────────────────────────────────────────────────
    all_output_file = out_dir / "all_predictions.xlsx"
    positive_output_file = out_dir / "positive_predictions.xlsx"

    df_out.to_excel(all_output_file, index=False)

    positive_df = df_out[df_out["Prediction"] == 1]
    positive_df.to_excel(positive_output_file, index=False)

    print(f"\n  All predictions      -> {all_output_file}")
    print(f"  Positive predictions -> {positive_output_file}")
    print(f"  Total proteins scored : {len(df_out)}")
    print(f"  Fold-switch candidates: {len(positive_df)}")

    return df_out, positive_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict fold-switching status, confidence, and consensus "
                    "fold-switch regions from a Morpheus "
                    "combined_diversity_summary.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", required=True,
                        help="Path to combined_diversity_summary.csv "
                             "(output of diversity_profiler.py)")
    parser.add_argument("--detail_dir", default=None,
                        help="Path to diversity_results/detail/ (per-protein "
                             "*_metrics.csv files) for fold-switch region "
                             "prediction. Omit to skip region prediction.")
    parser.add_argument("--model", required=True,
                        help="Path to trained SVM pipeline .pkl")
    parser.add_argument("--out", default="./predictions",
                        help="Output directory for prediction spreadsheets")
    args = parser.parse_args()

    print("\nParameters")
    print(f"  Summary CSV  : {args.summary}")
    print(f"  Detail dir   : {args.detail_dir or '(none — regions skipped)'}")
    print(f"  Model        : {args.model}")
    print(f"  Output dir   : {args.out}\n")

    run_predictions(
        summary_csv=args.summary,
        model_path=args.model,
        out_dir=args.out,
        detail_dir=args.detail_dir,
    )


if __name__ == "__main__":
    main()
