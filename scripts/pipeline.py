#!/usr/bin/env python3
"""
pipeline.py
===========
End-to-end Morpheus diversity + fold-switch prediction pipeline.

Runs:
  fragment_picker.py  ->  diversity_profiler.py  ->  predict_foldswitch.py

in a single command, accepting either a FASTA or Excel file as input.

Output features (combined_diversity_summary.csv)
-------------------------------------------------
  Max_Entropy_3di   | Mean_Entropy_3di
  Max_diversity     | Mean_diversity
  Max_ss_entropy    | Mean_ss_entropy

Per-protein plots (publication-quality, Nature style)
-----------------------------------------------------
  Figure 1 — Combined entropy  : 3Di + SS entropy, min-max normalised.
  Figure 2 — SS diversity      : stacked H / E / L area + diversity score.
  Figure 3 — 3Di heatmap       : per-position 20-state probability matrix.

Fold-switch prediction (predictions/all_predictions.xlsx, positive_predictions.xlsx)
--------------------------------------------------------------------------------------
  Prediction (0/1) | Probability (confidence) |
  Entropy_3di_roll20_hotspots | SS_Entropy_roll20_hotspots  (raw per-metric hotspots) |
  Fold_Switching_Regions_Predicted  (consensus region: overlap of the two hotspot
                                      sets, merged; falls back to whichever one
                                      metric detected if only one fired) |
  FoldSwitch_Residues  (residue count spanned by the consensus region)

Usage
-----
  python pipeline.py --input seqs.fasta --db kmer_index.sqlite \\
        --model /path/to/svm_pipeline_model.pkl
  python pipeline.py --input seqs.xlsx  --db kmer_index.sqlite \\
        --k_position 3 --plddt_cutoff 70 --rolling_window 20 \\
        --model /path/to/svm_pipeline_model.pkl \\
        --out results/ --no_plots
  python pipeline.py --input seqs.fasta --db kmer_index.sqlite \\
        --model /path/to/svm_pipeline_model.pkl --workers 8
  python pipeline.py --input seqs.fasta --db kmer_index.sqlite --skip_prediction
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_script(name: str) -> Path:
    """
    Locate a sibling script (fragment_picker.py / diversity_profiler.py /
    predict_foldswitch.py). Looks next to pipeline.py first, then falls
    back to PATH, so the pipeline still works if the scripts are installed
    somewhere else on the system.
    """
    here = Path(__file__).resolve().parent / name
    if here.exists():
        return here
    return Path(name)


def run(cmd: list, step: str) -> None:
    """Run a subprocess step; print a clear banner, and abort the whole pipeline on failure."""
    print(f"\n{'='*60}")
    print(f"  STEP : {step}")
    print(f"  CMD  : {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        sys.exit(
            f"\n[PIPELINE ERROR] '{step}' failed "
            f"(exit code {result.returncode}).\nAborting."
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Morpheus end-to-end pipeline: "
            "fragment_picker -> diversity_profiler -> predict_foldswitch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / DB ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--input", required=True,
        help=(
            "Input FASTA (.fa / .fasta / .fna / .faa) "
            "or Excel (.xlsx / .xls) file"
        ),
    )
    parser.add_argument(
        "--db",
        default=(
            "/home/user/Documents/Morpheus-3D_code/database/kmer_indexed_db.sqlite"
        ),
        help="Path to the Morpheus SQLite k-mer index",
    )

    # ── diversity_profiler knobs ─────────────────────────────────────────────
    parser.add_argument(
        "--k_position", type=int, default=3,
        help="1-based intra-k-mer position to analyse (1 … k)",
    )
    parser.add_argument(
        "--plddt_cutoff", type=float, default=70.0,
        help="Minimum pLDDT at k_position to keep a hit",
    )
    parser.add_argument(
        "--rolling_window", type=int, default=20,
        help="Smoothing window width (k-mer positions)",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip per-protein diversity plots",
    )

    # ── Parallelism ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel worker processes to use for BOTH "
             "fragment_picker.py and diversity_profiler.py "
             "(0 = use all available CPU cores)",
    )

    # ── predict_foldswitch knobs ──────────────────────────────────────────────
    parser.add_argument(
        "--model",
        default=(
            "/home/user/Documents/Main/Morpheus_3di_Paper/"
            "training/model/svm_pipeline_model.pkl"
        ),
        help="Path to trained SVM pipeline .pkl used for fold-switch "
             "classification",
    )
    parser.add_argument(
        "--skip_prediction", action="store_true",
        help="Skip the fold-switch prediction step (classification stops "
             "after diversity_profiler.py)",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--out", default="/home/user/Documents/Morpheus-3D_code/resuol/test_pdbs",
        help="Root output directory (sub-folders created automatically)",
    )

    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    input_path      = Path(args.input).resolve()
    db_path         = Path(args.db).resolve()
    out_root        = Path(args.out).resolve()
    hits_dir        = out_root / "fragment_hits"
    profiler_out    = out_root / "diversity_results"
    predict_out     = out_root / "predictions"

    summary_csv     = profiler_out / "combined_diversity_summary.csv"
    detail_dir      = profiler_out / "detail"

    picker_script   = find_script("fragment_picker.py")
    profiler_script = find_script("diversity_profiler.py")
    predict_script  = find_script("predict_foldswitch.py")

    model_path = Path(args.model).resolve() if not args.skip_prediction else None

    # Resolve "0 = all cores" to an actual integer ONCE, here, and pass that
    # same resolved value to every sub-script. fragment_picker.py already
    # understands 0 as "use all cores", but diversity_profiler.py does not
    # apply that convention to its own --workers flag (it just clamps to
    # max(1, workers)), so forwarding a raw 0 to it silently pinned it to a
    # single worker regardless of how many cores were actually available.
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    # Fail fast with a readable summary rather than letting a missing file
    # blow up halfway through a multi-step subprocess chain.
    errors = []
    if not input_path.exists():
        errors.append(f"Input file not found        : {input_path}")
    if not db_path.exists():
        errors.append(f"SQLite database not found   : {db_path}")
    if not picker_script.exists():
        errors.append(f"fragment_picker.py not found    : {picker_script}")
    if not profiler_script.exists():
        errors.append(f"diversity_profiler.py not found : {profiler_script}")
    if not args.skip_prediction:
        if not predict_script.exists():
            errors.append(f"predict_foldswitch.py not found : {predict_script}")
        if model_path is not None and not model_path.exists():
            errors.append(f"SVM model .pkl not found    : {model_path}")
    if errors:
        sys.exit(
            "\n[PIPELINE ERROR] Pre-flight checks failed:\n  "
            + "\n  ".join(errors)
        )

    out_root.mkdir(parents=True, exist_ok=True)

    # ── Banner ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Morpheus Diversity + Fold-Switch Prediction Pipeline")
    print("=" * 60)
    print(f"  Input          : {input_path}")
    print(f"  Database       : {db_path}")
    print(f"  K position     : {args.k_position}")
    print(f"  pLDDT cutoff   : {args.plddt_cutoff}")
    print(f"  Rolling window : {args.rolling_window}")
    print(f"  Workers        : {n_workers}"
          f"{'  (all cores)' if args.workers == 0 else ''}")
    print(f"  Plots          : {'no' if args.no_plots else 'yes'}")
    print(f"  Prediction     : {'skipped' if args.skip_prediction else 'yes'}")
    if not args.skip_prediction:
        print(f"  SVM model      : {model_path}")
    print(f"  Output root    : {out_root}")
    print("=" * 60)

    # ── Step 1: fragment_picker.py ───────────────────────────────────────────
    picker_cmd = [
        sys.executable, str(picker_script),
        "--input",   str(input_path),
        "--db",      str(db_path),
        "--out",     str(hits_dir),
        "--workers", str(n_workers),
    ]
    run(picker_cmd, "fragment_picker.py  (k-mer database query)")

    # ── Step 2: diversity_profiler.py ────────────────────────────────────────
    profiler_cmd = [
        sys.executable, str(profiler_script),
        "--hits_dir",       str(hits_dir),
        "--out",            str(profiler_out),
        "--k_position",     str(args.k_position),
        "--plddt_cutoff",   str(args.plddt_cutoff),
        "--rolling_window", str(args.rolling_window),
        "--workers",        str(n_workers),
    ]
    if args.no_plots:
        profiler_cmd.append("--no_plots")

    run(profiler_cmd, "diversity_profiler.py  (diversity feature computation)")

    # ── Step 3: predict_foldswitch.py ────────────────────────────────────────
    if not args.skip_prediction:
        if not summary_csv.exists():
            sys.exit(
                f"\n[PIPELINE ERROR] Expected summary CSV not found after "
                f"diversity_profiler.py: {summary_csv}\nAborting."
            )

        predict_cmd = [
            sys.executable, str(predict_script),
            "--summary",    str(summary_csv),
            "--detail_dir", str(detail_dir),
            "--model",      str(model_path),
            "--out",        str(predict_out),
        ]
        run(predict_cmd, "predict_foldswitch.py  (SVM classification + hotspot regions)")

    # ── Done ─────────────────────────────────────────────────────────────────
    plots_note = " (skipped)" if args.no_plots else ""
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    print(f"\n  Fragment hits    → {hits_dir}/")
    print(f"  Diversity plots  → {profiler_out}/plots/{plots_note}")
    print(f"  Detail CSVs      → {profiler_out}/detail/")
    print(f"  Summary CSV      → {summary_csv}")
    if not args.skip_prediction:
        print(f"  All predictions  → {predict_out}/all_predictions.xlsx")
        print(f"  Positive hits    → {predict_out}/positive_predictions.xlsx")
    print()


if __name__ == "__main__":
    main()
