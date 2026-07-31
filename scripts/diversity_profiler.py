#!/usr/bin/env python3
"""
diversity_profiler.py (v2)
==========================
Compute k-mer diversity features from hit CSV files and produce
a single publication-quality aligned three-panel figure.

Output
------
  combined_diversity_summary.csv — feature table
  {protein_name}_aligned_panels.pdf / .png — single aligned figure:
    Panel A: Combined 3Di + SS entropy (normalised, smoothed)
    Panel B: SS diversity stacked bar (H / E / L fractions)
    Panel C: 3Di heatmap (20-state probabilities)

Per-protein detail CSVs also exported:
  {protein_name}_metrics.csv
  {protein_name}_3di_probs.csv
  {protein_name}_ss_fracs.csv

Usage
-----
  python diversity_profiler.py --hits_dir fragment_hits/ --out diversity_results/
  python diversity_profiler.py --hits_dir fragment_hits/ --out diversity_results/ \\
        --k_position 3 --plddt_cutoff 70 --rolling_window 20 --no_plots
  python diversity_profiler.py --hits_dir fragment_hits/ --out diversity_results/ \\
        --workers 8    # parallelize across proteins (default: all cores)
"""

import argparse
import math
import os
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed for batch plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency here
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=UserWarning)


# ── Publication-ready Matplotlib style ───────────────────────────────────────
# Centralised styling so every generated figure looks consistent without
# repeating rcParams calls per plot.
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.titleweight":    "bold",
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          "#e5e5e5",
    "grid.linewidth":      0.5,
    "grid.linestyle":      "-",
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.major.size":    3.5,
    "ytick.major.size":    3.5,
    "lines.linewidth":     1.4,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.transparent": False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# ----- Colorblind-safe Nature/Cell palette -----

C_3DI   = "#0072B2"   # Blue
C_SS    = "#CC7979"   # Purple

C_HELIX = "#D55E00"   # Red-orange
C_SHEET = "#0072B2"   # Blue
C_LOOP  = "#009E73"   # Green


# ── Alphabet constants ────────────────────────────────────────────────────────
STATES_3DI  = list("ACDEFGHIKLMNPQRSTVWY")   # 20-letter 3Di alphabet
MAX_ENT_3DI = np.log2(len(STATES_3DI))       # ≈ 4.322 bits, max possible entropy

# Collapse the extended (DSSP-style) secondary-structure alphabet down to the
# standard three-state H / E / L scheme used throughout this pipeline.
_SS3_MAP = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "L", "S": "L", "C": "L",
    "-": "L", "U": "L", "P": "L",
}

METRIC_COLS = [
    "Entropy_3di",
    "diversity",
    "ss_entropy",
]

FEATURE_COLS = [
    "Max_Entropy_3di",  "Mean_Entropy_3di",
    "Max_diversity",    "Mean_diversity",
    "Max_ss_entropy",   "Mean_ss_entropy",
]


# ── Shared helpers ────────────────────────────────────────────────────────────

def ss3(c: str) -> str:
    """Map a single secondary-structure code to its 3-state (H/E/L) bucket."""
    return _SS3_MAP.get(c.upper(), "L")


def parse_plddt(raw) -> list[float] | None:
    """Parse a comma-separated pLDDT string into a list of floats, or None if empty/invalid."""
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    try:
        return [float(v.strip()) for v in str(raw).split(",")]
    except ValueError:
        return None


def cluster_weights(group: pd.DataFrame) -> dict:
    """
    Return {row_index: weight} where weight = 1 / cluster_size.

    This down-weights hits coming from over-represented clusters so a single
    large cluster of near-identical structures doesn't dominate the entropy
    calculation at a given k-mer position.
    """
    counts = Counter(group["Cluster ID"].tolist())
    return {
        idx: 1.0 / counts[row["Cluster ID"]]
        for idx, row in group.iterrows()
    }


def rolling_avg(series: pd.Series, window: int, min_periods=None) -> pd.Series:
    """Rolling mean, centered.

    min_periods=None (default) matches strict pandas behaviour: a full
    `window` of data is required, so the first/last (window/2) points come
    back as NaN. Pass min_periods=1 to keep the smoothing continuous all the
    way to the edges, using however many neighbours are actually available
    there (tapering, not dropping, near the boundary).
    """
    return series.rolling(window=window, center=True, min_periods=min_periods).mean()


def minmax_norm(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]; returns all-zero if the series is (near) constant."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-10:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ── 3Di entropy ───────────────────────────────────────────────────────────────

def compute_3di_entropy(
    group: pd.DataFrame,
    k_idx: int,
    plddt_cutoff: float,
) -> tuple[float, dict]:
    """
    Cluster-weighted 20-state Shannon entropy at position k_idx (bits).
    Returns (entropy, {state: probability}).

    Hits below the pLDDT cutoff at this position are excluded — low-confidence
    structural predictions shouldn't be allowed to inflate the apparent
    conformational diversity.
    """
    weights = cluster_weights(group)
    wt = {s: 0.0 for s in STATES_3DI}

    for idx, row in group.iterrows():
        plddt = parse_plddt(row.get("pLDDT Scores (per residue)", ""))
        if plddt is not None and plddt[k_idx] < plddt_cutoff:
            continue
        seq = str(row.get("3Di Sequence", "")).strip()
        if k_idx >= len(seq):
            continue
        state = seq[k_idx].upper()
        if state not in STATES_3DI:
            continue
        wt[state] += weights[idx]

    total = sum(wt.values())
    if total == 0:
        return 0.0, {s: 0.0 for s in STATES_3DI}

    entropy = -sum(p * np.log2(p) for w in wt.values() if (p := w / total) > 0)
    probs   = {s: (wt[s] / total) for s in STATES_3DI}
    return entropy, probs


# ── SS features ───────────────────────────────────────────────────────────────

def _scorefn_chen(h: float, e: float, l: float) -> float:
    """Diversity score (Chen et al. 2018): 1 / (h² + e² + l²)."""
    denom = h**2 + e**2 + l**2
    return 1.0 / denom if denom > 0 else 1.0


def _scorefn_entropy_ss3(h: float, e: float, l: float) -> float:
    """3-state Shannon entropy (bits)."""
    return -sum(p * np.log2(p) for p in (h, e, l) if p > 1e-9)


def compute_ss_features(
    group: pd.DataFrame,
    k_idx: int,
    plddt_cutoff: float,
) -> dict:
    """
    Cluster-weighted SS features at position k_idx.
    Returns: diversity, ss_entropy, h_frac, e_frac, l_frac

    Falls back to a uniform 1/3, 1/3, 1/3 split (maximal uncertainty) when
    there's no usable SS data at this position, rather than silently
    reporting zero diversity.
    """
    weights      = cluster_weights(group)
    kept_ss      = []
    kept_weights = []

    for idx, row in group.iterrows():
        plddt = parse_plddt(row.get("pLDDT Scores (per residue)", ""))
        if plddt is not None and plddt[k_idx] < plddt_cutoff:
            continue
        ss = str(row.get("SS Sequence", "")).strip()
        if not ss or k_idx >= len(ss) or ss[k_idx].upper() == "U":
            continue
        kept_ss.append(ss)
        kept_weights.append(weights[idx])

    if not kept_ss:
        return {
            "diversity":  1.0,
            "ss_entropy": 0.0,
            "h_frac":     1 / 3,
            "e_frac":     1 / 3,
            "l_frac":     1 / 3,
        }

    h_sum = e_sum = l_sum = 0.0
    for i, ss in enumerate(kept_ss):
        mid = ss3(ss[k_idx])
        if   mid == "H": h_sum += kept_weights[i]
        elif mid == "E": e_sum += kept_weights[i]
        else:            l_sum += kept_weights[i]

    total_hel = h_sum + e_sum + l_sum
    if total_hel > 0:
        h, e, l = h_sum / total_hel, e_sum / total_hel, l_sum / total_hel
    else:
        h = e = l = 1 / 3

    return {
        "diversity":  _scorefn_chen(h, e, l),
        "ss_entropy": _scorefn_entropy_ss3(h, e, l),
        "h_frac":     h,
        "e_frac":     e,
        "l_frac":     l,
    }


# ── Per-protein processing ────────────────────────────────────────────────────

def load_hits_file(path: Path) -> pd.DataFrame:
    """Load a fragment_picker.py output file (CSV or XLSX), normalising column names."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _compute_position_chunk(
    chunk:        list[tuple],
    k_idx:        int,
    plddt_cutoff: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Compute metric/prob/ss rows for one CHUNK of (K-mer Position, group)
    pairs. This is the unit of work dispatched to a worker process when
    parallelizing WITHIN a single protein (as opposed to across proteins) —
    it's what actually keeps every core busy on a single very long
    sequence, where per-file parallelism alone would have nothing to
    split across.
    """
    metric_rows, prob_rows, ss_frac_rows = [], [], []
    for pos, grp in chunk:
        ent_3di, probs = compute_3di_entropy(grp, k_idx, plddt_cutoff)
        ss             = compute_ss_features(grp, k_idx, plddt_cutoff)

        residue_pos = int(pos) + k_idx   # 1-based position along full sequence

        metric_rows.append({
            "ResiduePosition": residue_pos,
            "Entropy_3di":  round(ent_3di,         6),
            "diversity":    round(ss["diversity"],  6),
            "ss_entropy":   round(ss["ss_entropy"], 6),
        })

        prob_row = {"ResiduePosition": residue_pos}
        prob_row.update({s: round(probs[s], 6) for s in STATES_3DI})
        prob_rows.append(prob_row)

        ss_frac_rows.append({
            "ResiduePosition": residue_pos,
            "H": round(ss["h_frac"], 6),
            "E": round(ss["e_frac"], 6),
            "L": round(ss["l_frac"], 6),
        })

    return metric_rows, prob_rows, ss_frac_rows


def compute_profile(
    df:           pd.DataFrame,
    k_position:   int,
    plddt_cutoff: float,
    executor:     "ProcessPoolExecutor | None" = None,
    n_chunks:     int = 1,
) -> dict:
    """
    Per-residue diversity features, mapped onto the FULL protein length.

    A k-mer starting at window position `p` (1-based, from fragment_picker's
    "K-mer Position") covers residues p .. p+k-1. The intra-k-mer position
    being analysed (`k_position`, 1-based) therefore corresponds to the
    actual residue position:

        residue_position = p + (k_position - 1)

    Because windows only start at p = 1 .. (n - k + 1), the first
    (k_position - 1) residues and the last (k - k_position) residues of the
    protein are never the analysed position of any window and have no
    computed value. Those residues are kept in the output (so the table is
    exactly `n` rows long) with all metrics set to 0.

    Returns a dict with keys:
      metrics_df       — DataFrame (index=ResiduePosition, 1..n) with
                         Entropy_3di, diversity, ss_entropy, zero-filled
                         at the uncovered termini. For detail CSVs/plots.
      metrics_df_valid — same columns, but ONLY the residues that actually
                         had a window computed (no zero-fill, no
                         reindexing). Use this for rolling-window
                         Max_/Mean_ summary features, so the zero-padding
                         never leaks into those numbers.
      probs_3di_df — DataFrame (index=ResiduePosition, 1..n) with 20 state
                     columns
      ss_fracs_df  — DataFrame (index=ResiduePosition, 1..n) with H, E, L
                     columns

    `executor` / `n_chunks`: when an executor is passed and n_chunks > 1,
    the per-K-mer-Position loop below (the actual computational cost, and
    what dominates runtime for very long proteins) is split into n_chunks
    pieces and farmed out across the pool instead of run in a single
    process. This is what lets one huge protein use every core, not just
    one — per-protein/per-file parallelism alone can't help when there's
    only one (or few) protein(s) to spread across workers.
    """
    k_idx = k_position - 1

    if df.empty:
        empty_metrics = pd.DataFrame(columns=METRIC_COLS)
        empty_probs   = pd.DataFrame(columns=STATES_3DI)
        empty_ss      = pd.DataFrame(columns=["H", "E", "L"])
        for d in (empty_metrics, empty_probs, empty_ss):
            d.index.name = "ResiduePosition"
        return {
            "metrics_df":       empty_metrics,
            "metrics_df_valid": empty_metrics.copy(),
            "probs_3di_df":     empty_probs,
            "ss_fracs_df":      empty_ss,
        }

    # k-mer length, read straight from the data (not assumed/hard-coded)
    kmer_lengths = df["K-mer AA"].dropna().astype(str).str.len()
    k = int(kmer_lengths.mode().iloc[0]) if len(kmer_lengths) else k_position

    # last window starts at (n - k + 1) -> full protein length n
    max_kmer_pos = int(df["K-mer Position"].max())
    seq_len = max_kmer_pos + k - 1

    groups = list(df.groupby("K-mer Position"))

    metric_rows  = []
    prob_rows    = []
    ss_frac_rows = []

    if executor is not None and n_chunks > 1 and len(groups) > n_chunks:
        chunk_size = max(1, math.ceil(len(groups) / n_chunks))
        chunks = [groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)]
        for m, p, s in executor.map(
            _compute_position_chunk, chunks,
            [k_idx] * len(chunks), [plddt_cutoff] * len(chunks),
        ):
            metric_rows.extend(m)
            prob_rows.extend(p)
            ss_frac_rows.extend(s)
    else:
        metric_rows, prob_rows, ss_frac_rows = _compute_position_chunk(
            groups, k_idx, plddt_cutoff
        )

    full_index = pd.RangeIndex(1, seq_len + 1, name="ResiduePosition")

    def _make_raw(rows, columns):
        return (
            pd.DataFrame(rows)
            .sort_values("ResiduePosition")
            .set_index("ResiduePosition")
        )[columns]

    def _pad(d):
        # zero-fill the terminal residues that no window ever covers
        return d.reindex(full_index, fill_value=0.0)

    metrics_valid = _make_raw(metric_rows, METRIC_COLS)

    return {
        "metrics_df":       _pad(metrics_valid),
        "metrics_df_valid": metrics_valid,
        "probs_3di_df":     _pad(_make_raw(prob_rows,    STATES_3DI)),
        "ss_fracs_df":      _pad(_make_raw(ss_frac_rows, ["H", "E", "L"])),
    }


# ── Combined aligned three-panel plot ─────────────────────────────────────────

def plot_combined_panels(
    metrics_df:     pd.DataFrame,
    metrics_valid:  pd.DataFrame,
    ss_fracs_df:    pd.DataFrame,
    probs_3di_df:   pd.DataFrame,
    name:           str,
    out_dir:        Path,
    rolling_window: int,
) -> None:
    """
    Publication-quality figure with three perfectly aligned panels.

    Uses pcolormesh for Panel C to ensure exact coordinate alignment
    with Panels A and B. Uniform spine styling, proper spacing, and
    Nature/Cell-style layout.
    """

    # ── Prepare data ──────────────────────────────────────────────────────────
    x = metrics_df.index.values  # residue positions (1-based)
    n_pos = len(x)

    # ----- Raw curves (full-length) -----
    ent_raw = minmax_norm(metrics_df["Entropy_3di"])
    ss_raw  = minmax_norm(metrics_df["ss_entropy"])

    # ----- Smoothed curves (valid residues only) -----
    valid_idx = metrics_valid.index

    ent_roll = pd.Series(np.nan, index=metrics_df.index)
    ss_roll  = pd.Series(np.nan, index=metrics_df.index)

    ent_roll.loc[valid_idx] = rolling_avg(
        minmax_norm(metrics_valid["Entropy_3di"]),
        rolling_window,
        min_periods=1
    )

    ss_roll.loc[valid_idx] = rolling_avg(
        minmax_norm(metrics_valid["ss_entropy"]),
        rolling_window,
        min_periods=1
    )
    # SS fractions (use actual residue positions)
    h = ss_fracs_df["H"].values
    e = ss_fracs_df["E"].values
    l = ss_fracs_df["L"].values
    x_ss = ss_fracs_df.index.values

    # 3Di probability matrix
    mat = probs_3di_df[STATES_3DI].values.T.astype(float)  # (20, n_pos)

    # ── Create figure with 3 subplots ─────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[2.7, 2.0, 2.0],   # B and C same size
        hspace=0.10,
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # Set identical x-axis limits for all panels
    x_lim = (x[0] - 0.5, x[-1] + 0.5)
    ax_a.set_xlim(x_lim)
    ax_b.set_xlim(x_lim)
    ax_c.set_xlim(x_lim)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL A: Combined entropy
    # ──────────────────────────────────────────────────────────────────────────
    ax_a.plot(x, ent_raw, color=C_3DI, lw=0.5, alpha=0.25)
    ax_a.plot(x, ss_raw,  color=C_SS,  lw=0.5, alpha=0.25)
    ax_a.fill_between(x, ent_roll, alpha=0.15, color=C_3DI)
    ax_a.fill_between(x, ss_roll, alpha=0.15, color=C_SS)
    ax_a.plot(x, ent_roll, color=C_3DI, lw=2.0,
              label="3Di entropy", solid_capstyle="round")
    ax_a.plot(x, ss_roll,  color=C_SS,  lw=2.0,
              label="SS entropy",  solid_capstyle="round")

    ax_a.set_ylim(-0.04, 1.08)
    ax_a.set_ylabel("Entropy Combined", fontsize=15, fontweight="bold", labelpad=10)
    ax_a.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_a.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax_a.tick_params(which="minor", length=2, color="#aaaaaa")
    ax_a.tick_params(which="major", labelsize=9, length=4)

    # Legend
    leg_a = ax_a.legend(loc="upper right", frameon=True,
                        framealpha=0.95, edgecolor="#cccccc",
                        handlelength=1.6, handletextpad=0.5, fontsize=12)
    leg_a.get_frame().set_linewidth(0.6)

    # Panel label
    ax_a.text(-0.055, 1.05, "A", transform=ax_a.transAxes,
              fontsize=12, fontweight="bold", ha="right", va="bottom")

    ax_a.grid(True, axis="y", color="#e5e5e5", lw=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL B: SS diversity (stacked H / E / L bars)
    # ──────────────────────────────────────────────────────────────────────────
    width = 0.8

    ax_b.bar(x_ss, h, width, label="Helix",
             color=C_HELIX, alpha=0.82, edgecolor="none")
    ax_b.bar(x_ss, e, width, bottom=h, label="Sheet",
             color=C_SHEET, alpha=0.82, edgecolor="none")
    ax_b.bar(x_ss, l, width, bottom=h + e, label="Coil / Loop",
             color=C_LOOP, alpha=0.82, edgecolor="none")

    ax_b.set_ylim(0, 1.02)
    ax_b.set_ylabel("SS Fraction", fontsize=15, fontweight="bold", labelpad=10)
    ax_b.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_b.tick_params(which="major", labelsize=9, length=4)

    # Legend
    leg_b = ax_b.legend(loc="upper right", ncol=3, frameon=True,
                        framealpha=0.92, edgecolor="#cccccc",
                        handlelength=1.2, handletextpad=0.5,
                        columnspacing=1.0, fontsize=10)
    leg_b.get_frame().set_linewidth(0.6)

    # Panel label
    ax_b.text(-0.055, 1.05, "B", transform=ax_b.transAxes,
              fontsize=12, fontweight="bold", ha="right", va="bottom")

    ax_b.grid(True, axis="y", color="#e5e5e5", lw=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL C: 3Di heatmap using pcolormesh (perfect coordinate alignment)
    # ──────────────────────────────────────────────────────────────────────────

    ax_c.set_facecolor("#fafafa")
    vmax = np.quantile(mat, 0.95)
    BRIGHT_CMAP = "viridis"

    # pcolormesh: x-coordinates, y-coordinates, data
    # Create bin edges for x (residue positions)
    x_bins = np.concatenate([[x[0] - 0.5], x + 0.5])
    y_bins = np.arange(len(STATES_3DI) + 1)

    from matplotlib.colors import PowerNorm

    im = ax_c.pcolormesh(
        x_bins,
        y_bins,
        mat,
        cmap='viridis',
        norm=PowerNorm(gamma=0.5),
        shading='flat',
        edgecolors="#b4b1b1",
        linewidth=0.001
    )
    ax_c.set_aspect('auto')
    # Y-axis: 3Di states
    ax_c.set_yticks(np.arange(len(STATES_3DI)))
    ax_c.set_yticklabels(STATES_3DI, fontsize=15, fontweight="bold",
                         fontfamily="monospace", color="black")
    ax_c.set_ylabel("3Di States", fontsize=15, fontweight="bold", labelpad=10)
    ax_c.tick_params(which="major", labelsize=9, length=4)

    # X-axis: residue position labels (only on Panel C)
    tick_step = max(1, n_pos // 15)
    xtick_idx = [i for i in range(0, n_pos, tick_step)]
    xticks = [x[i] for i in xtick_idx]
    xticklabels = [str(int(x[i])) for i in xtick_idx]

    for ax in [ax_a, ax_b, ax_c]:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    ax_c.set_xticks([x[i] for i in xtick_idx])
    ax_c.set_xticklabels(
        [str(int(x[i])) for i in xtick_idx],
        fontsize=9,
        color="black",
        rotation=0,
    )
    ax_c.set_xlabel("Residue Position", fontsize=13, fontweight="bold", labelpad=10)

    ax_c.grid(False)

    # Panel label
    ax_c.text(-0.055, 1.01, "C", transform=ax_c.transAxes,
              fontsize=12, fontweight="bold", ha="right", va="bottom",
              color="black")

    # Colorbar same height as heatmap
    pos = ax_c.get_position()

    cax = fig.add_axes([
        pos.x1 + 0.02,     # move right
        pos.y0,            # same bottom
        0.012,             # width
        pos.height         # same height as heatmap
    ])

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Probability", fontsize=12, fontweight="bold", labelpad=10, color="black")
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(0.8)
    cb.ax.yaxis.set_tick_params(labelsize=8, length=3, colors="black")

    # ──────────────────────────────────────────────────────────────────────────
    # Uniform spine styling for all panels (clean borders)
    # ──────────────────────────────────────────────────────────────────────────
    for ax in [ax_a, ax_b, ax_c]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    # ──────────────────────────────────────────────────────────────────────────
    # Overall title and finalization
    # ──────────────────────────────────────────────────────────────────────────
    fig.suptitle(name, fontsize=12, fontweight="bold", y=0.995)
    fig.patch.set_facecolor("white")

    # Nature/Cell-style spacing
    fig.subplots_adjust(
        left=0.10,
        right=0.90,   # leave room for colorbar
        top=0.96,
        bottom=0.09,
        hspace=0.08
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────────
    stem = out_dir / f"{name}_aligned_panels"
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def plot_protein(
    profile:        dict,
    name:           str,
    plots_dir:      Path,
    rolling_window: int,
) -> None:
    """Unpack a protein's profile dict and hand it off to the plotting routine."""
    metrics_df       = profile["metrics_df"]
    metrics_valid    = profile["metrics_df_valid"]
    ss_fracs_df      = profile["ss_fracs_df"]
    probs_3di_df     = profile["probs_3di_df"]

    plot_combined_panels(
        metrics_df,
        metrics_valid,
        ss_fracs_df,
        probs_3di_df,
        name,
        plots_dir,
        rolling_window,
    )


# ── Per-protein worker (runs inside a pool process) ──────────────────────────
# This is exactly the body that used to live inline in process_all()'s for-loop.
# Pulling it out to module level is what makes it picklable/dispatchable to a
# ProcessPoolExecutor — each worker computes one protein's full profile
# (feature extraction + plotting are both CPU-bound, so this is where the
# multi-core speedup actually comes from) and writes its own files straight
# to disk. Only the small summary-row dict is shipped back to the main
# process, never the big per-residue DataFrames.

def _process_one_protein(
    fp:             Path,
    detail_dir:     Path,
    plots_dir:      Path,
    k_position:     int,
    plddt_cutoff:   float,
    rolling_window: int,
    do_plots:       bool,
) -> tuple[str, dict | None, str | None]:
    """
    Returns (name, record_or_None, error_or_None).
      - record is the {"Protein": ..., "Max_..."/"Mean_..."} summary row.
      - record is None (with a status string) if the protein was skipped
        (no usable data) or errored out — either way the batch continues.
    """
    name = fp.stem
    if name.endswith("_hits"):
        name = name[:-5]

    try:
        df      = load_hits_file(fp)
        profile = compute_profile(df, k_position, plddt_cutoff)
        metrics       = profile["metrics_df"]        # full length, zero-padded
        metrics_valid = profile["metrics_df_valid"]  # only computed residues

        if metrics.empty:
            return name, None, "no usable data"

        # Edge-safe rolling profile for structure-based colouring
        # (e.g. blue-white-red per-residue mapping): rolled with
        # min_periods=1 on the valid-only data, so the first/last
        # ~(rolling_window/2) real residues are smoothed with fewer
        # neighbours instead of coming back NaN, and the smoothing
        # is never contaminated by the zero-padded k-mer termini.
        # Residues with no computed window at all stay 0.
        roll_valid = pd.DataFrame(index=metrics_valid.index)
        for col in ["Entropy_3di", "ss_entropy"]:
            roll_valid[f"{col}_roll{rolling_window}"] = rolling_avg(
                metrics_valid[col], rolling_window, min_periods=1
            )
        roll_full = roll_valid.reindex(metrics.index, fill_value=0.0)
        metrics = metrics.join(roll_full)

        # Save per-protein detail tables (full-length, residue-indexed)
        metrics.to_csv(detail_dir / f"{name}_metrics.csv")
        profile["probs_3di_df"].to_csv(detail_dir / f"{name}_3di_probs.csv")
        profile["ss_fracs_df"].to_csv(detail_dir / f"{name}_ss_fracs.csv")

        if do_plots:
            plot_protein(profile, name, plots_dir, rolling_window)

        # Rolling Max_/Mean_ summary features: computed on metrics_valid
        # only (no zero-padded termini), matching the original
        # window-only behaviour.
        rec = {"Protein": name}
        for col in METRIC_COLS:
            r = metrics_valid[col].rolling(
                window=rolling_window, center=True
            ).mean().dropna()
            rec[f"Max_{col}"]  = round(float(r.max()),  5) if len(r) else float("nan")
            rec[f"Mean_{col}"] = round(float(r.mean()), 5) if len(r) else float("nan")

        return name, rec, None

    except Exception as exc:  # noqa: BLE001 — surfaced to the main process, not swallowed
        return name, None, str(exc)


def _plot_protein_task(
    profile:        dict,
    name:           str,
    plots_dir:      Path,
    rolling_window: int,
) -> str:
    """Thin picklable wrapper so plotting can be submitted to a pool as its own task."""
    plot_protein(profile, name, plots_dir, rolling_window)
    return name


def _finish_protein(
    name:           str,
    profile:        dict,
    detail_dir:     Path,
    rolling_window: int,
) -> dict | None:
    """
    Shared tail-end logic (rolling-smoothed columns, detail CSVs, summary
    row) used by BOTH parallelization strategies below, so the two code
    paths can't silently drift apart. Returns the summary record, or None
    if the protein had no usable data.
    """
    metrics       = profile["metrics_df"]
    metrics_valid = profile["metrics_df_valid"]

    if metrics.empty:
        return None

    roll_valid = pd.DataFrame(index=metrics_valid.index)
    for col in ["Entropy_3di", "ss_entropy"]:
        roll_valid[f"{col}_roll{rolling_window}"] = rolling_avg(
            metrics_valid[col], rolling_window, min_periods=1
        )
    roll_full = roll_valid.reindex(metrics.index, fill_value=0.0)
    metrics = metrics.join(roll_full)

    metrics.to_csv(detail_dir / f"{name}_metrics.csv")
    profile["probs_3di_df"].to_csv(detail_dir / f"{name}_3di_probs.csv")
    profile["ss_fracs_df"].to_csv(detail_dir / f"{name}_ss_fracs.csv")

    rec = {"Protein": name}
    for col in METRIC_COLS:
        r = metrics_valid[col].rolling(
            window=rolling_window, center=True
        ).mean().dropna()
        rec[f"Max_{col}"]  = round(float(r.max()),  5) if len(r) else float("nan")
        rec[f"Mean_{col}"] = round(float(r.mean()), 5) if len(r) else float("nan")
    return rec


# ── Dataset loop ──────────────────────────────────────────────────────────────
#
# Two parallelization strategies, chosen automatically based on how many
# protein hit-files there are relative to worker count:
#
#  (A) MANY files, >= workers  -> parallelize ACROSS proteins (one whole
#      protein per worker). Simple, low overhead, great when you have a
#      batch of many proteins.
#
#  (B) FEW files, < workers    -> parallelize WITHIN each protein, across
#      its k-mer positions, using a single shared pool. This is the path
#      that matters for e.g. one 1800-residue protein: strategy (A) would
#      only ever occupy ONE core for it, no matter how many workers you
#      pass, because there's only one task to hand out. Plot rendering is
#      also dispatched to the same pool asynchronously so it overlaps with
#      the next protein's computation instead of blocking on it.

def process_all(
    hits_dir:       Path,
    out_dir:        Path,
    k_position:     int,
    plddt_cutoff:   float,
    rolling_window: int,
    do_plots:       bool,
    workers:        int = 1,
) -> pd.DataFrame:
    """
    Iterate over every hit file in `hits_dir`, compute per-residue profiles
    and summary features, optionally generate plots, and write out the
    combined summary CSV. See module notes above `process_all` for how
    `workers` gets applied depending on file count vs. worker count.
    """

    plots_dir  = out_dir / "plots"
    detail_dir = out_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    if do_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    hit_files = sorted(
        list(hits_dir.glob("*.csv")) + list(hits_dir.glob("*.xlsx"))
    )
    if not hit_files:
        raise FileNotFoundError(f"No CSV / XLSX files found in: {hits_dir}")

    print(f"\nFound {len(hit_files)} hit file(s) in {hits_dir}")
    print(f"Workers: {workers}")

    records = []
    errors  = []
    t0 = time.time()

    if workers <= 1:
        # Sequential fallback — same code path, just called in-process.
        for fp in tqdm(hit_files, desc="Processing proteins"):
            name, rec, err = _process_one_protein(
                fp, detail_dir, plots_dir,
                k_position, plddt_cutoff, rolling_window, do_plots,
            )
            if rec is not None:
                records.append(rec)
            elif err:
                errors.append((name, err))

    elif len(hit_files) >= workers:
        # Strategy (A): parallel across proteins — enough files to keep
        # every worker busy with a whole protein each.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_protein,
                    fp, detail_dir, plots_dir,
                    k_position, plddt_cutoff, rolling_window, do_plots,
                ): fp
                for fp in hit_files
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Processing proteins"
            ):
                name, rec, err = fut.result()
                if rec is not None:
                    records.append(rec)
                elif err:
                    errors.append((name, err))

    else:
        # Strategy (B): fewer files than workers (the common case for one
        # or a handful of very long proteins) — parallelize WITHIN each
        # protein instead, across a single shared pool.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending_plots = []

            for fp in tqdm(hit_files, desc="Processing proteins"):
                name = fp.stem
                if name.endswith("_hits"):
                    name = name[:-5]
                try:
                    df = load_hits_file(fp)
                    profile = compute_profile(
                        df, k_position, plddt_cutoff,
                        executor=pool, n_chunks=workers,
                    )
                    rec = _finish_protein(name, profile, detail_dir, rolling_window)
                    if rec is None:
                        errors.append((name, "no usable data"))
                        continue
                    records.append(rec)

                    if do_plots:
                        pending_plots.append((
                            name,
                            pool.submit(_plot_protein_task, profile, name, plots_dir, rolling_window),
                        ))
                except Exception as exc:
                    errors.append((name, str(exc)))

            if pending_plots:
                print(f"\nWaiting for {len(pending_plots)} plot(s) to finish rendering...")
                for name, fut in tqdm(pending_plots, desc="Rendering plots"):
                    try:
                        fut.result()
                    except Exception as exc:
                        errors.append((f"{name} (plot)", str(exc)))

    elapsed = time.time() - t0
    print(f"\nProcessed {len(hit_files)} protein(s) in {elapsed:.1f}s.")
    if errors:
        print(f"  {len(errors)} skipped/errored:")
        for name, err in errors:
            print(f"    - {name}: {err}")

    if not records:
        print("\nNo proteins processed successfully.")
        return pd.DataFrame()

    summary = (
        pd.DataFrame(records)
        .set_index("Protein")
        .reindex(columns=FEATURE_COLS)
    )

    out_csv = out_dir / "combined_diversity_summary.csv"
    summary.to_csv(out_csv)
    print(f"\nSummary CSV → {out_csv}")
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute diversity features from Morpheus hit CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hits_dir", required=True,
                        help="Folder of *_hits.csv / .xlsx files from fragment_picker.py")
    parser.add_argument("--out", default="./diversity_results",
                        help="Output directory (plots, detail CSVs, summary)")
    parser.add_argument("--k_position", type=int, default=3,
                        help="1-based intra-k-mer position to analyse (1 … k)")
    parser.add_argument("--plddt_cutoff", type=float, default=70.0,
                        help="Minimum pLDDT at k_position to keep a hit")
    parser.add_argument("--rolling_window", type=int, default=20,
                        help="Smoothing window width (k-mer positions)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip per-protein plots")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes "
                             "(default: all available CPU cores; use 1 "
                             "for the old sequential behaviour)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"K position      : {args.k_position}")
    print(f"pLDDT cutoff    : {args.plddt_cutoff}")
    print(f"Rolling window  : {args.rolling_window}")
    print(f"Plots           : {'no' if args.no_plots else 'yes'}")

    summary = process_all(
        hits_dir       = Path(args.hits_dir),
        out_dir        = out_dir,
        k_position     = args.k_position,
        plddt_cutoff   = args.plddt_cutoff,
        rolling_window = args.rolling_window,
        do_plots       = not args.no_plots,
        workers        = max(1, args.workers),
    )

    if not summary.empty:
        print("\n── Feature summary ─────────────────────────────────────────────")
        print(summary.describe().round(4).to_string())
        print()


if __name__ == "__main__":
    main()