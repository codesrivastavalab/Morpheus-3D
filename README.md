# Morpheus-3D

**Structural diversity–guided detection and localization of protein fold switching.**

Morpheus-3D is a sequence-based framework for identifying fold-switching
(metamorphic) proteins and localizing the residues responsible for the
switch. It quantifies residue-level tertiary structural diversity using
Shannon entropy over the Foldseek 3Di structural alphabet, combined with
secondary-structure entropy, and scores both with a trained SVM classifier.

An interactive web platform for exploring predictions is available at
**[morpheus.slicearrow.com](https://morpheus.slicearrow.com/)**.

![workflow](assets/workflow.png)

Given a set of protein sequences, the pipeline:

1. Queries a pre-built SQLite k-mer index for structural (3Di + secondary
   structure) hits.
2. Computes per-residue diversity features — 3Di Shannon entropy, SS
   diversity, and SS entropy — cluster-weighted to avoid over-represented
   structural clusters skewing the signal.
3. Runs a trained SVM classifier on the summary features to predict
   fold-switching status, and calls consensus fold-switch region(s) from
   rolling-window hotspots.

## Database

The SQLite k-mer index and sequence database are hosted on Hugging Face:
**[sreeharshk/Morpheus3D-Database](https://huggingface.co/datasets/sreeharshk/Morpheus3D-Database)**

Download `Morpheus3D_database_sqlite.zip` and unzip it to get
`kmer_indexed_db.sqlite`, the `--db` argument used below.

## Scripts

| Script | Purpose |
|---|---|
| `fragment_picker.py` | K-mer window query against the SQLite index. Writes one `*_hits.csv` per sequence. |
| `diversity_profiler.py` | Per-residue entropy/diversity features, detail CSVs, summary CSV, and optional per-protein figure. |
| `predict_foldswitch.py` | SVM classification + consensus fold-switch region calling. |
| `pipeline.py` | Runs all three stages end-to-end. |

## Requirements

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate morpheus3d
```

(Or, if you'd rather manage it yourself: `pip install pandas numpy matplotlib scikit-learn joblib tqdm openpyxl`.)

Also needed: the SQLite index above, and a trained SVM pickle
(`svm_pipeline_model.pkl`) exposing `.predict()` / `.predict_proba()`.

## Usage

```bash
python pipeline.py \
    --input seqs.fasta \
    --db kmer_indexed_db.sqlite \
    --model svm_pipeline_model.pkl \
    --out results/
```

Skip classification:

```bash
python pipeline.py --input seqs.fasta --db kmer_indexed_db.sqlite --skip_prediction
```

K-mer length is fixed at 7 (matches the index). Other flags:

| Flag | Default | Meaning |
|---|---|---|
| `--k_position` | 3 | 1-based position within the k-mer to analyse |
| `--plddt_cutoff` | 70.0 | Minimum per-residue pLDDT to keep a structural hit |
| `--rolling_window` | 20 | Smoothing window for entropy curves and hotspot calling |
| `--no_plots` | off | Skip per-protein figures |

### Individual stages

```bash
python fragment_picker.py --input seqs.fasta --db kmer_indexed_db.sqlite --out fragment_hits/

python diversity_profiler.py --hits_dir fragment_hits/ --out diversity_results/ \
    --k_position 3 --plddt_cutoff 70 --rolling_window 20

python predict_foldswitch.py \
    --summary diversity_results/combined_diversity_summary.csv \
    --detail_dir diversity_results/detail \
    --model svm_pipeline_model.pkl \
    --out predictions/
```

## Output layout

```
results/
├── fragment_hits/{protein}_hits.csv
├── diversity_results/
│   ├── detail/{protein}_metrics.csv | _3di_probs.csv | _ss_fracs.csv
│   ├── plots/{protein}_aligned_panels.pdf/.png
│   └── combined_diversity_summary.csv
└── predictions/
    ├── all_predictions.xlsx
    └── positive_predictions.xlsx
```

### `fragment_hits/{protein}_hits.csv`
Raw output of `fragment_picker.py`. One row per k-mer window per database
hit: window position, the k-mer itself, the matched protein/cluster ID,
its 3Di and secondary-structure k-mer, and per-residue pLDDT. Windows with
no database match are kept as `NO_HIT` rows rather than dropped.

### `diversity_results/detail/{protein}_metrics.csv`
Per-residue table (one row per position, full protein length) with
`Entropy_3di`, `diversity`, and `ss_entropy`, plus their rolling-window
smoothed versions. This is the file `predict_foldswitch.py` reads to call
hotspot regions.

### `diversity_results/detail/{protein}_3di_probs.csv`
Per-residue probability distribution over the 20 3Di states — the raw
values behind the heatmap panel of the figure.

### `diversity_results/detail/{protein}_ss_fracs.csv`
Per-residue fractional occupancy of Helix / Sheet / Loop across all
retrieved hits — the raw values behind the stacked-bar panel of the figure.

### `diversity_results/plots/{protein}_aligned_panels.pdf` / `.png`
The three-panel figure per protein: combined entropy trace, SS
composition, and 3Di probability heatmap, all aligned on the same residue
axis. Skipped if `--no_plots` is set.

### `diversity_results/combined_diversity_summary.csv`
One row per protein — the feature table consumed by the SVM classifier.
Columns: `Max_Entropy_3di`, `Mean_Entropy_3di`, `Max_diversity`,
`Mean_diversity`, `Max_ss_entropy`, `Mean_ss_entropy` (rolling-window
max/mean of each metric across the protein).

### `predictions/all_predictions.xlsx`
Every protein scored by the classifier, with:
- `Prediction` — 0 (monomorphic) or 1 (predicted fold-switching)
- `Probability` — classifier confidence
- `Entropy_3di_roll20_hotspots`, `SS_Entropy_roll20_hotspots` — raw
  per-metric hotspot regions before reconciliation
- `Fold_Switching_Regions_Predicted` — consensus region (overlap of the
  two hotspot sets where both fire, otherwise whichever one fired)
- `FoldSwitch_Residues` — residue count spanned by the consensus region

### `predictions/positive_predictions.xlsx`
Same columns as above, filtered to `Prediction == 1` — the fold-switch
candidates.

## Database schema

```sql
CREATE TABLE kmers (
    kmer_id INTEGER PRIMARY KEY,
    kmer_aa TEXT
);

CREATE TABLE hits (
    kmer_id     INTEGER REFERENCES kmers(kmer_id),
    protein_id  TEXT,
    cluster_id  TEXT,
    ss_kmer     TEXT,
    tdi_kmer    TEXT,
    plddt_kmer  TEXT,
    plddt_mean  REAL
);
```

## Method notes

- Hits weighted by `1 / cluster_size` to prevent large clusters dominating entropy.
- Hits below `--plddt_cutoff` at the analysed position are excluded.
- K-mer termini with no computed window are zero-padded in detail CSVs/plots
  but excluded from `Max_/Mean_` summary features.
- Fold-switch regions: rolling-averaged entropy/SS-entropy thresholded
  (0.95 / 0.40, window 20) into hotspot runs; overlapping regions merged
  into the consensus call, or the single metric used if only one fires.

## Citation

If you use Morpheus-3D, please cite:

> Kuniyil, S., Subramanian, V., Sekhar, A., Arun, A., Lakshmanan, A. &
> Srivastava, A. Morpheus-3D: Structural Diversity-Guided Detection and
> Localization of Protein Fold Switching. *bioRxiv* (2026).

```bibtex
@article{kuniyil2026morpheus3d,
  title   = {Morpheus-3D: Structural Diversity-Guided Detection and
             Localization of Protein Fold Switching},
  author  = {Kuniyil, Sreeharsh and Subramanian, Vijay and Sekhar, Ashok
             and Arun, Akanksha and Lakshmanan, Anand and
             Srivastava, Anand},
  journal = {bioRxiv},
  year    = {2026}
}
```

## License

MIT
