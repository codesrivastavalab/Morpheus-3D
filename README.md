# Morpheus-3D

**Structural diversity–guided detection and localization of protein fold switching.**

Morpheus-3D is a sequence-based framework for identifying fold-switching
(metamorphic) proteins and localizing the residues responsible for the
switch. It quantifies residue-level tertiary structural diversity using
Shannon entropy over the Foldseek 3Di structural alphabet, combined with
secondary-structure entropy, and scores both with a trained SVM classifier.

An interactive web platform for exploring predictions is available at
**[morpheus.slicearrow.com](https://morpheus.slicearrow.com/)**.

![Morpheus-3D workflow](docs/workflow.png)


## Database

The SQLite k-mer index and sequence database are hosted on Hugging Face:
**[sreeharshk/Morpheus3D-Database](https://huggingface.co/datasets/sreeharshk/Morpheus3D-Database)**

Download `Morpheus3D_database_sqlite.zip` and unzip it to get
`Morpheus3D_database.sqlite`, the `--db` argument used below.

## Scripts

| Script | Purpose |
|---|---|
| `fragment_picker.py` | K-mer window query against the SQLite index. Writes one `*_hits.csv` per sequence. |
| `diversity_profiler.py` | Per-residue entropy/diversity features, detail CSVs, summary CSV, and optional per-protein figure. |
| `predict_foldswitch.py` | SVM classification + consensus fold-switch region calling. |
| `pipeline.py` | Runs fragment retrieval, structural diversity profiling, and fold-switch prediction end-to-end with optional multi-core parallel processing. |

## Requirements

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate morpheus3d
```

(Or: `pip install pandas numpy matplotlib scikit-learn joblib tqdm openpyxl`.)


### Input requirements

- Input sequences must be at least **26 amino acids** long.
- During batch processing, sequences shorter than **26 residues** are automatically skipped while valid sequences continue processing. If all input sequences are shorter than 26 residues, the pipeline exits with a clear error message.
- Only the **20 standard amino acids** (`A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`) are supported.
- Sequences containing ambiguous or non-standard amino acids (`B`, `J`, `O`, `U`, `X`, or `Z`) are not supported.

## Usage

`--input` accepts a single sequence or a batch — a FASTA file (`.fasta`)
with any number of `>` entries. Each sequence is scored independently, and
every stage's output has one row per input sequence.

```bash
python pipeline.py \
    --input seqs.fasta \
    --db Morpheus3D_database.sqlite \
    --model svm_pipeline_model.pkl \
    --workers 0 \
    --out results/
```

Skip classification:

```bash
python pipeline.py --input seqs.fasta --db Morpheus3D_database.sqlite --skip_prediction
```


K-mer length is fixed at 7 (matches the index). Other flags:

| Flag | Default | Meaning |
|---|---|---|
| `--k_position` | 3 | Position within the k-mer to analyse |
| `--plddt_cutoff` | 70.0 | Minimum per-residue pLDDT to keep a structural hit |
| `--rolling_window` | 20 | Smoothing window for entropy curves and hotspot calling |
| `--workers` | 0 | Number of parallel worker processes to use (0 = use all available CPU cores) |
| `--no_plots` | off | Skip per-protein figures |



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
Every protein scored by the classifier — one row per input sequence,
whether you ran on a single sequence or a multi-FASTA/XLSX batch. Columns:
- `Prediction` — 0 (monomorphic) or 1 (predicted fold-switching)
- `Probability` — classifier confidence
- `Entropy_3di_roll20_hotspots`, `SS_Entropy_roll20_hotspots` — raw
  per-metric hotspot regions before reconciliation
- `Fold_Switching_Regions_Predicted` — consensus region (overlap of the
  two hotspot sets where both fire, otherwise whichever one fired)
- `FoldSwitch_Residues` — residue count spanned by the consensus region

### `predictions/positive_predictions.xlsx`
Same columns as above, filtered to `Prediction == 1` — just the proteins
from the batch classified as fold-switching.


## Citation
If you use Morpheus-3D, please cite:

Sreeharsh Kuniyil, Vijay Subramanian, Akanksha Arun, Anand Lakshmanan, Ashok Sekhar, Anand Srivastava. **Morpheus-3D: Structural Diversity-Guided Detection and Localization of Protein Fold Switching.** bioRxiv (2026). https://doi.org/10.64898/2026.08.16.745091
