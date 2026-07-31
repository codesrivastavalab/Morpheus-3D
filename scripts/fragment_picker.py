#!/usr/bin/env python3
"""
fragment_picker.py
==================
Query the Morpheus k-mer SQLite index with a FASTA or XLSX file.

For every k-mer window in each input sequence, this script looks up all
database hits and writes one standardised CSV per sequence to the output
directory. These CSVs are consumed directly by diversity_profiler.py.

Output CSV columns
------------------
  K-mer Position | K-mer AA | Protein ID | Cluster ID |
  SS Sequence    | 3Di Sequence | pLDDT Scores (per residue) | Mean pLDDT

Usage
-----
  python fragment_picker.py --input seqs.fasta --db kmer_index.sqlite
  python fragment_picker.py --input seqs.xlsx  --db kmer_index.sqlite --out hits/

Minimum sequence length
------------------------
Sequences shorter than MIN_SEQ_LENGTH (26 aa) are skipped, since the
downstream rolling-window fold-switch prediction (predict_foldswitch.py)
cannot produce a meaningful result on a profile shorter than its own
smoothing window. In a multi-sequence FASTA/XLSX input, offending proteins
are logged and skipped while the rest of the batch continues normally; if
every sequence in the input is too short, the run aborts with an error
rather than handing an empty result set to the next pipeline stage.
"""

import argparse
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# K-mer length is fixed to match how the SQLite index was built — it isn't
# a runtime choice, so it lives here as a constant rather than a CLI flag.
K_MER_LENGTH = 7

# Downstream, predict_foldswitch.py smooths the per-position entropy profile
# with a centered rolling window (default width 20) before calling
# fold-switch hotspot regions. A sequence shorter than that window can never
# produce a valid rolling average, so predictions on it would be meaningless
# (empty/blank regions) even if the k-mer query itself succeeds.
#
# We enforce a hard floor here — before any DB queries are issued — equal to
# the rolling window plus enough residues to actually slide it across the
# sequence:
#   MIN_SEQ_LENGTH = ROLLING_WINDOW_DEFAULT + K_MER_LENGTH - 1 = 20 + 7 - 1 = 26
ROLLING_WINDOW_DEFAULT = 20
MIN_SEQ_LENGTH = ROLLING_WINDOW_DEFAULT + K_MER_LENGTH - 1  # 26 residues


# ── Database ──────────────────────────────────────────────────────────────────

def open_db_readonly(path: str) -> sqlite3.Connection:
    """
    Open the SQLite index in true read-only mode (via the `mode=ro` URI flag,
    not just by convention), so we never accidentally take a write lock on
    the index while querying it. The cache/mmap settings just speed up
    repeated k-mer lookups on a large index.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size  = -524288")     # 512 MB page cache
    conn.execute("PRAGMA mmap_size   = 4294967296")  # 4 GB memory-map
    conn.execute("PRAGMA temp_store  = MEMORY")
    conn.row_factory = sqlite3.Row
    return conn


def query_kmer(conn: sqlite3.Connection, kmer_aa: str) -> list:
    """Look up every hit in the database whose amino-acid k-mer matches exactly."""
    cur = conn.execute(
        """
        SELECT h.protein_id,
               h.cluster_id,
               h.ss_kmer,
               h.tdi_kmer,
               h.plddt_kmer,
               h.plddt_mean
        FROM   hits  h
        JOIN   kmers m ON m.kmer_id = h.kmer_id
        WHERE  m.kmer_aa = ?
        """,
        (kmer_aa,),
    )
    return cur.fetchall()


# Max host parameters per query. SQLite's own ceiling (SQLITE_MAX_VARIABLE_NUMBER)
# is 999 on most default builds, so we stay comfortably under that.
_BATCH_CHUNK = 500


def query_kmers_batch(conn: sqlite3.Connection, kmers: list[str]) -> dict:
    """
    Look up many k-mers in as few round-trips as possible.

    A ~1800-residue protein has ~1800 overlapping k-mer windows; querying
    them one at a time (the original behaviour) means ~1800 separate
    round-trips to SQLite per protein. Here we de-duplicate the k-mers
    (repeats are common in real sequences) and look them up with chunked
    `IN (...)` queries instead, which is the single biggest speed win for
    long proteins — independent of, and stacking with, the process-level
    parallelism below.

    Returns {kmer_aa: [row, row, ...]} for every kmer in `kmers` (missing
    keys just mean no hits and are omitted).
    """
    result: dict = {}
    unique_kmers = list(dict.fromkeys(kmers))  # de-dupe, preserve order

    for i in range(0, len(unique_kmers), _BATCH_CHUNK):
        chunk = unique_kmers[i : i + _BATCH_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        cur = conn.execute(
            f"""
            SELECT m.kmer_aa   AS kmer_aa,
                   h.protein_id,
                   h.cluster_id,
                   h.ss_kmer,
                   h.tdi_kmer,
                   h.plddt_kmer,
                   h.plddt_mean
            FROM   hits  h
            JOIN   kmers m ON m.kmer_id = h.kmer_id
            WHERE  m.kmer_aa IN ({placeholders})
            """,
            chunk,
        )
        for row in cur.fetchall():
            result.setdefault(row["kmer_aa"], []).append(row)

    return result


# ── Sequence readers ──────────────────────────────────────────────────────────

def read_fasta(path: str) -> list[tuple[str, str]]:
    """Minimal FASTA parser: returns a list of (header, sequence) tuples."""
    entries, header, seq = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    entries.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header:
        entries.append((header, "".join(seq)))
    return entries


def read_xlsx_seqs(path: str) -> list[tuple[str, str]]:
    """
    Read sequences from an XLSX file.

    Recognised ID column names  : ID, Name, Header, Accession, Protein, Protein_ID
    Recognised Seq column names : Sequence, Seq, AA, AA_Sequence
    Falls back to (col0, col1) when the sheet has exactly two columns, so
    quick ad-hoc spreadsheets still work without renaming headers.
    """
    df = pd.read_excel(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

    id_aliases  = {"id", "name", "header", "accession", "protein", "protein_id"}
    seq_aliases = {"sequence", "seq", "aa", "aa_sequence"}

    id_col  = next((c for c in df.columns if c.lower() in id_aliases),  None)
    seq_col = next((c for c in df.columns if c.lower() in seq_aliases), None)

    if id_col is None or seq_col is None:
        if len(df.columns) == 2:
            id_col, seq_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(
                f"Cannot detect ID / Sequence columns in {path}.\n"
                f"Found: {list(df.columns)}\n"
                "Rename them to 'ID' and 'Sequence' (or similar)."
            )

    return [
        (row[id_col].strip(), row[seq_col].strip().upper())
        for _, row in df.iterrows()
        if row[seq_col].strip()
    ]


def load_sequences(path: str) -> list[tuple[str, str]]:
    """Dispatch to the right reader based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in {".fa", ".fasta", ".fna", ".faa"}:
        return read_fasta(path)
    if ext in {".xlsx", ".xls"}:
        return read_xlsx_seqs(path)
    raise ValueError(f"Unsupported format '{ext}'. Expected .fasta or .xlsx.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(header: str) -> str:
    """Turn a FASTA header / sequence ID into a filesystem-safe file stem."""
    word = header.split()[0]
    return "".join(c for c in word if c.isalnum() or c in "-_.")


def pick_fragments(sequence: str, conn: sqlite3.Connection, k: int) -> pd.DataFrame:
    """
    Slide a k-mer window across `sequence`, query each k-mer against the
    database, and return a DataFrame with the canonical column layout
    expected by diversity_profiler.py.

    Windows with no database hit are still written out as "NO_HIT" rows
    (rather than being dropped) so the downstream profiler can see exactly
    which positions had no coverage.
    """
    n = len(sequence)
    windows = [sequence[pos : pos + k] for pos in range(n - k + 1)]

    hits_by_kmer = query_kmers_batch(conn, windows)

    rows = []
    for pos, kmer in enumerate(windows):
        kmer_pos = pos + 1      # 1-based window position along the query
        hits = hits_by_kmer.get(kmer, [])

        if hits:
            for h in hits:
                rows.append({
                    "K-mer Position":             kmer_pos,
                    "K-mer AA":                   kmer,
                    "Protein ID":                 h["protein_id"],
                    "Cluster ID":                 h["cluster_id"],
                    "SS Sequence":                h["ss_kmer"]    or "",
                    "3Di Sequence":               h["tdi_kmer"]   or "",
                    "pLDDT Scores (per residue)": h["plddt_kmer"] or "",
                    "Mean pLDDT":                 h["plddt_mean"],
                })
        else:
            rows.append({
                "K-mer Position":             kmer_pos,
                "K-mer AA":                   kmer,
                "Protein ID":                 "NO_HIT",
                "Cluster ID":                 "N/A",
                "SS Sequence":                "",
                "3Di Sequence":               "",
                "pLDDT Scores (per residue)": "",
                "Mean pLDDT":                 None,
            })

    return pd.DataFrame(rows)


# ── Parallel worker plumbing ────────────────────────────────────────────────
# Each worker process opens ONE read-only SQLite connection (in _init_worker)
# and reuses it for every sequence handed to that process — we don't want to
# reopen the DB per-sequence, and a sqlite3.Connection can't be pickled
# across a process boundary anyway, so it has to live in worker-global state
# rather than being passed as a task argument.
_worker_conn    = None
_worker_k       = None
_worker_out_dir = None


def _init_worker(db_path: str, k_mer_length: int, out_dir: str) -> None:
    global _worker_conn, _worker_k, _worker_out_dir
    _worker_conn    = open_db_readonly(db_path)
    _worker_k       = k_mer_length
    _worker_out_dir = Path(out_dir)


def _process_one_sequence(header_seq: tuple) -> dict:
    """Run in a worker process: query + write one sequence's hit CSV."""
    header, seq = header_seq
    name = safe_name(header)
    length = len(seq)

    if length < _worker_k:
        # Can't even form a single k-mer window — nothing to query.
        return {"name": name, "status": "skip_short_kmer", "length": length}

    if length < MIN_SEQ_LENGTH:
        # Long enough to query, but too short for the downstream rolling-
        # window fold-switch prediction to produce a meaningful result.
        return {"name": name, "status": "skip_short_min", "length": length}

    try:
        df = pick_fragments(seq, _worker_conn, _worker_k)
        df.to_csv(_worker_out_dir / f"{name}_hits.csv", index=False)
        return {"name": name, "status": "ok", "n_windows": len(seq) - _worker_k + 1}
    except Exception as exc:
        return {"name": name, "status": "error", "error": str(exc)}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Morpheus k-mer SQLite index from a FASTA or XLSX file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Input FASTA (.fa / .fasta) or Excel (.xlsx) file")
    parser.add_argument("--db",
                        default="/home/user/Documents/Main/Morpheus_3di_Paper/"
                                "database/kmer_indexed_db.sqlite",
                        help="Path to the Morpheus SQLite index")
    parser.add_argument("--out", default="./fragment_hits",
                        help="Output directory for per-sequence CSV files")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel worker processes, one "
                             "sequence in flight per worker (0 = use all "
                             "available CPU cores)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading sequences : {args.input}")
    try:
        sequences = load_sequences(args.input)
    except Exception as exc:
        sys.exit(f"[ERROR] {exc}")

    if not sequences:
        sys.exit("[ERROR] No sequences found in input file.")

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    # Never spin up more workers than there are sequences to process.
    n_workers = max(1, min(n_workers, len(sequences)))

    print(f"  {len(sequences)} sequence(s) found.")
    print(f"Database          : {args.db}")
    print(f"K-mer length      : {K_MER_LENGTH}")
    print(f"Output dir        : {out_dir}")
    print(f"Workers           : {n_workers}\n")

    # Quick open/close sanity check in the main process so a bad --db path
    # fails fast with a clear message, instead of failing inside every
    # worker process separately.
    try:
        open_db_readonly(args.db).close()
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot open database: {exc}")

    ok = 0
    skipped_kmer = 0
    skipped_min = 0
    errored = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(args.db, K_MER_LENGTH, str(out_dir)),
    ) as pool:
        futures = {pool.submit(_process_one_sequence, s): s[0] for s in sequences}

        # Every sequence is handled independently, so a short/malformed/
        # errored protein never blocks the rest of the batch — it's just
        # logged and skipped while the others keep processing normally.
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Querying k-mers"):
            res = fut.result()
            status = res["status"]

            if status == "ok":
                ok += 1
            elif status == "skip_short_kmer":
                tqdm.write(
                    f"  [SKIP] '{res['name']}' ({res['length']} aa) — "
                    f"shorter than the k-mer length (k={K_MER_LENGTH}); "
                    f"cannot be queried against the index."
                )
                skipped_kmer += 1
            elif status == "skip_short_min":
                tqdm.write(
                    f"  [SKIP] '{res['name']}' ({res['length']} aa) — "
                    f"below the {MIN_SEQ_LENGTH}-residue minimum required "
                    f"for fold-switch prediction (the downstream rolling-"
                    f"window analysis needs at least {MIN_SEQ_LENGTH} aa "
                    f"to produce a meaningful result)."
                )
                skipped_min += 1
            elif status == "error":
                tqdm.write(f"  [ERROR] '{res['name']}' — {res['error']}")
                errored += 1

    total_skipped = skipped_kmer + skipped_min

    print(f"\nDone.  Results saved to: {out_dir}/")
    print(f"  {ok} sequence(s) processed successfully.")
    if skipped_kmer:
        print(f"  {skipped_kmer} sequence(s) skipped — shorter than k={K_MER_LENGTH}.")
    if skipped_min:
        print(f"  {skipped_min} sequence(s) skipped — shorter than the "
              f"{MIN_SEQ_LENGTH} aa minimum needed for fold-switch prediction.")
    if errored:
        print(f"  {errored} sequence(s) failed — see [ERROR] lines above.")

    # If literally nothing survived filtering, stop here with a clear,
    # actionable error instead of silently handing an empty output
    # directory to the next pipeline step (diversity_profiler.py).
    if ok == 0:
        sys.exit(
            "\n[ERROR] No sequences were successfully processed.\n"
            f"  {total_skipped} sequence(s) were too short "
            f"(minimum {MIN_SEQ_LENGTH} residues required for fold-switch "
            f"prediction), and {errored} failed with an error.\n"
            "Nothing to hand off to diversity_profiler.py — aborting."
        )


if __name__ == "__main__":
    main()
