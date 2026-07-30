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
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# K-mer length is fixed to match how the SQLite index was built — it isn't
# a runtime choice, so it lives here as a constant rather than a CLI flag.
K_MER_LENGTH = 7


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
    rows = []
    n    = len(sequence)

    for pos in range(n - k + 1):
        kmer     = sequence[pos : pos + k]
        kmer_pos = pos + 1      # 1-based window position along the query

        hits = query_kmer(conn, kmer)

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

    print(f"  {len(sequences)} sequence(s) found.")
    print(f"Database          : {args.db}")
    print(f"K-mer length      : {K_MER_LENGTH}")
    print(f"Output dir        : {out_dir}\n")

    try:
        conn = open_db_readonly(args.db)
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot open database: {exc}")

    skipped = 0
    for header, seq in tqdm(sequences, desc="Querying k-mers"):
        name = safe_name(header)

        if len(seq) < K_MER_LENGTH:
            tqdm.write(f"  [SKIP] '{name}' — shorter than k={K_MER_LENGTH}")
            skipped += 1
            continue

        df = pick_fragments(seq, conn, K_MER_LENGTH)
        df.to_csv(out_dir / f"{name}_hits.csv", index=False)

    conn.close()

    print(f"\nDone.  Results saved to: {out_dir}/")
    if skipped:
        print(f"  ({skipped} sequence(s) skipped — too short for k={K_MER_LENGTH})")


if __name__ == "__main__":
    main()
