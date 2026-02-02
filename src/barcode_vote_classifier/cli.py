__version__ = "0.1.0"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barcode_vote_classify.py

One-shot pipeline:
  minimap2 (PAF) -> filter by identity & coverage -> vote-classify by category prefix.

Designed for barcode reference indexing like:
  TargetID format: "Category|Accession"   (default sep="|")
Example:
  Bacteria|BC00123
  Archaea|BC00999

Filtering logic (same as your awk):
  identity = nmatch / alnlen
  coverage = alnlen / tlen
Keep hit if identity > ID_THR and coverage > COV_THR
Vote weight = nmatch

Outputs:
  1) final results: read_id/contig_id   final_category   vote_score
  2) optional filtered hits tsv: ReadID TargetID Score(nmatch)

Requirements:
  - minimap2 in PATH
  - python3

Usage examples:
  # classify reads
  ./barcode_vote_classify.py \
    -q fastp/ACC.q20_90.q30_80.fastq.gz \
    -r /user-storage/home/.../barcode_reference/barcode_ref.mmi \
    -o out/ACC \
    -t 112

  # classify contigs
  ./barcode_vote_classify.py \
    -q contigs.fasta \
    -r /user-storage/home/.../barcode_reference/barcode_ref.mmi \
    -o out/contigs \
    -t 56

  # custom separator
  ./barcode_vote_classify.py -q reads.fq.gz -r barcode_ref.mmi -o out/prefix --sep "|"

Notes:
  - This script uses minimap2 PAF output fields:
    qname qlen qstart qend strand tname tlen tstart tend nmatch alnlen mapq ...
"""

import argparse
import collections
import os
import shutil
import subprocess
import sys
from typing import Dict, DefaultDict, Tuple, Optional, TextIO


def parse_args():
    p = argparse.ArgumentParser(
        description="Barcode classification by minimap2 + vote logic (one script)."
    )
    p.add_argument("-q", "--query", required=True,
                   help="Query reads/contigs file (fastq/fasta; .gz ok).")
    p.add_argument("-r", "--ref_mmi", required=True,
                   help="Minimap2 index (.mmi), e.g. barcode_reference/barcode_ref.mmi")
    p.add_argument("-o", "--out_prefix", required=True,
                   help="Output prefix. Will write <prefix>.final_results.tsv and optionally <prefix>.filtered_hits.tsv")

    p.add_argument("-t", "--threads", type=int, default=28,
                   help="Threads for minimap2 (default: 28)")
    p.add_argument("-N", "--max_hits", type=int, default=10,
                   help="minimap2 -N (max target hits per query; default: 10)")
    p.add_argument("--secondary", choices=["no", "yes"], default="no",
                   help="Whether to allow secondary alignments (default: no -> --secondary=no)")

    p.add_argument("--id_thr", type=float, default=0.5,
                   help="Identity threshold: nmatch/alnlen > id_thr (default: 0.5)")
    p.add_argument("--cov_thr", type=float, default=0.5,
                   help="Reference coverage threshold: alnlen/tlen > cov_thr (default: 0.5)")

    p.add_argument("--sep", default="|",
                   help="Separator splitting Category from Accession in TargetID (default: '|')")

    p.add_argument("--save_filtered_hits", action="store_true",
                   help="If set, save <prefix>.filtered_hits.tsv (can be very large).")

    p.add_argument("--min_mapq", type=int, default=0,
                   help="Optional MAPQ filter: keep hit only if mapq >= min_mapq (default: 0)")

    return p.parse_args()


def require_exe(name: str):
    if shutil.which(name) is None:
        print(f"[ERROR] '{name}' not found in PATH.", file=sys.stderr)
        sys.exit(2)


def run_minimap2_paf(query: str, ref_mmi: str, threads: int, max_hits: int, secondary: str) -> subprocess.Popen:
    # minimap2 PAF output on stdout
    cmd = [
        "minimap2",
        "-x", "map-hifi",
        "-t", str(threads),
        "-N", str(max_hits),
    ]
    if secondary == "no":
        cmd.append("--secondary=no")
    cmd.extend([ref_mmi, query])

    print("[INFO] Running:", " ".join(cmd), file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)


def paf_iter(p: subprocess.Popen):
    assert p.stdout is not None
    for line in p.stdout:
        if not line.strip():
            continue
        yield line.rstrip("\n")


def parse_paf_line(line: str) -> Optional[Tuple[str, str, int, int, int, int, int]]:
    """
    Return:
      qname, tname, tlen, nmatch, alnlen, mapq, (qlen not used)
    PAF fields:
      1 qname
      2 qlen
      6 tname
      7 tlen
      10 nmatch
      11 alnlen
      12 mapq
    """
    parts = line.split("\t")
    if len(parts) < 12:
        return None
    try:
        qname = parts[0]
        qlen = int(parts[1])  # not used, but keep for debugging if needed
        tname = parts[5]
        tlen = int(parts[6])
        nmatch = int(parts[9])
        alnlen = int(parts[10])
        mapq = int(parts[11])
        return qname, tname, tlen, nmatch, alnlen, mapq, qlen
    except ValueError:
        return None


def category_from_target(tname: str, sep: str) -> str:
    return tname.split(sep)[0] if sep in tname else "unclassified"


def main():
    args = parse_args()
    require_exe("minimap2")

    if not os.path.exists(args.query):
        print(f"[ERROR] query not found: {args.query}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ref_mmi):
        print(f"[ERROR] ref_mmi not found: {args.ref_mmi}", file=sys.stderr)
        sys.exit(1)

    out_final = f"{args.out_prefix}.final_results.tsv"
    out_hits = f"{args.out_prefix}.filtered_hits.tsv"

    vote_bins: DefaultDict[str, DefaultDict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    stats_hits = {
        "total_paf": 0,
        "kept": 0,
        "filtered_short": 0,
        "filtered_thr": 0,
        "filtered_mapq": 0,
    }

    hits_fh: Optional[TextIO] = None
    if args.save_filtered_hits:
        os.makedirs(os.path.dirname(out_hits) or ".", exist_ok=True)
        hits_fh = open(out_hits, "w")
        hits_fh.write("read_id\ttarget_id\tscore\n")

    proc = run_minimap2_paf(args.query, args.ref_mmi, args.threads, args.max_hits, args.secondary)

    # stream parse
    for line in paf_iter(proc):
        stats_hits["total_paf"] += 1
        rec = parse_paf_line(line)
        if rec is None:
            continue
        qname, tname, tlen, nmatch, alnlen, mapq, _qlen = rec

        # protect divide-by-zero
        if alnlen <= 0 or tlen <= 0:
            stats_hits["filtered_short"] += 1
            continue

        if mapq < args.min_mapq:
            stats_hits["filtered_mapq"] += 1
            continue

        identity = nmatch / alnlen
        coverage = alnlen / tlen

        if identity > args.id_thr and coverage > args.cov_thr:
            stats_hits["kept"] += 1
            # vote
            cat = category_from_target(tname, args.sep)
            vote_bins[qname][cat] += nmatch

            if hits_fh is not None:
                hits_fh.write(f"{qname}\t{tname}\t{nmatch}\n")
        else:
            stats_hits["filtered_thr"] += 1

    # collect minimap2 stderr
    assert proc.stderr is not None
    stderr_txt = proc.stderr.read()
    rc = proc.wait()

    if hits_fh is not None:
        hits_fh.close()

    if rc != 0:
        print("[ERROR] minimap2 failed (non-zero exit). stderr follows:", file=sys.stderr)
        print(stderr_txt, file=sys.stderr)
        sys.exit(rc)

    # write final results
    os.makedirs(os.path.dirname(out_final) or ".", exist_ok=True)
    stats_reads = collections.defaultdict(int)

    with open(out_final, "w") as out:
        out.write("read_id\tfinal_category\tvote_score\n")
        for read_id in sorted(vote_bins.keys()):
            cats = vote_bins[read_id]
            winner = max(cats, key=cats.get)
            score = cats[winner]
            out.write(f"{read_id}\t{winner}\t{score}\n")
            stats_reads[winner] += 1

    # summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("[SUMMARY] minimap2 PAF lines:", stats_hits["total_paf"], file=sys.stderr)
    print("[SUMMARY] kept hits:", stats_hits["kept"], file=sys.stderr)
    print("[SUMMARY] filtered by thresholds:", stats_hits["filtered_thr"], file=sys.stderr)
    print("[SUMMARY] filtered by MAPQ:", stats_hits["filtered_mapq"], file=sys.stderr)
    print("[SUMMARY] filtered invalid lengths:", stats_hits["filtered_short"], file=sys.stderr)

    print("\n[SUMMARY] Read/Contig classifications:", file=sys.stderr)
    print(f"{'Category':<25} | {'Count':<10}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    total = 0
    for cat in sorted(stats_reads.keys()):
        print(f"{cat:<25} | {stats_reads[cat]:<10}", file=sys.stderr)
        total += stats_reads[cat]
    print("-" * 40, file=sys.stderr)
    print(f"{'Total Unique IDs':<25} | {total:<10}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    print(f"[INFO] Final results: {out_final}", file=sys.stderr)
    if args.save_filtered_hits:
        print(f"[INFO] Filtered hits:  {out_hits}", file=sys.stderr)


if __name__ == "__main__":
    main()
