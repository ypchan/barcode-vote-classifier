#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barcode_vote: Barcode reads/contigs classifier using minimap2 (PAF) + voting.

Subcommands
  - download : Download barcode_ref.mmi into cache (or --dir) and VERIFY sha256 (required).
  - config   : Show or set config.ini (reference path).
  - show-ref : Print the resolved reference path (cli/env/config/cache) and its source.
  - classify : Run minimap2 -> filter (identity & coverage) -> vote classification.

Default cache location
  - Prefer:  $TMPDIR/barcode-vote-classifier/barcode_ref.mmi  (HPC-friendly)
  - Fallback:~/.cache/barcode-vote-classifier/barcode_ref.mmi

Reference resolution order
  1) --ref (classify/show-ref)
  2) ENV: BARCODE_REF_MMI
  3) Config: ~/.config/barcode-vote-classifier/config.ini
  4) Cache:  (see above)

USAGE EXAMPLES
  1) Download and configure reference once:
      barcode_vote download --write-config

  2) Classify reads (FASTQ; gz supported):
      barcode_vote classify \
        -q fastp/SAMPLE.fastq.gz \
        -o out/SAMPLE \
        -t 56 \
        --sep "|" \
        --id-thr 0.5 --cov-thr 0.5 \
        -N 10 --secondary no \
        --save-filtered-hits

  3) Classify contigs (FASTA; gz supported):
      barcode_vote classify -q contigs.fasta -o out/contigs -t 28

  4) Use a shared reference (override cache):
      export BARCODE_REF_MMI=/shared/db/barcode_ref.mmi
      barcode_vote classify -q reads.fq.gz -o out/sample

  5) Check which reference will be used:
      barcode_vote show-ref -v
"""

from __future__ import annotations

import collections
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import DefaultDict, Optional, Tuple, TextIO

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.text import Text


# -----------------------------
# App / Constants
# -----------------------------
APP_NAME = "barcode-vote-classifier"
ENV_REF = "BARCODE_REF_MMI"
ENV_URL = "BARCODE_REF_URL"
ENV_SHA256 = "BARCODE_REF_SHA256"

DEFAULT_REF_URL = "https://github.com/ypchan/barcode-vote-classifier/releases/download/v0.1.0/barcode_ref.mmi"
DEFAULT_REF_SHA256 = "d97974d1e871875f449423ddf1b40ecab801df1e24c6f7e5f0af52dcc56e0087"

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


# -----------------------------
# Config / Cache
# -----------------------------
def config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / APP_NAME
    return Path.home() / ".config" / APP_NAME


def config_file() -> Path:
    return config_dir() / "config.ini"


def cache_dir(prefer_tmp: bool = True) -> Path:
    # HPC-friendly: prefer TMPDIR
    if prefer_tmp:
        tmp = os.environ.get("TMPDIR")
        if tmp:
            return Path(tmp) / APP_NAME

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / APP_NAME

    return Path.home() / ".cache" / APP_NAME


def read_config_ref() -> Optional[str]:
    cfg = config_file()
    if not cfg.exists():
        return None
    for line in cfg.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ref_mmi="):
            val = line.split("=", 1)[1].strip()
            return val or None
    return None


def write_config_ref(path: str) -> None:
    config_dir().mkdir(parents=True, exist_ok=True)
    cfg = config_file()
    cfg.write_text(f"ref_mmi={path}\n")
    console.print(f"[bold green]OK[/bold green] Wrote config: {cfg}")


# -----------------------------
# sha256 / Download
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sha256(s: str) -> str:
    """
    Validate sha256 string.
    Accepts:
      - 64-hex string
      - 'sha256:<64-hex>' (common copy/paste format)
    Returns normalized lowercase 64-hex.
    """
    s = s.strip().lower()
    if s.startswith("sha256:"):
        s = s.split(":", 1)[1].strip()

    if len(s) != 64 or any(ch not in "0123456789abcdef" for ch in s):
        raise typer.BadParameter("Invalid sha256. Expect 64 hex characters (optionally prefixed by 'sha256:').")
    return s


def download_with_progress(url: str, dest: Path, expected_sha256: str) -> None:
    """
    Download a file with a progress bar and verify sha256.
    On sha256 mismatch, removes temporary file and raises.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    console.print(f"[yellow]INFO[/yellow] Downloading: {url}")
    console.print(f"[yellow]INFO[/yellow] To: {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": "barcode_vote/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        task_id = progress.add_task("Downloading", total=total_size)

        h = hashlib.sha256()
        with progress:
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    progress.update(task_id, advance=len(chunk))

        got = h.hexdigest().lower()

    if got != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch: expected={expected_sha256} got={got}")

    tmp.replace(dest)
    console.print(f"[bold green]OK[/bold green] Downloaded & verified: {dest}")


def resolve_ref_path(cli_ref: Optional[str], prefer_tmp_cache: bool = True) -> Tuple[str, str]:
    """
    Resolve reference path with precedence:
      1) CLI --ref
      2) ENV BARCODE_REF_MMI
      3) config.ini
      4) cache
    Returns (path, source_label).
    """
    if cli_ref:
        p = Path(cli_ref)
        if not p.exists():
            raise FileNotFoundError(f"--ref not found: {cli_ref}")
        return str(p), "cli"

    env_ref = os.environ.get(ENV_REF)
    if env_ref:
        p = Path(env_ref)
        if not p.exists():
            raise FileNotFoundError(f"{ENV_REF} not found: {env_ref}")
        return str(p), "env"

    cfg_ref = read_config_ref()
    if cfg_ref:
        p = Path(cfg_ref)
        if p.exists():
            return str(p), "config"

    cached = cache_dir(prefer_tmp=prefer_tmp_cache) / "barcode_ref.mmi"
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached), "cache"

    raise FileNotFoundError(
        "Reference .mmi not found.\n"
        "Run: barcode_vote download --url <URL> --sha256 <SHA256> --write-config\n"
        f"Or set ENV {ENV_REF}=/path/to/barcode_ref.mmi\n"
        "Or pass --ref /path/to/barcode_ref.mmi"
    )


# -----------------------------
# minimap2 + vote classification
# -----------------------------
def require_exe(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' not found in PATH. Please install it or load a module that provides it.")


def parse_paf_line(line: str):
    """
    Parse minimap2 PAF.
    Required fields:
      1  qname
      2  qlen
      6  tname
      7  tlen
      10 nmatch
      11 alnlen
      12 mapq
    """
    parts = line.split("\t")
    if len(parts) < 12:
        return None
    try:
        qname = parts[0]
        qlen = int(parts[1])
        tname = parts[5]
        tlen = int(parts[6])
        nmatch = int(parts[9])
        alnlen = int(parts[10])
        mapq = int(parts[11])
        return qname, tname, qlen, tlen, nmatch, alnlen, mapq
    except ValueError:
        return None


def category_from_target(tname: str, sep: str) -> str:
    """Extract the major category prefix from the target ID."""
    return tname.split(sep)[0] if sep in tname else "unclassified"


# -----------------------------
# Commands
# -----------------------------
@app.command("download")
def cmd_download(
    url: str = typer.Option(
        None,
        help="Download URL for barcode_ref.mmi (or set BARCODE_REF_URL).",
        show_default=False,
    ),
    sha256: str = typer.Option(
        None,
        help="Expected sha256 for verification (or set BARCODE_REF_SHA256). Accepts 'sha256:<hash>'.",
        show_default=False,
    ),
    dir: str = typer.Option(
        "",
        help="Directory to store reference (default: cache dir).",
        show_default=True,
    ),
    force: bool = typer.Option(
        False,
        help="Force re-download even if file exists.",
        show_default=True,
    ),
    write_config: bool = typer.Option(
        False,
        "--write-config",
        help="Write downloaded path into config.ini.",
        show_default=True,
    ),
    no_tmp_cache: bool = typer.Option(
        False,
        "--no-tmp-cache",
        help="Do not use TMPDIR as cache location.",
        show_default=True,
    ),
):
    """
    Download barcode_ref.mmi into cache (or --dir) and VERIFY sha256.

    Example:
      barcode_vote download --write-config
    """
    u = url or os.environ.get(ENV_URL) or DEFAULT_REF_URL
    if not u:
        raise typer.BadParameter("No URL provided. Use --url, set BARCODE_REF_URL, or set DEFAULT_REF_URL.")

    s = sha256 or os.environ.get(ENV_SHA256) or DEFAULT_REF_SHA256
    if not s:
        raise typer.BadParameter(
            "sha256 is REQUIRED for download. Provide --sha256, set BARCODE_REF_SHA256, or set DEFAULT_REF_SHA256."
        )
    s = normalize_sha256(s)

    target_dir = Path(dir) if dir else cache_dir(prefer_tmp=not no_tmp_cache)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / "barcode_ref.mmi"

    if dest.exists() and dest.stat().st_size > 0 and not force:
        got = sha256_file(dest).lower()
        if got == s:
            console.print(f"[bold green]OK[/bold green] Already downloaded and verified: {dest}")
            if write_config:
                write_config_ref(str(dest))
            return
        console.print("[yellow]WARN[/yellow] Existing file sha256 mismatch; re-downloading.")

    download_with_progress(u, dest, expected_sha256=s)

    if write_config:
        write_config_ref(str(dest))


@app.command("config")
def cmd_config(
    set_ref: str = typer.Option(
        "",
        "--set-ref",
        help="Set ref_mmi path into config.ini.",
        show_default=True,
    )
):
    """
    Show or set config.ini (reference path).

    Examples:
      barcode_vote config
      barcode_vote config --set-ref /path/to/barcode_ref.mmi
    """
    if set_ref:
        p = Path(set_ref)
        if not p.exists():
            raise typer.BadParameter(f"--set-ref not found: {p}")
        write_config_ref(str(p))
        return

    cfg = config_file()
    if cfg.exists():
        console.print(cfg.read_text().strip())
    else:
        console.print("# no config file")


@app.command("show-ref")
def cmd_show_ref(
    ref: str = typer.Option(
        "",
        "--ref",
        help="Explicit ref path override (highest priority).",
        show_default=True,
    ),
    no_tmp_cache: bool = typer.Option(
        False,
        "--no-tmp-cache",
        help="Do not use TMPDIR as cache location.",
        show_default=True,
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Print additional info to stderr.",
        show_default=True,
    ),
):
    """
    Print resolved reference path and its source (cli/env/config/cache).

    Example:
      barcode_vote show-ref -v
    """
    p, src = resolve_ref_path(ref if ref else None, prefer_tmp_cache=not no_tmp_cache)
    # Print path to stdout for scripting
    print(p)
    if verbose:
        console.print(f"[yellow]INFO[/yellow] source={src}", file=sys.stderr)


@app.command("classify")
def cmd_classify(
    query: str = typer.Option(..., "-q", "--query", help="Reads/contigs file (FASTQ/FASTA; .gz supported).", show_default=False),
    out_prefix: str = typer.Option(..., "-o", "--out-prefix", help="Output prefix (writes .final_results.tsv).", show_default=False),
    ref: str = typer.Option("", "--ref", help="Reference .mmi path override (highest priority).", show_default=True),
    threads: int = typer.Option(28, "-t", "--threads", help="Threads for minimap2.", show_default=True),
    max_hits: int = typer.Option(10, "-N", "--max-hits", help="minimap2 -N (max hits per query).", show_default=True),
    secondary: str = typer.Option("no", "--secondary", help="Allow secondary alignments (yes/no).", show_default=True),
    preset: str = typer.Option("map-hifi", "--preset", help="minimap2 preset (-x).", show_default=True),
    id_thr: float = typer.Option(0.5, "--id-thr", help="Identity threshold: nmatch/alnlen > id_thr.", show_default=True),
    cov_thr: float = typer.Option(0.5, "--cov-thr", help="Coverage threshold: alnlen/tlen > cov_thr.", show_default=True),
    min_mapq: int = typer.Option(0, "--min-mapq", help="Keep hits only if mapq >= min_mapq.", show_default=True),
    sep: str = typer.Option("|", "--sep", help="Separator splitting Category from Accession in TargetID.", show_default=True),
    save_filtered_hits: bool = typer.Option(False, "--save-filtered-hits", help="Save <prefix>.filtered_hits.tsv.", show_default=True),
    no_tmp_cache: bool = typer.Option(False, "--no-tmp-cache", help="Do not use TMPDIR as cache location.", show_default=True),
):
    """
    Run minimap2 -> filter (identity+coverage) -> vote classify.

    Example:
      barcode_vote classify -q reads.fq.gz -o out/sample -t 56 --sep "|" --id-thr 0.5 --cov-thr 0.5 -N 10 --secondary no
    """
    require_exe("minimap2")

    ref_path, ref_src = resolve_ref_path(ref if ref else None, prefer_tmp_cache=not no_tmp_cache)
    console.print(f"[yellow]INFO[/yellow] Using ref: {ref_path} (source={ref_src})")

    q = Path(query)
    if not q.exists():
        raise typer.BadParameter(f"Query not found: {q}")

    out_prefix_p = Path(out_prefix)
    out_prefix_p.parent.mkdir(parents=True, exist_ok=True)

    out_final = str(out_prefix_p) + ".final_results.tsv"
    out_hits = str(out_prefix_p) + ".filtered_hits.tsv"

    vote_bins: DefaultDict[str, DefaultDict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    hits_fh: Optional[TextIO] = None
    if save_filtered_hits:
        hits_fh = open(out_hits, "w", encoding="utf-8")
        hits_fh.write("read_id\ttarget_id\tscore\n")

    cmd = ["minimap2", "-x", preset, "-t", str(threads), "-N", str(max_hits)]
    if secondary == "no":
        cmd.append("--secondary=no")
    cmd += [ref_path, str(q)]

    console.print(f"[yellow]INFO[/yellow] Running minimap2: {' '.join(cmd)}")

    # Stream stdout for parsing; let stderr pass through to terminal for real-time debugging.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)

    total_paf = 0
    kept_hits = 0

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        total_paf += 1
        rec = parse_paf_line(line)
        if rec is None:
            continue

        qname, tname, _qlen, tlen, nmatch, alnlen, mapq = rec
        if alnlen <= 0 or tlen <= 0:
            continue
        if mapq < min_mapq:
            continue

        identity = nmatch / alnlen
        coverage = alnlen / tlen

        if identity > id_thr and coverage > cov_thr:
            kept_hits += 1
            cat = category_from_target(tname, sep)
            vote_bins[qname][cat] += nmatch
            if hits_fh:
                hits_fh.write(f"{qname}\t{tname}\t{nmatch}\n")

    rc = proc.wait()
    if hits_fh:
        hits_fh.close()

    if rc != 0:
        raise RuntimeError(f"minimap2 failed with exit code {rc}")

    stats = collections.defaultdict(int)
    with open(out_final, "w", encoding="utf-8") as out:
        out.write("read_id\tfinal_category\tvote_score\n")
        for rid in sorted(vote_bins.keys()):
            cats = vote_bins[rid]
            winner = max(cats, key=cats.get)
            score = cats[winner]
            out.write(f"{rid}\t{winner}\t{score}\n")
            stats[winner] += 1

    # Summary
    console.print("")
    console.print("[bold]SUMMARY[/bold]")
    console.print(f"paf_lines={total_paf} kept_hits={kept_hits} unique_ids={len(vote_bins)}")

    for cat in sorted(stats.keys()):
        console.print(f"  {cat}: {stats[cat]}")

    console.print(f"[bold green]OK[/bold green] Final results: {out_final}")
    if save_filtered_hits:
        console.print(f"[bold green]OK[/bold green] Filtered hits:  {out_hits}")


def main():
    app()


if __name__ == "__main__":
    main()
