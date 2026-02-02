#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barcode-vote: Barcode reads/contigs classifier using minimap2 (PAF) + voting.

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
    barcode-vote download \
      --write-config

2) Classify reads (FASTQ; gz supported):
    barcode-vote classify \
      -q fastp/SAMPLE.fastq.gz \
      -o out/SAMPLE \
      -t 56 \
      --sep "|" \
      --id-thr 0.5 --cov-thr 0.5 \
      -N 10 --secondary no \
      --save-filtered-hits

3) Classify contigs (FASTA; gz supported):
    barcode-vote classify -q contigs.fasta -o out/contigs -t 28

4) Use a shared reference (override cache):
    export BARCODE_REF_MMI=/shared/db/barcode_ref.mmi
    barcode-vote classify -q reads.fq.gz -o out/sample

5) Check which reference will be used:
    barcode-vote show-ref -v
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import DefaultDict, Optional, Tuple, TextIO


# -----------------------------
# Constants
# -----------------------------
APP_NAME = "barcode-vote-classifier"
ENV_REF = "BARCODE_REF_MMI"
ENV_URL = "BARCODE_REF_URL"
ENV_SHA256 = "BARCODE_REF_SHA256"

# Optional: hardcode defaults for one-command download experience.
DEFAULT_REF_URL = ""
DEFAULT_REF_SHA256 = ""  # Must be 64 hex chars if set; download enforces sha256.


# -----------------------------
# ANSI colors for CLI help
# -----------------------------
def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() and sys.stderr.isatty()


class C:
    """ANSI color codes (minimal set)."""
    EN = _color_enabled()
    RESET = "\033[0m" if EN else ""
    BOLD = "\033[1m" if EN else ""
    GREEN = "\033[32m" if EN else ""
    CYAN = "\033[36m" if EN else ""
    YELLOW = "\033[33m" if EN else ""
    RED = "\033[31m" if EN else ""


def _ok(s: str) -> str:
    return f"{C.GREEN}{s}{C.RESET}"


def _info(s: str) -> str:
    return f"{C.YELLOW}{s}{C.RESET}"


def _err(s: str) -> str:
    return f"{C.RED}{s}{C.RESET}"


class ColorHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """
    Colorize option strings in help output (e.g. -q/--query) and metavars.
    Defaults are shown automatically via ArgumentDefaultsHelpFormatter.
    """

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            return super()._format_action_invocation(action)

        option_strs = [f"{C.GREEN}{s}{C.RESET}" for s in action.option_strings]
        if action.nargs == 0:
            return ", ".join(option_strs)

        default_metavar = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default_metavar)
        return f"{', '.join(option_strs)} {C.CYAN}{args_string}{C.RESET}"


# -----------------------------
# Config / Cache helpers
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
    print(f"{_ok('[OK]')} Wrote config: {cfg}", file=sys.stderr)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sha256(s: str) -> str:
    s = s.strip().lower()
    if len(s) != 64 or any(ch not in "0123456789abcdef" for ch in s):
        raise RuntimeError("Invalid sha256. Expect exactly 64 hex characters.")
    return s


def download_file(url: str, dest: Path, expected_sha256: str) -> None:
    """
    Download a file and verify sha256. sha256 mismatch raises RuntimeError and leaves no partial file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    print(f"{_info('[INFO]')} Downloading: {url}", file=sys.stderr)
    print(f"{_info('[INFO]')} To: {dest}", file=sys.stderr)

    urllib.request.urlretrieve(url, tmp)

    got = sha256_file(tmp)
    if got.lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch: expected={expected_sha256} got={got}")

    tmp.replace(dest)


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

    msg = (
        "Reference .mmi not found.\n"
        "Run: barcode-vote download --url <URL> --sha256 <SHA256> --write-config\n"
        f"Or set ENV {ENV_REF}=/path/to/barcode_ref.mmi\n"
        "Or pass --ref /path/to/barcode_ref.mmi"
    )
    raise FileNotFoundError(msg)


# -----------------------------
# minimap2 + vote classification
# -----------------------------
def require_exe(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' not found in PATH. Please install it or load a module that provides it.")


def parse_paf_line(line: str):
    """
    Parse minimap2 PAF.
    Expected fields:
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
    """
    Extract the major category prefix from the target ID.
    Example: "Bacteria|ACC001" with sep="|" -> "Bacteria"
    """
    return tname.split(sep)[0] if sep in tname else "unclassified"


def cmd_classify(args: argparse.Namespace) -> None:
    require_exe("minimap2")

    ref_path, ref_src = resolve_ref_path(args.ref, prefer_tmp_cache=not args.no_tmp_cache)
    print(f"{_info('[INFO]')} Using ref: {ref_path} (source={ref_src})", file=sys.stderr)

    q = Path(args.query)
    if not q.exists():
        raise FileNotFoundError(f"Query not found: {q}")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_final = str(out_prefix) + ".final_results.tsv"
    out_hits = str(out_prefix) + ".filtered_hits.tsv"

    vote_bins: DefaultDict[str, DefaultDict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    hits_fh: Optional[TextIO] = None
    if args.save_filtered_hits:
        hits_fh = open(out_hits, "w")
        hits_fh.write("read_id\ttarget_id\tscore\n")

    cmd = [
        "minimap2", "-x", args.preset,
        "-t", str(args.threads),
        "-N", str(args.max_hits),
    ]
    if args.secondary == "no":
        cmd.append("--secondary=no")
    cmd += [ref_path, str(q)]

    print(f"{_info('[INFO]')} Running: {' '.join(cmd)}", file=sys.stderr)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

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
        if mapq < args.min_mapq:
            continue

        identity = nmatch / alnlen
        coverage = alnlen / tlen

        if identity > args.id_thr and coverage > args.cov_thr:
            kept_hits += 1
            cat = category_from_target(tname, args.sep)
            vote_bins[qname][cat] += nmatch
            if hits_fh:
                hits_fh.write(f"{qname}\t{tname}\t{nmatch}\n")

    assert proc.stderr is not None
    stderr = proc.stderr.read()
    rc = proc.wait()

    if hits_fh:
        hits_fh.close()

    if rc != 0:
        print(stderr, file=sys.stderr)
        raise RuntimeError(f"minimap2 failed with exit code {rc}")

    stats = collections.defaultdict(int)
    with open(out_final, "w") as out:
        out.write("read_id\tfinal_category\tvote_score\n")
        for rid in sorted(vote_bins.keys()):
            cats = vote_bins[rid]
            winner = max(cats, key=cats.get)
            score = cats[winner]
            out.write(f"{rid}\t{winner}\t{score}\n")
            stats[winner] += 1

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"{_info('[SUMMARY]')} paf_lines={total_paf} kept_hits={kept_hits} unique_ids={len(vote_bins)}", file=sys.stderr)
    print(f"{'Category':<25} | {'Count':<10}", file=sys.stderr)
    print("-" * 45, file=sys.stderr)
    total = 0
    for k in sorted(stats.keys()):
        print(f"{k:<25} | {stats[k]:<10}", file=sys.stderr)
        total += stats[k]
    print("-" * 45, file=sys.stderr)
    print(f"{'Total':<25} | {total:<10}", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)

    print(f"{_ok('[OK]')} Final results: {out_final}", file=sys.stderr)
    if args.save_filtered_hits:
        print(f"{_ok('[OK]')} Filtered hits:  {out_hits}", file=sys.stderr)


def cmd_download(args: argparse.Namespace) -> None:
    url = args.url or os.environ.get(ENV_URL) or DEFAULT_REF_URL
    if not url:
        raise RuntimeError(
            "No URL provided. Use --url or set BARCODE_REF_URL, or hardcode DEFAULT_REF_URL in the tool."
        )

    sha = args.sha256 or os.environ.get(ENV_SHA256) or DEFAULT_REF_SHA256
    if not sha:
        raise RuntimeError(
            "sha256 is REQUIRED for download.\n"
            "Provide --sha256 <64hex> or set BARCODE_REF_SHA256, or hardcode DEFAULT_REF_SHA256."
        )
    sha = _validate_sha256(sha)

    target_dir = Path(args.dir) if args.dir else cache_dir(prefer_tmp=not args.no_tmp_cache)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / "barcode_ref.mmi"

    if dest.exists() and dest.stat().st_size > 0 and not args.force:
        got = sha256_file(dest)
        if got.lower() == sha:
            print(f"{_ok('[OK]')} Already downloaded and verified: {dest}", file=sys.stderr)
            if args.write_config:
                write_config_ref(str(dest))
            return
        print(f"{_info('[WARN]')} Existing file sha256 mismatch; re-downloading.", file=sys.stderr)

    download_file(url, dest, expected_sha256=sha)
    print(f"{_ok('[OK]')} Downloaded & verified: {dest}", file=sys.stderr)

    if args.write_config:
        write_config_ref(str(dest))


def cmd_show_ref(args: argparse.Namespace) -> None:
    p, src = resolve_ref_path(args.ref, prefer_tmp_cache=not args.no_tmp_cache)
    print(p)
    if args.verbose:
        print(f"# source={src}", file=sys.stderr)


def cmd_config(args: argparse.Namespace) -> None:
    if args.set_ref:
        p = Path(args.set_ref)
        if not p.exists():
            raise FileNotFoundError(f"--set-ref not found: {p}")
        write_config_ref(str(p))
        return

    cfg = config_file()
    if cfg.exists():
        print(cfg.read_text().strip())
    else:
        print("# no config file")


# -----------------------------
# Argument parsing
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    desc = (
        f"{C.BOLD}barcode-vote{C.RESET}: minimap2 + vote classifier with cached reference\n\n"
        f"{C.BOLD}Typical workflow{C.RESET}\n"
        f"  1) Download ref (once) + write config:\n"
        f"     {C.GREEN}barcode-vote download --url <URL> --sha256 <SHA256> --write-config{C.RESET}\n\n"
        f"  2) Classify:\n"
        f"     {C.GREEN}barcode-vote classify -q reads.fq.gz -o out/sample -t 56{C.RESET}\n"
    )

    p = argparse.ArgumentParser(
        prog="barcode-vote",
        description=desc,
        formatter_class=ColorHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # download
    d = sub.add_parser(
        "download",
        help="Download barcode_ref.mmi into cache (or --dir) and VERIFY sha256 (required).",
        formatter_class=ColorHelpFormatter,
    )
    d.add_argument("--url", default="https://github.com/ypchan/barcode-vote-classifier/releases/download/v0.1.0/barcode_ref.mmi", help="Download URL for barcode_ref.mmi (or set BARCODE_REF_URL)")
    d.add_argument("--sha256", default="d97974d1e871875f449423ddf1b40ecab801df1e24c6f7e5f0af52dcc56e0087", help="Expected sha256 for verification (or set BARCODE_REF_SHA256)")
    d.add_argument("--dir", default="", help="Directory to store reference (default: cache dir)")
    d.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    d.add_argument("--write-config", action="store_true", help="Write downloaded path into config.ini")
    d.add_argument("--no-tmp-cache", action="store_true", help="Do not use TMPDIR as cache location")
    d.set_defaults(func=cmd_download)

    # show-ref
    s = sub.add_parser(
        "show-ref",
        help="Print resolved reference path and its source (cli/env/config/cache).",
        formatter_class=ColorHelpFormatter,
    )
    s.add_argument("--ref", default="", help="Explicit ref path override (highest priority)")
    s.add_argument("--no-tmp-cache", action="store_true", help="Do not use TMPDIR as cache location")
    s.add_argument("-v", "--verbose", action="store_true", help="Print additional info to stderr")
    s.set_defaults(func=cmd_show_ref)

    # config
    c = sub.add_parser(
        "config",
        help="Show or set config.ini (reference path).",
        formatter_class=ColorHelpFormatter,
    )
    c.add_argument("--set-ref", default="", help="Set ref_mmi path into config.ini")
    c.set_defaults(func=cmd_config)

    # classify
    k = sub.add_parser(
        "classify",
        help="Run minimap2 -> filter (identity+coverage) -> vote classify.",
        formatter_class=ColorHelpFormatter,
    )
    k.add_argument("-q", "--query", required=True, help="Reads/contigs file (FASTQ/FASTA; .gz supported)")
    k.add_argument("-o", "--out-prefix", required=True, help="Output prefix (writes .final_results.tsv)")
    k.add_argument("--ref", default="", help="Reference .mmi path override (highest priority)")
    k.add_argument("-t", "--threads", type=int, default=28, help="Threads for minimap2")
    k.add_argument("-N", "--max-hits", type=int, default=10, help="minimap2 -N (max hits per query)")
    k.add_argument("--secondary", choices=["no", "yes"], default="no", help="Allow secondary alignments")
    k.add_argument("--preset", default="map-hifi", help="minimap2 preset (-x)")
    k.add_argument("--id-thr", type=float, default=0.5, help="Identity threshold: nmatch/alnlen > id_thr")
    k.add_argument("--cov-thr", type=float, default=0.5, help="Coverage threshold: alnlen/tlen > cov_thr")
    k.add_argument("--min-mapq", type=int, default=0, help="Keep hits only if mapq >= min_mapq")
    k.add_argument("--sep", default="|", help="Separator splitting Category from Accession in TargetID")
    k.add_argument("--save-filtered-hits", action="store_true", help="Save <prefix>.filtered_hits.tsv")
    k.add_argument("--no-tmp-cache", action="store_true", help="Do not use TMPDIR as cache location")
    k.set_defaults(func=cmd_classify)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print(_err("[ERROR] interrupted"), file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(_err(f"[ERROR] {e}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

