"""Command-line entry for `sunxcorr`.

Provides minimal CLI to inspect package metadata and run quick checks.

Examples
--------
>>> from sunxcorr import __version__
>>> isinstance(__version__, str)
True
"""

from __future__ import annotations

import argparse
import sys

from . import __author__, __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sunxcorr", description="SunXCorr utilities")
    p.add_argument(
        "--version", action="store_true", help="print package version and exit"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    parser = _build_parser()
    ns = parser.parse_args(argv)
    if ns.version:
        print(f"sunxcorr {__version__} (author: {__author__})")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
