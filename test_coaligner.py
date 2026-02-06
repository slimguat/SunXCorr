#!/usr/bin/env python3
"""Smoke-test runner for the Coaligner workflow.

This script mirrors the data-loading steps showcased in tutorial.ipynb and
executes the binned cross-correlation search end-to-end. Run it from the
Cross_correlation directory so the default relative FITS paths resolve.
"""
from __future__ import annotations
from typing import Any, Mapping, cast

import argparse
from pathlib import Path
import sys
import os

import numpy as np
import astropy.units as u
from sunpy.map import Map, GenericMap

from coaligner import Coaligner
from help_funcs import get_EUI_paths

DEFAULT_SYNTHETIC_REFERENCE_DIR = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Coaligner demo search.")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("fits_files/SPICE_706_2024-10-17T000050.215.fits"),
        help="Path to the SPICE raster (default mirrors tutorial.ipynb).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("fits_files/FSI_174_2024-10-17T014055.208.fits"),
        help="Path to the FSI reference map (default mirrors tutorial.ipynb).",
    )
    parser.add_argument(
        "--extend-arcsec",
        type=float,
        default=700.0,
        help="Half-width of the padding added around the target FOV (arcsec).",
    )
    parser.add_argument(
        "--bin-kernel-arcsec",
        type=float,
        default=50.0,
        help="Smoothing kernel size applied before binning (arcsec).",
    )
    parser.add_argument(
        "--shift-x",
        type=int,
        default=20,
        help="Maximum absolute pixel shift to explore along X (binned grid).",
    )
    parser.add_argument(
        "--shift-y",
        type=int,
        default=20,
        help="Maximum absolute pixel shift to explore along Y (binned grid).",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.7,
        help="Correlation threshold that switches the search into plateau mode.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=32,
        help="Number of neighbor shifts sampled per iteration.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of worker processes (defaults to os.cpu_count()-1).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=4,
        help="Verbosity level passed to Coaligner.",
    )
    parser.add_argument(
        "--phase",
        choices=("global", "binning", "one-map", "synthetic", "synthetic-raster"),
        default="global",
        help=(
            "Which portion of the workflow to execute. Use 'synthetic' or 'synthetic-raster' "
            "to run only the synthetic raster alignment."
        ),
    )
    parser.add_argument(
        "--seed-dx",
        type=float,
        default=None,
        help="Coarse best-shift dx (binned pixels) for one-map-only runs.",
    )
    parser.add_argument(
        "--seed-dy",
        type=float,
        default=None,
        help="Coarse best-shift dy (binned pixels) for one-map-only runs.",
    )
    parser.add_argument(
        "--one-map-max-corr",
        type=float,
        default=-1.0,
        help="Target correlation threshold for the one-map phase (set <0 to force plateau mode).",
    )
    parser.add_argument(
        "--one-map-neighbors",
        type=int,
        default=64,
        help="Neighbor budget per iteration for the one-map phase.",
    )
    parser.add_argument(
        "--synthetic-reference-dir",
        "--synthetic-eui-dir",
        dest="synthetic_reference_dir",
        type=Path,
        default=None,
        help="Directory scanned when auto-searching for FSI rasters (defaults to CWD).",
    )
    parser.add_argument(
        "--synthetic-reference-keyword",
        type=str,
        default=None,
        help="Filename token required when auto-finding reference rasters (defaults to 'fsi174').",
    )
    parser.add_argument(
        "--synthetic-reference-exclude",
        metavar="TOKEN",
        action="append",
        default=None,
        help="Filename token that excludes a raster when auto-searching (repeatable).",
    )
    
    parser.add_argument(
        "--synthetic-scale-range",
        type=float,
        nargs=2,
        metavar=("SX", "SY"),
        default=None,
        help="Optional multiplicative scale bounds (e.g. 0.7 1.3) overriding synthetic scale_range.",
    )
    parser.add_argument(
        "--synthetic-scale-step",
        type=float,
        nargs=2,
        metavar=("STEP_X", "STEP_Y"),
        default=None,
        help="Optional scale increments applied per index step along X/Y.",
    )
    return parser.parse_args()


def _validate_path(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    return path


def _extract_meta_datetime(meta: Mapping[str, Any], keys: tuple[str, ...]) -> np.datetime64 | None:
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        try:
            return np.datetime64(value)
        except Exception:
            try:
                return np.datetime64(str(value))
            except Exception:
                continue
    return None

# import _vprint
from help_funcs import _vprint
def _auto_reference_sources(
    target_map: GenericMap,
    args: argparse.Namespace,
    reference_path: Path,
) -> list[Path]:
    meta = cast(Mapping[str, Any], target_map.meta or {})
    date_beg = _extract_meta_datetime(meta, ("DATE-BEG", "DATE_BEG", "DATE-OBS", "DATE_OBS"))
    date_end = _extract_meta_datetime(meta, ("DATE-END", "DATE_END", "DATE-OBS", "DATE_OBS"))
    date_avg = _extract_meta_datetime(meta, ("DATE-AVG", "DATE_AVG", "DATE-OBS", "DATE_OBS"))
    if date_avg is None:
        date_avg = np.datetime64(target_map.date.isot)
    if date_beg is None or date_end is None:
        half_day = np.timedelta64(12, "h")
        date_beg = date_avg - half_day
        date_end = date_avg + half_day
    if date_beg > date_end:
        date_beg, date_end = date_end, date_beg

    local_dir = args.synthetic_reference_dir
    if local_dir is None and DEFAULT_SYNTHETIC_REFERENCE_DIR.is_dir():
        local_dir = DEFAULT_SYNTHETIC_REFERENCE_DIR
    if local_dir is None:
        local_dir = reference_path.parent
    verbose = max(0, int(args.verbose))
    candidate_paths = get_EUI_paths(date_beg, date_end, local_dir=local_dir, verbose=verbose)

    channel_token = (args.synthetic_reference_keyword or "fsi174").lower()
    exclude_tokens = [tok.lower() for tok in (args.synthetic_reference_exclude or []) if tok]
    if "short" not in exclude_tokens:
        exclude_tokens.append("short")

    filtered: list[Path] = []
    for path in candidate_paths:
        name = path.name.lower()
        if channel_token and channel_token not in name:
            continue
        if any(token in name for token in exclude_tokens if token):
            continue
        filtered.append(path)

    return filtered or candidate_paths


def main() -> int:
    args = _parse_args()
    target_path = _validate_path(args.target, "target")
    reference_path = _validate_path(args.reference, "reference")
    target_map: GenericMap = cast(GenericMap,Map(target_path))
    reference_map: GenericMap = cast(GenericMap, Map(reference_path))

    resolved_sources = _auto_reference_sources(target_map, args, reference_path)
    if not resolved_sources:
        resolved_sources = [reference_path]

    coaligner = Coaligner(
        target_map,
        reference_map,
        list_of_reference_maps=resolved_sources,
        verbose=args.verbose,
        n_jobs=(args.jobs or max(1, (os.cpu_count() or 2) - 1)),
        n_neighbors=args.neighbors,
    )
    coaligner.target_FOV_expension_arcsec = args.extend_arcsec * u.arcsec
    coaligner.bin_kernel_arcsec = args.bin_kernel_arcsec * u.arcsec
    coaligner.xcorr_binned_kwargs["shift_range"] = (args.shift_x, args.shift_y)
    coaligner.xcorr_binned_kwargs["max_corr"] = args.max_corr
    coaligner.xcorr_one_map_kwargs["max_corr"] = args.one_map_max_corr
    if args.synthetic_reference_dir is not None:
        coaligner.synthetic_kwargs["reference_local_dir"] = args.synthetic_reference_dir
    elif DEFAULT_SYNTHETIC_REFERENCE_DIR.is_dir():
        coaligner.synthetic_kwargs["reference_local_dir"] = DEFAULT_SYNTHETIC_REFERENCE_DIR
    if args.synthetic_reference_keyword:
        coaligner.synthetic_kwargs["reference_channel_keyword"] = args.synthetic_reference_keyword
    if args.synthetic_reference_exclude:
        coaligner.synthetic_kwargs["reference_exclude_tokens"] = tuple(args.synthetic_reference_exclude)
    if args.synthetic_scale_range is not None:
        coaligner.synthetic_kwargs["scale_range"] = tuple(float(val) for val in args.synthetic_scale_range)
    if args.synthetic_scale_step is not None:
        coaligner.synthetic_kwargs["scale_step_x"] = float(args.synthetic_scale_step[0])
        coaligner.synthetic_kwargs["scale_step_y"] = float(args.synthetic_scale_step[1])

    try:
        ran_phases: list[str] = []
        synthetic_only = args.phase in {"synthetic", "synthetic-raster"}

        if args.phase == "global":
            coaligner.run_global_xcorr()
            ran_phases = ["binning", "one_map"]
            if coaligner.procedures.get("synthetic_raster"):
                ran_phases.append("synthetic_raster")
        elif args.phase == "binning":
            coaligner.run_binned_xcorr()
            ran_phases = ["binning"]
        else:
            if args.phase == "one-map":
                if args.seed_dx is None or args.seed_dy is None:
                    raise ValueError("One-map mode requires both --seed-dx and --seed-dy.")
                coaligner.run_one_map_xcorr(seed_shift=(args.seed_dx, args.seed_dy))
                ran_phases = ["one_map"]
            elif synthetic_only:
                coaligner.run_synthetic_raster_xcorr()
                ran_phases = ["synthetic_raster"]
            else:
                raise ValueError(f"Unhandled phase option: {args.phase}")

        label_map = {"binning": "Binning", "one_map": "One-map", "synthetic_raster": "Synthetic"}
        exit_code = 0
        for phase_key in ran_phases:
            result = coaligner.results.get(phase_key) or {}
            best = result.get("best") or {}
            if not best:
                print(f"[{label_map[phase_key]}] Phase did not return a best shift.")
                exit_code = 1
                continue
            corr_val = best.get("corr", float("nan"))
            dx_val = best.get("dx", float("nan"))
            dy_val = best.get("dy", float("nan"))
            sx_val = best.get("sx", best.get("squeeze_x"))
            sy_val = best.get("sy", best.get("squeeze_y"))
            scale_fragment = ""
            if sx_val is not None or sy_val is not None:
                scale_fragment = (
                    f", sx={float(sx_val):.4f}" if sx_val is not None else ""
                ) + (
                    f", sy={float(sy_val):.4f}" if sy_val is not None else ""
                )
            print(
                f"[{label_map[phase_key]}] corr={corr_val:.5f} at dx={dx_val:.2f} px, dy={dy_val:.2f} px{scale_fragment}",
            )
        return exit_code
    finally:
        coaligner.close()


if __name__ == "__main__":
     sys.exit(main())
