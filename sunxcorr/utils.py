"""Utility functions for coalignment processes.

This module is large and legacy; the above per-file mypy relaxations are
temporary to allow incremental typing fixes.

Examples
--------
>>> # Safe import-only example (do not run heavy I/O in doctests)
>>> from sunxcorr import utils
>>> hasattr(utils, '_as_datetime64_ms')
True
"""

from __future__ import annotations

import copy
import datetime
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    Union,
    cast,
)

import astropy
import astropy.units as u
import numpy as np
import pandas as pd
import sunpy
import sunpy.map.maputils
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    SqrtStretch,
    interval,
    stretch,
)
from astropy.wcs import WCS
from matplotlib.colors import Normalize
from typing import Any as _Any

# Use a concrete NDArray[Any] alias at runtime and for type checking.
# Importing and subscripting numpy.typing.NDArray is safe here and avoids
# "Bad number of arguments for type alias" errors when the module is
# analyzed by mypy or executed.
from numpy.typing import NDArray as _NDArray

NDArray = _NDArray[_Any]
from reproject import reproject_interp
from scipy.ndimage import map_coordinates, uniform_filter
from sunpy.map import GenericMap, Map


# Small helper: cast various date-like inputs to numpy.datetime64[ms]
def _as_datetime64_ms(val: _Any) -> Union[np.datetime64, NDArray]:
    """Cast a value to numpy.datetime64 with millisecond precision.

    This is a concise helper used to normalize various inputs (str, datetime,
    np.datetime64) into `numpy.datetime64[ms]` which avoids Pylance overload
    complaints from direct `np.datetime64(..., "ms")` calls.
    """
    return np.asarray(val, dtype="datetime64[ms]")


# import os

WCSLinearMode = Literal["pc2_unit", "cd_basis_unit", "cdelt_invariant", "cd"]


def build_corrected_wcs_meta_scale_shift(
    map_in: GenericMap,
    dx_opt: float = 0.0,
    dy_opt: float = 0.0,
    sx_opt: float = 1.0,
    sy_opt: float = 1.0,
    verbose: int = 0,
    *,
    linear_mode: WCSLinearMode = "cd_basis_unit",
) -> Dict[str, Any]:
    """
    Construct corrected WCS metadata encoding a pixel-domain affine registration
    (anisotropic scaling about CRPIX followed by a translation), without resampling.

        Pixel-domain transform (registration model)
        ------------------------------------------
            P' = P0 + ΔS (P - P0) + Δ

        with:
        - P  = (pix1, pix2)^T
        - P0 = (CRPIX1, CRPIX2)^T
        - ΔS = diag(s_x, s_y)  (forward scale factors in WCS pixel axis order)
        - Δ  = (Δx, Δy)^T      (forward translation in pixels; +x right, +y down)

        Linear WCS model (pre-projection)
        ---------------------------------
            X = X0 + M (P - P0)

        with:
        - X  = (world1, world2)^T  (intermediate world coords before projection)
        - X0 = (CRVAL1, CRVAL2)^T
        - M  = CD (the 2x2 linear Jacobian at CRPIX)
        - P  = (pix1, pix2)^T
        - P0 = (CRPIX1, CRPIX2)^T

        Enforcing invariance of world coordinates
        -----------------------------------------
            WCS_old(P) = WCS_new(P')   for all P

        Choosing P0' = P0 (CRPIX unchanged) yields:
            M'  = M (ΔS)^{-1}
            X0' = X0 - M' Δ

        Optimizer convention
        --------------------
        The optimizer is assumed to return the inverse correction:
        - s_x = 1 / sx_opt,   s_y = 1 / sy_opt
        - Δx  = -dx_opt,      Δy  = -dy_opt

        FITS linear WCS encodings
        -------------------------
        The linear mapping may be represented either as:
        - CD matrix directly:
                CD_ij  (preferred for unambiguous storage)
            or
        - PC + CDELT factorization (FITS row scaling):
                CD = diag(CDELT1, CDELT2) @ PC

            i.e. CDELT scales ROWS, not columns:
                CD1_j = CDELT1 * PC1_j
                CD2_j = CDELT2 * PC2_j

    Parameters
    ----------
    map_in : GenericMap
        Input map whose WCS metadata will be adjusted to encode the provided
        scale and shift corrections.
    dx_opt, dy_opt : float
        Pixel translation corrections (applied after scaling).
    sx_opt, sy_opt : float
        Scale correction factors in pixel units (forward scaling applied to
        pixel coordinates).
    verbose : int
        Verbosity level for debug printing.
    linear_mode : {'pc2_unit', 'cd_basis_unit', 'cdelt_invariant', 'cd'}
        How to store the corrected linear transform in FITS header keys.

    Returns
    -------
    dict
        A dictionary of header-like WCS metadata keys (CRPIX, CRVAL, CD/PC/CDELT)
        representing the corrected WCS without performing any resampling.

        - "pc2_unit" (default) :
            Enforce global (Frobenius) normalization of PC:
                ||PC||_F^2 = sum_{i,j} PC_{i,j}^2 = 1
            while preserving CD' exactly by a single global rescaling:
                PC'    = PC / k
                CDELT' = CDELT * k        (same k applied to both axes)
            so that diag(CDELT') @ PC' == CD'.

        - "cd_basis_unit" :
            Enforce row-wise unit-norm of PC (basis vectors in FITS row-scaled sense):
                ||PC[0,:]||_2 = 1   and   ||PC[1,:]||_2 = 1
            while preserving CD' exactly by per-row rescaling:
                PC'[0,:] = PC[0,:] / r1      CDELT1' = CDELT1 * r1
                PC'[1,:] = PC[1,:] / r2      CDELT2' = CDELT2 * r2
            so that diag(CDELT') @ PC' = CD'.

        Returns
        -------
        dict
            Header-like metadata dict with:
            - CRPIX unchanged
            - updated CRVAL1/2
            - corrected linear mapping stored per `linear_mode`
    """
    meta = cast(Dict[str, Any], copy.deepcopy(map_in.wcs.to_header()))

    crval1 = float(meta["CRVAL1"])
    crval2 = float(meta["CRVAL2"])

    # ----- read current linear matrix as CD -----
    def _get_cd_from_header(h: Dict[str, Any]) -> np.ndarray:
        if all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
            return np.array(
                [
                    [float(h["CD1_1"]), float(h["CD1_2"])],
                    [float(h["CD2_1"]), float(h["CD2_2"])],
                ],
                dtype=float,
            )

        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        pc11 = float(h.get("PC1_1", 1.0))
        pc12 = float(h.get("PC1_2", 0.0))
        pc21 = float(h.get("PC2_1", 0.0))
        pc22 = float(h.get("PC2_2", 1.0))

        return np.array(
            [[cdelt1 * pc11, cdelt1 * pc12], [cdelt2 * pc21, cdelt2 * pc22]],
            dtype=float,
        )

    CD = _get_cd_from_header(meta)

    # ----- optimizer -> forward affine params -----
    if sx_opt == 0.0 or sy_opt == 0.0:
        raise ValueError("sx_opt and sy_opt must be non-zero.")

    s_x = 1.0 / sx_opt
    s_y = 1.0 / sy_opt
    dx = -dx_opt
    dy = -dy_opt

    # ----- corrected linear matrix: CD' = CD @ inv(diag(sx,sy)) (rescales columns) -----
    inv_DS = np.array([[1.0 / s_x, 0.0], [0.0, 1.0 / s_y]], dtype=float)
    CDp = CD @ inv_DS

    # ----- update CRVAL with CRPIX fixed: CRVAL' = CRVAL - CD' @ [dx,dy] -----
    meta["CRVAL1"] = crval1 - (CDp[0, 0] * dx + CDp[0, 1] * dy)
    meta["CRVAL2"] = crval2 - (CDp[1, 0] * dx + CDp[1, 1] * dy)

    # ----- writers -----
    def _write_cd(h: Dict[str, Any], CDm: np.ndarray) -> None:
        h["CD1_1"], h["CD1_2"] = float(CDm[0, 0]), float(CDm[0, 1])
        h["CD2_1"], h["CD2_2"] = float(CDm[1, 0]), float(CDm[1, 1])
        for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
            h.pop(k, None)

    def _write_pc_from_cd_and_cdelt(
        h: Dict[str, Any], CDm: np.ndarray, cdelt1: float, cdelt2: float
    ) -> None:
        if cdelt1 == 0.0 or cdelt2 == 0.0:
            raise ValueError("Degenerate CDELT: cannot form PC.")
        h["CDELT1"] = float(cdelt1)
        h["CDELT2"] = float(cdelt2)

        h["PC1_1"] = float(CDm[0, 0] / cdelt1)
        h["PC1_2"] = float(CDm[0, 1] / cdelt1)
        h["PC2_1"] = float(CDm[1, 0] / cdelt2)
        h["PC2_2"] = float(CDm[1, 1] / cdelt2)

        for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
            h.pop(k, None)

    def _ensure_cdelt_exists(h: Dict[str, Any]) -> None:
        if "CDELT1" not in h or "CDELT2" not in h:
            raise ValueError(
                "CDELT1/2 not present in header; cannot use PC+CDELT modes."
            )

    def _write_pc_cdelt_invariant(h: Dict[str, Any], CDm: np.ndarray) -> None:
        _ensure_cdelt_exists(h)
        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        _write_pc_from_cd_and_cdelt(h, CDm, cdelt1, cdelt2)

    def _write_pc2_unit_frobenius(h: Dict[str, Any], CDm: np.ndarray) -> None:
        """
        pc2_unit: enforce sum(PC^2)=1 (Frobenius norm 1), preserving CD.
        Achieved by global scaling of PC by k and compensating both CDELT by k.
        """
        _ensure_cdelt_exists(h)
        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        if cdelt1 == 0.0 or cdelt2 == 0.0:
            raise ValueError("Degenerate CDELT: cannot form PC.")

        PC = np.array(
            [
                [CDm[0, 0] / cdelt1, CDm[0, 1] / cdelt1],
                [CDm[1, 0] / cdelt2, CDm[1, 1] / cdelt2],
            ],
            dtype=float,
        )

        frob = float(np.sqrt(np.sum(PC * PC)))
        if frob == 0.0:
            raise ValueError("Degenerate PC: Frobenius norm zero.")

        # Scale PC down, scale CDELT up -> CD unchanged
        PCp = PC / frob
        cdelt1p = cdelt1 * frob
        cdelt2p = cdelt2 * frob

        if verbose:
            print(f"pc2_unit: Frobenius(PC)={frob:.6g} -> enforcing ||PC||_F=1")

        _write_pc_from_cd_and_cdelt(h, CDm, cdelt1p, cdelt2p)
        # overwrite PC with the normalized one explicitly (avoids tiny float drift)
        h["PC1_1"], h["PC1_2"] = float(PCp[0, 0]), float(PCp[0, 1])
        h["PC2_1"], h["PC2_2"] = float(PCp[1, 0]), float(PCp[1, 1])

    def _write_cd_basis_unit_rows(h: Dict[str, Any], CDm: np.ndarray) -> None:
        """
        cd_basis_unit: enforce ||PC row1||=1 and ||PC row2||=1, preserving CD.
        Achieved by scaling each PC row i by ki and compensating CDELTi by ki.
        """
        _ensure_cdelt_exists(h)
        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        if cdelt1 == 0.0 or cdelt2 == 0.0:
            raise ValueError("Degenerate CDELT: cannot form PC.")

        PC = np.array(
            [
                [CDm[0, 0] / cdelt1, CDm[0, 1] / cdelt1],
                [CDm[1, 0] / cdelt2, CDm[1, 1] / cdelt2],
            ],
            dtype=float,
        )

        r1 = float(np.linalg.norm(PC[0, :]))
        r2 = float(np.linalg.norm(PC[1, :]))
        if r1 == 0.0 or r2 == 0.0:
            raise ValueError("Degenerate PC: row norm zero.")

        # Normalize rows, scale CDELT per row -> CD unchanged
        PCp = PC.copy()
        PCp[0, :] /= r1
        PCp[1, :] /= r2
        cdelt1p = cdelt1 * r1
        cdelt2p = cdelt2 * r2

        if verbose:
            print(
                f"cd_basis_unit: row norms (r1,r2)=({r1:.6g},{r2:.6g}) -> enforcing both = 1"
            )

        _write_pc_from_cd_and_cdelt(h, CDm, cdelt1p, cdelt2p)
        h["PC1_1"], h["PC1_2"] = float(PCp[0, 0]), float(PCp[0, 1])
        h["PC2_1"], h["PC2_2"] = float(PCp[1, 0]), float(PCp[1, 1])

    # ----- dispatch -----
    if linear_mode == "cd":
        _write_cd(meta, CDp)
    elif sx_opt == 1.0 and sy_opt == 1.0:
        _vprint(
            verbose,
            2,
            "No scaling correction (sx_opt=sy_opt=1.0), preserving original linear WCS.",
        )
        meta["PC1_1"] = meta.get("PC1_1", 1.0)
        meta["PC1_2"] = meta.get("PC1_2", 0.0)
        meta["PC2_1"] = meta.get("PC2_1", 0.0)
        meta["PC2_2"] = meta.get("PC2_2", 1.0)
        meta.pop("CD1_1", None)
        meta.pop("CD1_2", None)
        meta.pop("CD2_1", None)
        meta.pop("CD2_2", None)
        meta["CDELT1"] = meta.get("CDELT1", 1.0)
        meta["CDELT2"] = meta.get("CDELT2", 1.0)
        # No scaling correction, preserve original CD or PC+CDELT as-is
        pass

    elif linear_mode == "cdelt_invariant":
        _write_pc_cdelt_invariant(meta, CDp)

    elif linear_mode == "pc2_unit":
        _write_pc2_unit_frobenius(meta, CDp)

    elif linear_mode == "cd_basis_unit":
        _write_cd_basis_unit_rows(meta, CDp)

    else:
        raise ValueError(
            f"Unknown linear_mode='{linear_mode}'. "
            "Choose from 'pc2_unit', 'cd_basis_unit', 'cdelt_invariant', 'cd'."
        )

    return meta


def make_corrected_wcs_map(
    map_in: GenericMap,
    best_params: Dict[str, float],
    verbose: int = 0,
    linear_mode: WCSLinearMode = "pc2_unit",
) -> GenericMap:
    dx = float(best_params.get("dx", 0.0))
    dy = float(best_params.get("dy", 0.0))
    sx = float(best_params.get("squeeze_x", 1.0))
    sy = float(best_params.get("squeeze_y", 1.0))

    new_meta = build_corrected_wcs_meta_scale_shift(
        map_in,
        dx_opt=dx,
        dy_opt=dy,
        sx_opt=sx,
        sy_opt=sy,
        verbose=verbose,
        linear_mode=linear_mode,
    )

    return cast(
        GenericMap,
        sunpy.map.Map(map_in.data, new_meta, plot_settings=map_in.plot_settings),
    )


def no_nan_uniform_filter(
    data: NDArray,
    remove_percentile: float = 100,
    *args,
    **kwargs,
) -> NDArray:
    """Apply uniform filter while masking outliers and preserving NaN regions.

    Wraps scipy.ndimage.uniform_filter with preprocessing to handle NaN values
    and optionally remove extreme outliers before smoothing.

    Parameters
    ----------
    data : NDArray
        Input array to filter
    remove_percentile : float, default=100
        Percentile threshold for outlier removal (0-100).
        Values above this percentile are masked as NaN before filtering.
        Use 100 to disable outlier removal.
    *args
        Additional positional arguments passed to uniform_filter
        (typically filter size)
    **kwargs
        Additional keyword arguments passed to uniform_filter
        (mode, cval, etc.)

    Returns
    -------
    NDArray
        Smoothed array with original NaN locations preserved

    Notes
    -----
    The function performs these steps:
    1. Identify values above remove_percentile and mark as NaN
    2. Create binary mask of NaN locations
    3. Fill NaN locations with 0.0 for filtering
    4. Apply uniform_filter to filled array
    5. Restore NaN mask in output

    This approach prevents NaN propagation during filtering while preserving
    the locations where data is genuinely missing. Useful for smoothing solar
    images that may contain hot pixels or detector artifacts.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    >>> smoothed = no_nan_uniform_filter(data, size=3)
    >>> smoothed[1, 1]  # Center remains NaN
    nan

    See Also
    --------
    scipy.ndimage.uniform_filter : Underlying filter function
    """
    data_percentile: float = cast(float, np.nanpercentile(data, remove_percentile))
    data_cleaned: NDArray = np.where(data > data_percentile, np.nan, data)
    nan_mask: NDArray = np.isnan(data_cleaned)
    data_filled: NDArray = np.where(nan_mask, 0.0, data_cleaned)
    filtered_data: NDArray = uniform_filter(data_filled, *args, **kwargs)
    filtered_data[nan_mask] = np.nan
    return filtered_data


def normit(
    data: NDArray | None = None,
    interval: interval.BaseInterval | None = AsymmetricPercentileInterval(1, 99),
    stretch: stretch.BaseStretch | None = SqrtStretch(),
    vmin: float | None = None,
    vmax: float | None = None,
    clip: bool = False,
    invalid=-1.0,
) -> Normalize | None:
    """Normalize the data using the specified interval, stretch, vmin, and vmax.

    Args:
        data (numpy.ndarray): The data to be normalized.
        interval (astropy.visualization.Interval, optional): The interval to use for normalization.
            Defaults to AsymmetricPercentileInterval(1, 99).
        stretch (astropy.visualization.Stretch, optional): The stretch to apply to the data.
            Defaults to SqrtStretch().
        vmin (float, optional): The minimum value for normalization. Defaults to None.
        vmax (float, optional): The maximum value for normalization. Defaults to None.

    Returns:
        astropy.visualization.ImageNormalize | None: The normalized data or None if input data is all NaNs.
    """
    if vmin is not None or vmax is not None:
        interval = None
    if stretch is not None:
        if data is None or np.all(np.isnan(data)):
            return None
        return cast(
            Normalize,
            ImageNormalize(
                data,
                interval,
                stretch=stretch,
                vmin=vmin,
                vmax=vmax,
                clip=clip,
                invalid=invalid,
            ),
        )

    return cast(
        Normalize,
        ImageNormalize(
            data, interval, vmin=vmin, vmax=vmax, clip=clip, invalid=invalid
        ),
    )


def get_closest_EUIFSI174_paths(
    date_ref: Union[np.datetime64, NDArray],
    interval: np.timedelta64,
    local_dir: Union[str, Path] = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0,
) -> List[Path]:
    """
    Find the EUI FITS files whose timestamps are closest to a reference date,
    within a specified tolerance interval.

    Parameters
    ----------
    date_ref : np.datetime64
        Reference date/time.
    interval : np.timedelta64
        Maximum allowed offset (±) from date_ref.
    local_dir : str or Path, optional
        Base directory where EUI data is stored.
    verbose : int, optional
        Verbosity level for logging (default 0).

    Returns
    -------
    List[Path]
        Paths to the FITS files whose timestamps are the closest to date_ref,
        among those within [date_ref - interval, date_ref + interval].
        If no files lie within that window, returns an empty list.
    """
    # Normalize to millisecond precision
    date_ref = _as_datetime64_ms(date_ref)
    half_int = np.asarray(interval, dtype="timedelta64[ms]")
    lower = date_ref - half_int
    upper = date_ref + half_int
    assert lower <= upper, "Interval must be non-negative"

    local_dir = Path(local_dir)
    _vprint(verbose, 1, f"Searching EUI paths around {date_ref} ± {interval}")

    # 1) Gather all days that might contain candidates
    days = _find_all_days(lower, upper, verbose)  # type: ignore[arg-type]
    _vprint(verbose, 2, f"Checking {len(days)} days from {lower} to {upper}")

    # 2) Collect all FITS files in that window
    all_paths: List[Path] = []
    for day in days:
        _vprint(verbose, 2, f"  Grabbing data for {day}")
        all_paths.extend(_grab_EUI_data(day, local_dir, verbose))

    if not all_paths:
        _vprint(verbose, 1, "No EUI files found in the interval.")
        return []

    all_paths = list(split_eui_paths_by_mode(all_paths)["eui-fsi174-image"])

    # 3) Extract timestamps from filenames
    def extract_dt(p: Path) -> Union[np.datetime64, None]:
        m = re.search(r"(\d{8})T(\d{6})", p.name)
        if not m:
            return None
        dt = datetime.datetime.strptime(m.group(0), "%Y%m%dT%H%M%S")
        return cast(np.datetime64, _as_datetime64_ms(dt))

    _vprint(verbose, 2, f"Extracting timestamps from {len(all_paths)} files")
    paths_arr = np.array(all_paths, dtype=object)
    dates = np.array([extract_dt(p) for p in paths_arr], dtype="datetime64[ms]")

    # 4) Compute absolute differences to reference, mask out None
    valid_mask = dates != _as_datetime64_ms("NaT")
    if not np.any(valid_mask):
        _vprint(verbose, 1, "No valid timestamps parsed.")
        return []

    diffs = np.abs(dates[valid_mask] - date_ref)
    min_diff = diffs.min()
    _vprint(verbose, 1, f"Minimum time difference = {min_diff}")
    # 5) Select all paths that achieve this minimum difference
    candidates = paths_arr[valid_mask][diffs == min_diff]
    # Sort lexicographically for reproducibility
    closest = sorted(candidates.tolist())

    _vprint(verbose, 1, f"Found {len(closest)} closest file(s) within ±{interval}")
    return closest


# Imager manipulation functions.
def _find_all_days(
    date_min: Union[np.datetime64, NDArray],
    date_max: Union[np.datetime64, NDArray],
    verbose: int = 0,
) -> List[np.datetime64]:
    """
    Find all dates between date_min and date_max (inclusive).

    Parameters
    ----------
    date_min : np.datetime64
        The start date.
    date_max : np.datetime64
        The end date.
    verbose : int, optional
        Verbosity level for logging (default is 0).

    Returns
    -------
    List[np.datetime64]
        A list of np.datetime64 objects for each day in the range.
    """
    _vprint(verbose, 2, f"Finding all days between {date_min} and {date_max}")
    # Add one day to include date_max in the range
    end_plus_one = date_max + np.timedelta64(1, "D")
    # Cast to scalar np.datetime64 to satisfy pandas typing (we expect scalars here)
    date_list = pd.date_range(cast(np.datetime64, date_min), cast(np.datetime64, end_plus_one), freq="D").to_list()
    _vprint(verbose, 2, f"Found {len(date_list)} days")
    return cast(List[np.datetime64], date_list)


def _grab_EUI_data(
    date: Union[np.datetime64, NDArray],
    local_dir: str | Path = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0,
) -> List[Path]:
    """
    Retrieve EUI FITS files for a specific date from a local directory.

    Notes
    -----
    The `local_dir` must be structured as yyyy/mm/dd/filename.fits.

    Parameters
    ----------
    date : np.datetime64
        The date for which to grab EUI data.
    local_dir : str or Path, optional
        Base directory where EUI data is stored (default is
        "/archive/SOLAR-ORBITER/EUI/data_internal/L2").
    verbose : int, optional
        Verbosity level for logging (default is 0).

    Returns
    -------
    List[Path]
        A list of Path objects pointing to EUI FITS files for the specified date.
        Returns an empty list if the target folder does not exist.
    """
    # Convert to millisecond-precision datetime64 and then to Python datetime
    date = cast(np.datetime64, _as_datetime64_ms(date))
    date_dt = date.astype(datetime.datetime)
    year = date_dt.year
    month = date_dt.month
    day = date_dt.day

    local_dir = Path(local_dir)
    target_folder = local_dir / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"

    _vprint(verbose, 2, f"Looking for EUI data in {target_folder}")
    if target_folder.exists():
        eui_files = list(target_folder.glob("*.fits"))
        _vprint(
            verbose, 2, f"Found {len(eui_files)} EUI files for date {date_dt.date()}"
        )
        return eui_files
    else:
        _vprint(verbose, -1, f"Directory does not exist: {target_folder}")
        return []


def get_EUI_paths(
    date_min: Union[np.datetime64, NDArray],
    date_max: Union[np.datetime64, NDArray],
    local_dir: str | Path = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0,
) -> List[Path]:
    """
    Collect all EUI FITS file paths between date_min and date_max (inclusive).

    Parameters
    ----------
    date_min : np.datetime64
        The start date for the search.
    date_max : np.datetime64
        The end date for the search.
    local_dir : str or Path, optional
        Base directory where EUI data is stored (default is
        "/archive/SOLAR-ORBITER/EUI/data_internal/L2").
    verbose : int, optional
        Verbosity level for logging (default is 0).

    Returns
    -------
    List[Path]
        A sorted list of Path objects for EUI FITS files whose timestamps
        (extracted from filenames) lie between date_min and date_max.
    """
    # Ensure inputs are millisecond-precision datetime64
    date_min = _as_datetime64_ms(date_min)
    date_max = _as_datetime64_ms(date_max)
    assert date_min <= date_max, "date_min must be less than or equal to date_max"

    local_dir = Path(local_dir)
    _vprint(
        verbose, 2, f"Getting EUI paths from {date_min} to {date_max} in {local_dir}"
    )

    # 1) Gather all days in the range
    all_days = _find_all_days(date_min, date_max, verbose)

    # 2) Collect all FITS files across those days
    eui_paths: List[Path] = []
    for day in all_days:
        _vprint(verbose, 2, f"Grabbing EUI data for {day}")
        eui_paths.extend(_grab_EUI_data(day, local_dir, verbose))

    if not eui_paths:
        _vprint(verbose, 1, "No EUI files found in the specified date range.")
        return []

    # 3) Sort collected paths
    eui_paths_array = np.array(eui_paths, dtype=object)
    eui_paths_sorted = np.sort(eui_paths_array)

    # 4) Extract timestamps from filenames and convert to datetime64
    def _extract_datetime(path: Path) -> np.datetime64 | None:
        match = re.search(r"(\d{8})T(\d{6})", path.name)
        if match:
            dt_str = match.group(0)
            dt_obj = datetime.datetime.strptime(dt_str, "%Y%m%dT%H%M%S")
            return cast(np.datetime64, _as_datetime64_ms(dt_obj))
        return None

    _vprint(verbose, 2, "Extracting timestamps from filenames")
    vectorized_extract = np.vectorize(_extract_datetime)
    eui_dates = vectorized_extract(eui_paths_sorted)

    # 5) Filter paths whose extracted dates lie within [date_min, date_max]
    mask = np.array(
        [(dt is not None and date_min <= dt <= date_max) for dt in eui_dates]
    )
    filtered_paths = list(np.array(eui_paths_sorted, dtype=object)[mask])

    _vprint(
        verbose, 2, f"{len(filtered_paths)} files remain after filtering by date range"
    )
    return filtered_paths


def split_eui_paths_by_mode(
    list_paths: Iterable[Path], verbose: int = 0
) -> Dict[str, np.ndarray[Any]]:
    """
    Split EUI file paths into groups based on the instrument mode encoded in the filename.

    The mode is defined as the substring between the pattern 'solo_L[0-4]_' and the timestamp
    pattern '_YYYYMMDDThhMMSS' in each filename.

    Parameters
    ----------
    list_paths : np.ndarray[Path]
        Array of Path objects pointing to EUI FITS files.
    verbose : int, optional
        Verbosity level for logging (default is 0).

    Returns
    -------
    Dict[str, np.ndarray[Path]]
        A dictionary mapping each unique mode string to an array of Path objects
        corresponding to that mode.
    """
    # Ensure we work with a numpy array of Path
    list_paths = np.array(list_paths, dtype=object)
    _vprint(verbose, 2, "Splitting EUI data into different modes")
    _vprint(verbose, 2, f"Found {len(list_paths)} EUI paths")

    # Vectorized extraction of mode from each filename.
    # The regex 'solo_L[0-4]_' marks the start of mode,
    # and '_YYYYMMDDThhMMSS' marks the end.

    def _extract_mode(path: Path) -> str:
        name = path.name
        start_match = re.search(r"solo_L[0-4]_", name)
        end_match = re.search(r"_(\d{8})T(\d{6})", name)
        if start_match and end_match:
            return name[start_match.end() : end_match.start()]
        return ""

    modes = np.vectorize(_extract_mode)(list_paths)

    unique_modes = np.unique(modes)
    _vprint(verbose, 1, f"Found {len(unique_modes)} modes")
    _vprint(verbose, 2, "Modes found:\n\t" + "\n\t".join(unique_modes))

    # Initialize dictionary with an empty list for each mode
    dict_paths: Dict[str, list[Path]] = {mode: [] for mode in unique_modes}

    # Assign each path to its corresponding mode
    for path, mode in zip(list_paths, modes):
        if mode:
            dict_paths[mode].append(path)

    # Convert lists to numpy arrays for consistency
    dict_paths_array: Dict[str, np.ndarray[Any]] = {
        mode: np.array(paths, dtype=object) for mode, paths in dict_paths.items()
    }
    return dict_paths_array


def get_closest_EUIFSI304_paths(
    date_ref: np.datetime64,
    interval: np.timedelta64,
    local_dir: Union[str, Path] = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0,
) -> List[Path]:
    """
    Find the EUI FITS files (FSI 304) whose timestamps are closest to a reference date,
    within a specified tolerance interval.

    Parameters
    ----------
    date_ref : np.datetime64
        Reference date/time.
    interval : np.timedelta64
        Maximum allowed offset (±) from date_ref.
    local_dir : str or Path, optional
        Base directory where EUI data is stored.
    verbose : int, optional
        Verbosity level for logging (default 0).

    Returns
    -------
    List[Path]
        Paths to the FITS files whose timestamps are the closest to date_ref,
        among those within [date_ref - interval, date_ref + interval].
        If no files lie within that window, returns an empty list.
    """
    date_ref = cast(np.datetime64, _as_datetime64_ms(date_ref))  # Normalize to millisecond precision
    half_int = np.asarray(
        interval, dtype="timedelta64[ms]"
    )  # Convert interval to millisecond precision
    lower = date_ref - half_int
    upper = date_ref + half_int
    assert lower <= upper, "Interval must be non-negative"

    local_dir = Path(local_dir)
    _vprint(verbose, 1, f"Searching EUI paths around {date_ref} ± {interval}")

    # 1) Gather all days that might contain candidates
    days = _find_all_days(lower, upper, verbose)  # type: ignore[arg-type]
    _vprint(verbose, 2, f"Checking {len(days)} days from {lower} to {upper}")

    # 2) Collect all FITS files in that window
    all_paths: List[Path] = []
    for day in days:
        _vprint(verbose, 2, f"  Grabbing data for {day}")
        all_paths.extend(_grab_EUI_data(day, local_dir, verbose))

    if not all_paths:
        _vprint(verbose, 1, "No EUI files found in the interval.")
        return []

    all_paths = cast(List[Path], split_eui_paths_by_mode(all_paths)["eui-fsi304-image"])

    # 3) Extract timestamps from filenames
    def extract_dt(p: Path) -> Union[np.datetime64, None]:
        m = re.search(r"(\d{8})T(\d{6})", p.name)
        if not m:
            return None
        dt = datetime.datetime.strptime(m.group(0), "%Y%m%dT%H%M%S")
        return cast(np.datetime64, _as_datetime64_ms(dt))  # Normalize to millisecond precision

    _vprint(verbose, 2, f"Extracting timestamps from {len(all_paths)} files")
    paths_arr = np.array(all_paths, dtype=object)
    dates = np.array([extract_dt(p) for p in paths_arr], dtype="datetime64[ms]")

    # 4) Compute absolute differences to reference, mask out None
    valid_mask = dates != _as_datetime64_ms("NaT")
    if not np.any(valid_mask):
        _vprint(verbose, 1, "No valid timestamps parsed.")
        return []

    diffs = np.abs(dates[valid_mask] - date_ref)
    min_diff = diffs.min()
    _vprint(verbose, 1, f"Minimum time difference = {min_diff}")

    # 5) Select all paths that achieve this minimum difference
    candidates = paths_arr[valid_mask][diffs == min_diff]
    # Sort lexicographically for reproducibility
    closest = sorted(candidates.tolist())

    _vprint(verbose, 1, f"Found {len(closest)} closest file(s) within ±{interval}")
    return closest


# Helper functions for FSI map selection and time extraction.
def _nearest_imager_index(
    step_time: np.datetime64,
    fsi_times: np.ndarray,
    threshold: np.timedelta64 | None,
    verbose: int = 0,
) -> int:
    deltas = np.abs(fsi_times - step_time) / np.timedelta64(1, "s")
    idx = int(np.argmin(deltas))
    sel_dt = deltas[idx]
    _vprint(verbose, 2, f"Nearest FSI idx={idx}, Δt={sel_dt:.3f}s for step {step_time}")
    if threshold is not None and sel_dt > (threshold / np.timedelta64(1, "s")):
        raise ValueError(
            f"No FSI map within {threshold / np.timedelta64(1, 's')} s for step at {step_time}"
        )
    return idx


def _extract_map_time(
    entry: Union[GenericMap, str, Path], verbose: int = 0
) -> np.datetime64:
    if isinstance(entry, GenericMap):
        t = cast(np.datetime64, _as_datetime64_ms(entry.date.isot))
        _vprint(verbose, 2, f"Map time (GenericMap): {t}")
        return t
    header = None
    for ext in (1, 0):
        try:
            header = fits.getheader(entry, ext=ext)
            break
        except (FileNotFoundError, OSError, IndexError):
            continue
    if header is None:
        raise ValueError(f"Unable to read FITS header from {entry}")
    hdr = header
    date_key = hdr.get("DATE-AVG") or hdr.get("DATE-OBS") or hdr.get("DATE_BEG")
    if date_key is None:
        raise ValueError(f"No DATE-* keyword found in {entry}")
    t = cast(np.datetime64, _as_datetime64_ms(date_key))
    _vprint(verbose, 2, f"Map time ({Path(entry).name}): {t}")
    return t


def interpol2d(image, x, y, fill, order, dst=None):
    """ "
    taken from Frederic interpol2d function
    """
    bad = np.logical_or(x == np.nan, y == np.nan)
    x = np.where(bad, -1, x)
    y = np.where(bad, -1, y)

    coords = np.stack((y.ravel(), x.ravel()), axis=0)
    return_ = False

    if dst is None:
        return_ = True
        dst = np.empty(x.shape, dtype=image.dtype)

    map_coordinates(
        image,
        coords,
        order=order,
        mode="constant",
        cval=fill,
        output=dst.ravel(),
        prefilter=False,
    )
    if return_:
        return dst


# build synthetic FSI raster.
def meta_to_header(meta):
    hdr = fits.Header()
    for k, v in meta.items():
        try:
            hdr[k] = v
        except Exception:
            pass
    return hdr


def _pixel_world_with_optional_time(
    wcs_obj: WCS,
    xx: np.ndarray,
    yy: np.ndarray,
) -> tuple[SkyCoord, astropy.time.core.Time | None]:
    pixel_inputs: list[Any] = [xx * u.pix, yy * u.pix]
    if getattr(wcs_obj, "pixel_n_dim", 2) >= 3:
        pixel_inputs.append(np.zeros_like(xx))
    world_result = wcs_obj.pixel_to_world(*pixel_inputs)
    # Check if result is a sequence with time/temporal component
    if isinstance(world_result, (tuple, list)) and len(world_result) > 1:
        if isinstance(world_result[1], astropy.time.core.Time):
            coords_spice = cast(SkyCoord, world_result[0])
            time_payload = world_result[1]
        elif isinstance(world_result[1], u.Quantity):
            # Temporal WCS was added manually - convert Quantity to Time if possible
            coords_spice = cast(SkyCoord, world_result[0])
            try:
                # Try to convert Quantity to Time using map's reference time
                time_payload = astropy.time.Time(
                    world_result[1], format="mjd", scale="utc"
                )
            except Exception:
                time_payload = None
        else:
            coords_spice = cast(
                SkyCoord, world_result[0] if len(world_result) > 0 else world_result
            )
            time_payload = None
    else:
        coords_spice = cast(SkyCoord, world_result)
        time_payload = None
    return coords_spice, time_payload


def _coerce_step_times(
    time_payload: Any | None,
    nx: int,
    default_time: np.datetime64,
) -> NDArray:
    if time_payload is None:
        return np.full(nx, default_time, dtype="datetime64[ms]")
    try:
        dt_values = np.asarray(time_payload.to_datetime(), dtype="datetime64[ms]")
    except Exception:
        return np.full(nx, default_time, dtype="datetime64[ms]")
    if dt_values.ndim == 0:
        return np.full(nx, dt_values, dtype="datetime64[ms]")
    if dt_values.ndim > 1:
        return dt_values[0]
    return dt_values


# ==============================================================
# Helper: build synthetic raster from FSI maps
# ==============================================================
def build_synthetic_raster_from_maps(
    spice_map: GenericMap,
    fsi_maps: Sequence[Union[GenericMap, str, Path]],
    threshold_time: np.timedelta64 | None = None,
    order: int = 2,
    verbose: int = 0,
    **kwargs,
) -> GenericMap:
    """
    Sample FSI maps (or FITS paths) onto the spatial grid of a SPICE map and
    return a synthetic raster as a GenericMap without saving to disk.

    FSI inputs can be pre-loaded maps or file paths. For each SPICE slit
    column, the closest-in-time FSI map is chosen. Files are loaded lazily on
    first use to avoid reading tens of FITS files upfront.

    Parameters
    ----------
    spice_map : GenericMap
        Target raster geometry.
    fsi_maps : Sequence[GenericMap | str | Path]
        FSI images or paths to FITS files; each must carry a valid ``date``.
    threshold_time : np.timedelta64, optional
        Maximum allowed |Δt| between slit time and selected FSI time.
    order : int, optional
        Interpolation order for ``interpol2d`` (default 2).
    """
    if not fsi_maps:
        raise ValueError("fsi_maps must be non-empty")

    _vprint(verbose, 1, "Building synthetic raster from FSI maps")

    ny, nx = cast(NDArray, spice_map.data).shape

    fsi_entries: list[Union[GenericMap, Path, str]] = []
    fsi_times_list: list[np.datetime64] = []
    for entry in fsi_maps:
        fsi_entries.append(entry)
        fsi_times_list.append(_extract_map_time(entry, verbose=verbose))
    fsi_times: NDArray = np.array(fsi_times_list)

    WCS3D = WCS(meta_to_header(spice_map.meta))

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    coords_spice, time_matrix = _pixel_world_with_optional_time(WCS3D, xx, yy)
    step_times: NDArray = _coerce_step_times(
        time_matrix,
        nx,
        cast(np.datetime64, _as_datetime64_ms(spice_map.date.isot)),
    )
    _vprint(verbose, 2, "Computed step_times from WCS metadata")

    data_composed = np.full((ny, nx), np.nan, dtype=float)
    fsi_cache: dict[int, GenericMap] = {}
    for i in range(nx):
        idx = _nearest_imager_index(
            step_times[i], fsi_times, threshold_time, verbose=verbose
        )
        entry = fsi_entries[idx]
        if isinstance(entry, GenericMap):
            fsi_map = entry
        else:
            if idx not in fsi_cache:
                _vprint(
                    verbose,
                    2,
                    f"Loading FSI map idx {idx} from {cast(Path, entry).name}",
                )
                fsi_cache[idx] = cast(GenericMap, Map(entry))
            fsi_map = fsi_cache[idx]

        coords_col = SkyCoord(
            cast(NDArray, coords_spice.Tx)[:, i],
            cast(NDArray, coords_spice.Ty)[:, i],
            frame=coords_spice.frame,
        )
        x_fsi, y_fsi = fsi_map.world_to_pixel(coords_col)
        data_composed[:, i] = interpol2d(
            fsi_map.data,
            x=np.asarray(x_fsi),
            y=np.asarray(y_fsi),
            order=order,
            fill=np.nan,
        )

    hdr = cast(Dict, spice_map.meta).copy()
    hdr["SYNRASTR"] = "FSI->SPICE synthetic raster"
    hdr["SRCIMGS"] = len(fsi_maps)
    hdr.setdefault("DATE-AVG", spice_map.date.isot)

    plot_settings = getattr(spice_map, "plot_settings", {})
    plot_settings["norm"] = normit(data_composed)

    return GenericMap(data_composed, hdr, plot_settings=plot_settings)


# ==============================================================
# Helper: reproject map onto reference map grid
# ==============================================================
def reproject_map_to_reference(
    ref_map: GenericMap,
    input_map: GenericMap,
    order: Any = "bilinear",
) -> GenericMap:
    """
    Reproject input_map onto the WCS & pixel grid of ref_map.

    Parameters
    ----------
    ref_map : sunpy.map.Map
        Reference map (defines output WCS and shape).
    input_map : sunpy.map.Map
        Map to be reprojected.
    order : str or int
        Interpolation order. For some reproject versions we may need
        order=1 instead of "bilinear".

    Returns
    -------
    new_map : sunpy.map.Map
        input_map data reprojected onto ref_map's grid.
    """
    target_wcs = ref_map.wcs
    target_shape = cast(np.ndarray, ref_map.data).shape

    reprojected_data, footprint = reproject_interp(
        input_map,  # or (input_map.data, input_map.wcs)
        target_wcs,
        shape_out=target_shape,
        order=order,
    )

    # Mask outside-footprint pixels as NaN
    reprojected_data = np.where(footprint > 0, reprojected_data, np.nan)

    # Some versions/instrument combinations produce very small footprints
    # when passing the Map object directly; try passing the (data, wcs)
    # tuple as a fallback to improve coverage.
    finite_frac = np.isfinite(reprojected_data).sum() / reprojected_data.size
    if finite_frac < 0.05:
        try:
            reprojected_data2, footprint2 = reproject_interp(
                (input_map.data, input_map.wcs),
                target_wcs,
                shape_out=target_shape,
                order=order,
            )
            reprojected_data2 = np.where(footprint2 > 0, reprojected_data2, np.nan)
            if (
                np.isfinite(reprojected_data2).sum()
                > np.isfinite(reprojected_data).sum()
            ):
                reprojected_data = reprojected_data2
                footprint = footprint2
        except Exception:
            # If fallback fails, keep original reprojected_data
            pass

    new_meta = cast(dict, ref_map.meta).copy()
    new_map = sunpy.map.Map(
        reprojected_data, new_meta, plot_settings=input_map.plot_settings
    )

    return cast(GenericMap, new_map)


def enlarge_map_by_padding(
    smap: GenericMap,
    pad_x: int,
    pad_y: int,
    fill_value: Union[int, float] = np.nan,
) -> GenericMap:
    """Pad or crop a SunPy map while maintaining consistent WCS metadata.

    Extends the field of view by adding padding around the edges, or crops
    by using negative padding values. The WCS reference pixel (CRPIX) is
    adjusted to maintain coordinate consistency.

    Parameters
    ----------
    smap : GenericMap
        Input SunPy map to pad/crop
    pad_x : int
        Pixels to add on left and right (negative to crop)
    pad_y : int
        Pixels to add on top and bottom (negative to crop)
    fill_value : int or float, default=np.nan
        Value to fill padded regions

    Returns
    -------
    GenericMap
        New map with adjusted size and updated WCS metadata

    Raises
    ------
    ValueError
        If resulting dimensions would be non-positive

    Notes
    -----
    The function updates NAXIS1, NAXIS2, CRPIX1, and CRPIX2 in the metadata
    to ensure the World Coordinate System remains valid after padding.
    Original data is copied to the appropriate position in the new array.

    For positive padding, the original data is centered with fill_value in
    the added margins. For negative padding (cropping), data is extracted
    from the interior of the original map.

    Examples
    --------
    >>> import sunpy.map
    >>> import numpy as np
    >>> # Create a simple test map
    >>> data = np.ones((100, 100))
    >>> meta = {'naxis1': 100, 'naxis2': 100, 'crpix1': 50, 'crpix2': 50}
    >>> original = sunpy.map.Map(data, meta)
    >>> # Pad by 10 pixels on each side
    >>> padded = enlarge_map_by_padding(original, 10, 10)
    >>> padded.data.shape
    (120, 120)
    >>> padded.meta['crpix1']  # Reference pixel shifted
    60
    """
    _data: np.ndarray = cast(np.ndarray, smap.data)
    _meta: Mapping[str, Any] = cast(Mapping[str, Any], smap.meta)
    ny, nx = _data.shape
    new_nx = nx + 2 * pad_x
    new_ny = ny + 2 * pad_y
    if new_nx <= 0 or new_ny <= 0:
        raise ValueError(
            f"Cropping too large: resulting shape would be ({new_ny}, {new_nx})."
        )

    new_data = np.full((new_ny, new_nx), fill_value, dtype=_data.dtype)
    x_src0 = max(0, -pad_x)
    y_src0 = max(0, -pad_y)
    x_dst0 = max(0, pad_x)
    y_dst0 = max(0, pad_y)
    copy_w = min(nx - x_src0, new_nx - x_dst0)
    copy_h = min(ny - y_src0, new_ny - y_dst0)
    x_src1 = x_src0 + copy_w
    y_src1 = y_src0 + copy_h
    x_dst1 = x_dst0 + copy_w
    y_dst1 = y_dst0 + copy_h
    new_data[y_dst0:y_dst1, x_dst0:x_dst1] = smap.data[y_src0:y_src1, x_src0:x_src1]

    new_meta: dict = dict(_meta).copy()
    new_meta["naxis1"] = new_nx
    new_meta["naxis2"] = new_ny
    new_meta["crpix1"] = _meta["crpix1"] + pad_x
    new_meta["crpix2"] = _meta["crpix2"] + pad_y
    return cast(
        GenericMap,
        sunpy.map.Map(new_data, new_meta, plot_settings=smap.plot_settings),
    )


def clamp_point(point: Tuple[int, int], shift_x: int, shift_y: int) -> Tuple[int, int]:
    """Clamp a 2D point to lie within rectangular search bounds.

    Parameters
    ----------
    point : Tuple[int, int]
        Input point coordinates (dx, dy)
    shift_x : int
        Maximum allowed x-shift
    shift_y : int
        Maximum allowed y-shift

    Returns
    -------
    Tuple[int, int]
        Clamped coordinates within [-shift_x, shift_x] × [-shift_y, shift_y]
    """
    return (
        int(np.clip(point[0], -shift_x, shift_x)),
        int(np.clip(point[1], -shift_y, shift_y)),
    )


def get_coord_mat(map, as_skycoord=False):
    res = sunpy.map.maputils.all_coordinates_from_map(map)
    if as_skycoord:
        return res
    try:
        lon = res.spherical.lon.arcsec
        lat = res.spherical.lat.arcsec
    except AttributeError:
        lon = res.lon.value
        lat = res.lat.value
    return lon, lat


def _vprint(verbose: int, level: int, *args, **kwargs) -> None:
    """
    Print a message at a given verbosity level, prefixed by a label.

    Parameters
    ----------
    verbose : int
        Current verbosity setting.
    level : int
        The threshold level for this message:
          -1 → "Warning"
           0 → "Info"
           1 → "Verbose"
           2 → "Debug"
           3 → "Debug_Plot"
           4 → "Debug_Plot_Save"
    *args, **kwargs
        Passed to built-in print() after the prefix.
    """
    if verbose < level:
        return

    # Map levels to human-readable labels
    labels = {
        -1: ("Warning", "\033[91m"),  # bright red
        #  0: ("Info", "\033[96m"),         # cyan
        0: ("Info", "\033[0m"),  # default
        1: ("Verbose", "\033[92m"),  # green
        2: ("Debug", "\033[90m"),  # faint gray
        3: ("Debug_Plot", "\033[90m"),
        4: ("Debug_Plot", "\033[90m"),
    }
    prefix, color = labels.get(level, (f"Level_{level}", "\033[0m"))
    reset = "\033[0m"
    print(f"{color}[{prefix}]{reset}", *args, **kwargs)


def get_pixel_scale_quantity(map_obj: GenericMap) -> Tuple[u.Quantity, u.Quantity]:
    """
    Get pixel scale from map in angular units.

    Parameters
    ----------
    map_obj : GenericMap
        Map to extract pixel scale from

    Returns
    -------
    pixel_scale_x : u.Quantity
        Pixel scale in X (arcseconds per pixel)
    pixel_scale_y : u.Quantity
        Pixel scale in Y (arcseconds per pixel)
    """
    cdelt1 = abs(cast(Dict, map_obj.meta).get("CDELT1", 1.0))
    cdelt2 = abs(cast(Dict, map_obj.meta).get("CDELT2", 1.0))
    cunit1 = cast(Dict, map_obj.meta).get("CUNIT1", "arcsec")
    cunit2 = cast(Dict, map_obj.meta).get("CUNIT2", "arcsec")
    pixel_scale_x = (cdelt1 * u.Unit(cunit1)).to(u.arcsec)
    pixel_scale_y = (cdelt2 * u.Unit(cunit2)).to(u.arcsec)
    return pixel_scale_x, pixel_scale_y


def arcsec_to_pixels(
    value_arcsec: u.Quantity | Tuple[u.Quantity, u.Quantity], map_obj: GenericMap
) -> Tuple[int, int]:
    """
    Convert angular distance to pixels for a given map.

    Parameters
    ----------
    value_arcsec : u.Quantity or tuple of u.Quantity
        Angular distance with angle units. Can be:
        - Single value: uses same value for both X and Y
        - Tuple (x, y): different values for each axis
    map_obj : GenericMap
        Map to use for pixel scale

    Returns
    -------
    pixels : int or tuple of int
        Distance in pixels. Returns:
        - Single int if input was single value
        - Tuple (px_x, px_y) if input was tuple
    """
    pixel_scale_x, pixel_scale_y = get_pixel_scale_quantity(map_obj)

    if isinstance(value_arcsec, tuple):
        # Separate X and Y values
        value_x, value_y = value_arcsec
        pixels_x = int((value_x / pixel_scale_x).to(u.dimensionless_unscaled).value)
        pixels_y = int((value_y / pixel_scale_y).to(u.dimensionless_unscaled).value)
        return pixels_x, pixels_y
    else:
        # Single value: convert using both pixel scales
        pixels_x = int(
            (value_arcsec / pixel_scale_x).to(u.dimensionless_unscaled).value
        )
        pixels_y = int(
            (value_arcsec / pixel_scale_y).to(u.dimensionless_unscaled).value
        )
        return pixels_x, pixels_y


def pixels_to_arcsec(
    value_pixels: float | Tuple[float, float], map_obj: GenericMap
) -> Tuple[u.Quantity, u.Quantity] | u.Quantity:
    """
    Convert pixels to angular distance for a given map.

    Parameters
    ----------
    value_pixels : float or tuple of float
        Distance in pixels. Can be:
        - Single value: same for both axes
        - Tuple (px_x, px_y): separate for each axis
    map_obj : GenericMap
        Map to use for pixel scale

    Returns
    -------
    arcsec : u.Quantity or tuple of u.Quantity
        Angular distance in arcseconds.
        - Tuple (x, y) if input was tuple
        - Single Quantity if input was single value (uses X scale)
    """
    pixel_scale_x, pixel_scale_y = get_pixel_scale_quantity(map_obj)

    if isinstance(value_pixels, tuple):
        px_x, px_y = value_pixels
        arcsec_x = px_x * pixel_scale_x
        arcsec_y = px_y * pixel_scale_y
        return arcsec_x, arcsec_y
    else:
        # Single value: use X pixel scale for backward compatibility
        arcsec = value_pixels * pixel_scale_x
        return arcsec


def bin_map(map_obj: GenericMap, bin_factor: int | Tuple[int, int]) -> GenericMap:
    """
    Bin a map by averaging pixels.

    Parameters
    ----------
    map_obj : GenericMap
        Map to bin
    bin_factor : int or tuple of int
        Binning factor. Can be:
        - Single int: same binning for both axes
        - Tuple (bin_x, bin_y): different binning per axis

    Returns
    -------
    binned_map : GenericMap
        Binned map with updated WCS
    """
    # Handle tuple or single value
    if isinstance(bin_factor, tuple):
        bin_x, bin_y = bin_factor
    else:
        bin_x = bin_y = bin_factor

    if bin_x <= 1 and bin_y <= 1:
        return map_obj

    # Apply NaN-safe uniform filter, then downsample
    smoothed_data = no_nan_uniform_filter(
        np.asarray(map_obj.data),
        remove_percentile=99,
        size=(bin_y, bin_x),  # Note: numpy is (rows, cols) = (Y, X)
    )
    binned_data = smoothed_data[::bin_y, ::bin_x]

    # Update metadata for binned WCS
    new_meta = cast(Dict, map_obj.meta).copy()
    new_meta["CDELT1"] = cast(Dict, map_obj.meta)["CDELT1"] * bin_x
    new_meta["CDELT2"] = cast(Dict, map_obj.meta)["CDELT2"] * bin_y
    new_meta["CRPIX1"] = cast(Dict, map_obj.meta)["CRPIX1"] / bin_x
    new_meta["CRPIX2"] = cast(Dict, map_obj.meta)["CRPIX2"] / bin_y
    new_meta["NAXIS1"] = binned_data.shape[1]
    new_meta["NAXIS2"] = binned_data.shape[0]

    # Preserve plot_settings (norm, cmap, etc.) from original map
    plot_settings = getattr(map_obj, "plot_settings", None)
    return cast(GenericMap, Map(binned_data, new_meta, plot_settings=plot_settings))


def unbin_map_lattice(
    binned_map: GenericMap,
    bin_factor: Union[int, Tuple[int, int]],
    original_shape: Tuple[int, int],
) -> GenericMap:
    """
    Inverse of `bin_map` in the sense of restoring the original grid (shape + WCS),
    without inventing pixel values.

    It creates a NaN array of original_shape and inserts binned pixels at
    [0::bin_y, 0::bin_x].

    Parameters
    ----------
    binned_map : GenericMap
        Output of `bin_map`.
    bin_factor : int or (bin_x, bin_y)
        Same factor used in bin_map.
    original_shape : (ny, nx)
        Original (pre-bin) data shape.

    Returns
    -------
    GenericMap
        Map with data.shape == original_shape and WCS inverted back.
    """
    if isinstance(bin_factor, tuple):
        bin_x, bin_y = bin_factor
    else:
        bin_x = bin_y = bin_factor

    ny_out, nx_out = original_shape

    # If no binning, just return (or you could still rebuild meta)
    if bin_x <= 1 and bin_y <= 1:
        return binned_map

    out = np.full((ny_out, nx_out), np.nan, dtype=np.float64)

    data = np.asarray(binned_map.data, dtype=np.float64)
    # Expected lattice shape from slicing: out[0::bin_y, 0::bin_x]
    view = out[0::bin_y, 0::bin_x]

    # Defensive: handle slight mismatch if metadata/shape inconsistent
    ny_fill = min(view.shape[0], data.shape[0])
    nx_fill = min(view.shape[1], data.shape[1])
    view[:ny_fill, :nx_fill] = data[:ny_fill, :nx_fill]

    # Invert metadata updates (exact inverse of your bin_map)
    new_meta = cast(Dict, binned_map.meta).copy()
    new_meta["CDELT1"] = cast(Dict, binned_map.meta)["CDELT1"] / bin_x
    new_meta["CDELT2"] = cast(Dict, binned_map.meta)["CDELT2"] / bin_y
    new_meta["CRPIX1"] = cast(Dict, binned_map.meta)["CRPIX1"] * bin_x
    new_meta["CRPIX2"] = cast(Dict, binned_map.meta)["CRPIX2"] * bin_y
    new_meta["NAXIS1"] = nx_out
    new_meta["NAXIS2"] = ny_out

    plot_settings = getattr(binned_map, "plot_settings", None)
    return cast(GenericMap, Map(out, new_meta, plot_settings=plot_settings))


def apply_shift_to_map(
    map_obj: GenericMap, shift_x_arcsec: u.Quantity, shift_y_arcsec: u.Quantity
) -> GenericMap:
    """
    Apply shift to map by modifying WCS CRVAL.

    Parameters
    ----------
    map_obj : GenericMap
        Map to shift
    shift_x_arcsec : u.Quantity
        Shift in X (arcsec)
    shift_y_arcsec : u.Quantity
        Shift in Y (arcsec)

    Returns
    -------
    shifted_map : GenericMap
        Map with updated WCS
    """

    # Convert shifts to pixels using separate pixel scales
    pixel_scale_x, pixel_scale_y = get_pixel_scale_quantity(map_obj)
    dx = (shift_x_arcsec / pixel_scale_x).to(u.dimensionless_unscaled).value
    dy = (shift_y_arcsec / pixel_scale_y).to(u.dimensionless_unscaled).value

    # Apply shift via WCS correction
    best_params = {"dx": dx, "dy": dy, "squeeze_x": 1.0, "squeeze_y": 1.0}
    corrected_map = make_corrected_wcs_map(map_obj, best_params)

    return corrected_map


def apply_shift_and_scale_to_map(
    map_obj: GenericMap,
    shift_x_arcsec: u.Quantity,
    shift_y_arcsec: u.Quantity,
    scale_x: float,
    scale_y: float,
) -> GenericMap:
    """
    Apply shift and scale to map by modifying WCS.

    Parameters
    ----------
    map_obj : GenericMap
        Map to transform
    shift_x_arcsec : u.Quantity
        Shift in X (arcsec)
    shift_y_arcsec : u.Quantity
        Shift in Y (arcsec)
    scale_x : float
        Scale factor in X
    scale_y : float
        Scale factor in Y

    Returns
    -------
    transformed_map : GenericMap
        Map with updated WCS
    """

    # Convert shifts to pixels using separate pixel scales
    pixel_scale_x, pixel_scale_y = get_pixel_scale_quantity(map_obj)
    dx = (shift_x_arcsec / pixel_scale_x).to(u.dimensionless_unscaled).value
    dy = (shift_y_arcsec / pixel_scale_y).to(u.dimensionless_unscaled).value

    # Apply transformation
    best_params = {"dx": dx, "dy": dy, "squeeze_x": scale_x, "squeeze_y": scale_y}
    corrected_map = make_corrected_wcs_map(map_obj, best_params)

    return corrected_map
