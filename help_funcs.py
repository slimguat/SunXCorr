# -*- coding: utf-8 -*-


from typing import Sequence, Union
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunpy.map import GenericMap, Map
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from saffron import utils
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from typing import Sequence, Union
from saffron.utils import normit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock

def to_submap(
    target_map: GenericMap,
    source_map: GenericMap,
    expand: Sequence[u.Quantity] = (0 * u.arcsec, 0 * u.arcsec),
    solar_rotation: bool = True,
) -> GenericMap:
    """
    Create a submap of ``target_map`` corresponding to the spatial extent of
    ``source_map``, transformed into the observer frame of ``target_map``.

    The function extracts the full boundary of ``source_map`` in helioprojective
    coordinates, converts those boundary coordinates to the observer frame of
    ``target_map`` (optionally accounting for solar differential rotation),
    expands the resulting bounding box by a user-defined angular margin,
    and returns the corresponding submap from ``target_map``.

    This is typically used to spatially align observations taken at different
    times, viewpoints, or instruments.

    Parameters
    ----------
    target_map : GenericMap
        Map from which the submap will be extracted. The observer time and
        coordinate frame of this map define the final reference system.

    source_map : GenericMap
        Map whose spatial footprint is used to define the submap region.
        Its full coordinate extent is projected into the observer frame of
        ``target_map``.

    expand : Sequence[u.Quantity], optional
        Two-element sequence ``(dTx, dTy)`` specifying an angular expansion
        applied to the bounding box in helioprojective longitude (Tx) and
        latitude (Ty), respectively. Each element must have angular units.
        Default is no expansion.

    solar_rotation : bool, optional
        If ``True``, boundary coordinates are propagated to the observation
        time of ``target_map`` using solar differential rotation via
        ``solar_rotate_coordinate``.
        If ``False``, coordinates are only transformed geometrically to the
        target observer frame without temporal rotation.
        Default is ``True``.

    Returns
    -------
    sunpy.map.Map
        A submap of ``target_map`` whose spatial extent corresponds to the
        transformed (and optionally solar-rotated) boundaries of ``source_map``.

    Notes
    -----
    - If NaN values appear after coordinate transformation (e.g. due to
      off-disk or invalid projections), the function falls back to transforming
      the full coordinate grid of ``source_map`` and recomputes the bounding
      box using finite extrema.
    - The submap is defined in helioprojective coordinates using ``Tx`` and
      ``Ty`` limits.
    - This function assumes that ``expand`` has exactly two elements.

    """
    map1 = source_map
    map2 = target_map

    lonlat_coords = utils.get_coord_mat(map1, as_skycoord=True)

    bottom_left = SkyCoord(
        lonlat_coords.Tx.min(),
        lonlat_coords.Ty.min(),
        frame=lonlat_coords.frame,
    )
    top_right = SkyCoord(
        lonlat_coords.Tx.max(),
        lonlat_coords.Ty.max(),
        frame=lonlat_coords.frame,
    )

    if solar_rotation:
        bottom_left_rotated = solar_rotate_coordinate(
            bottom_left,
            observer=map2.observer_coordinate,
        )
        top_right_rotated = solar_rotate_coordinate(
            top_right,
            observer=map2.observer_coordinate,
        )
    else:
        bottom_left_rotated = bottom_left.transform_to(map2.coordinate_frame)
        top_right_rotated = top_right.transform_to(map2.coordinate_frame)

    if any(
        np.isnan(val)
        for val in (
            bottom_left_rotated.Tx,
            bottom_left_rotated.Ty,
            top_right_rotated.Tx,
            top_right_rotated.Ty,
        )
    ):
        print("\033[91mThere is a NaN in the transformed coordinates\033[00m")
        new_coords = lonlat_coords.transform_to(map2.coordinate_frame)
        bottom_left_rotated = SkyCoord(
            np.nanmin(new_coords.Tx),
            np.nanmin(new_coords.Ty),
            frame=new_coords.frame,
        )
        top_right_rotated = SkyCoord(
            np.nanmax(new_coords.Tx),
            np.nanmax(new_coords.Ty),
            frame=new_coords.frame,
        )

    offset_bottom_left = SkyCoord(
        bottom_left_rotated.Tx - expand[0],
        bottom_left_rotated.Ty - expand[1],
        frame=bottom_left_rotated.frame,
    )
    offset_top_right = SkyCoord(
        top_right_rotated.Tx + expand[0],
        top_right_rotated.Ty + expand[1],
        frame=top_right_rotated.frame,
    )

    return map2.submap(
        bottom_left=offset_bottom_left,
        top_right=offset_top_right,
    )


from pathlib import Path
from typing import List, Union, Dict
import pandas as pd

import re
import datetime

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
        -1: ("Warning", "\033[91m"),      # bright red
        #  0: ("Info", "\033[96m"),         # cyan
            0: ("Info", "\033[0m"),         # default
            1: ("Verbose", "\033[92m"),      # green
            2: ("Debug", "\033[90m"),        # faint gray
            3: ("Debug_Plot", "\033[90m"),
            4: ("Debug_Plot", "\033[90m"),
    }
    prefix, color = labels.get(level, (f"Level_{level}", "\033[0m"))
    reset = "\033[0m"
    print(f"{color}[{prefix}]{reset}", *args, **kwargs)

def get_closest_EUIFSI174_paths(
    date_ref: np.datetime64,
    interval: np.timedelta64,
    local_dir: Union[str, Path] = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0
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
    date_ref = np.datetime64(date_ref, "ms")
    half_int = np.timedelta64(interval, "ms")
    lower = date_ref - half_int
    upper = date_ref + half_int
    assert lower <= upper, "Interval must be non-negative"

    local_dir = Path(local_dir)
    _vprint(verbose, 1, f"Searching EUI paths around {date_ref} ± {interval}")

    # 1) Gather all days that might contain candidates
    days = _find_all_days(lower, upper, verbose)
    _vprint(verbose, 2, f"Checking {len(days)} days from {lower} to {upper}")

    # 2) Collect all FITS files in that window
    all_paths: List[Path] = []
    for day in days:
        _vprint(verbose, 3, f"  Grabbing data for {day}")
        all_paths.extend(_grab_EUI_data(day, local_dir, verbose))

    if not all_paths:
        _vprint(verbose, 1, "No EUI files found in the interval.")
        return []

    all_paths = split_eui_paths_by_mode(all_paths)['eui-fsi174-image']

    # 3) Extract timestamps from filenames
    def extract_dt(p: Path) -> Union[np.datetime64, None]:
        m = re.search(r"(\d{8})T(\d{6})", p.name)
        if not m:
            return None
        dt = datetime.datetime.strptime(m.group(0), "%Y%m%dT%H%M%S")
        return np.datetime64(dt, "ms")

    _vprint(verbose, 2, f"Extracting timestamps from {len(all_paths)} files")
    paths_arr = np.array(all_paths, dtype=object)
    dates = np.array([extract_dt(p) for p in paths_arr], dtype="datetime64[ms]")

    # 4) Compute absolute differences to reference, mask out None
    valid_mask = dates != np.datetime64("NaT", "ms")
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
    date_min: np.datetime64,
    date_max: np.datetime64,
    verbose: int = 0
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
    date_list = pd.date_range(date_min, end_plus_one, freq="D").to_list()
    _vprint(verbose, 2, f"Found {len(date_list)} days")
    return date_list
def _grab_EUI_data(
    date: np.datetime64,
    local_dir: str | Path = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0
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
    date = np.datetime64(date, "ms")
    date_dt = date.astype(datetime.datetime)
    year = date_dt.year
    month = date_dt.month
    day = date_dt.day

    local_dir = Path(local_dir)
    target_folder = local_dir / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"

    _vprint(verbose, 2, f"Looking for EUI data in {target_folder}")
    if target_folder.exists():
        eui_files = list(target_folder.glob("*.fits"))
        _vprint(verbose, 2, f"Found {len(eui_files)} EUI files for date {date_dt.date()}")
        return eui_files
    else:
        _vprint(verbose, -1, f"Directory does not exist: {target_folder}")
        return []
def get_EUI_paths(
    date_min: np.datetime64,
    date_max: np.datetime64,
    local_dir: str | Path = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0
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
    date_min = np.datetime64(date_min, "ms")
    date_max = np.datetime64(date_max, "ms")
    assert date_min <= date_max, "date_min must be less than or equal to date_max"

    local_dir = Path(local_dir)
    _vprint(verbose, 2, f"Getting EUI paths from {date_min} to {date_max} in {local_dir}")

    # 1) Gather all days in the range
    all_days = _find_all_days(date_min, date_max, verbose)

    # 2) Collect all FITS files across those days
    eui_paths: List[Path] = []
    for day in all_days:
        _vprint(verbose, 3, f"Grabbing EUI data for {day}")
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
            return np.datetime64(dt_obj)
        return None

    _vprint(verbose, 3, "Extracting timestamps from filenames")
    vectorized_extract = np.vectorize(_extract_datetime)
    eui_dates = vectorized_extract(eui_paths_sorted)

    # 5) Filter paths whose extracted dates lie within [date_min, date_max]
    mask = np.array(
        [(dt is not None and date_min <= dt <= date_max) for dt in eui_dates]
    )
    filtered_paths = list(np.array(eui_paths_sorted, dtype=object)[mask])

    _vprint(
        verbose,
        2,
        f"{len(filtered_paths)} files remain after filtering by date range"
    )
    return filtered_paths
def split_eui_paths_by_mode(
    list_paths: np.ndarray[Path],
    verbose: int = 0
) -> Dict[str, np.ndarray[Path]]:
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
    _vprint(verbose, 2, f"Splitting EUI data into different modes")
    _vprint(verbose, 2, f"Found {len(list_paths)} EUI paths")

    # Vectorized extraction of mode from each filename.
    # The regex 'solo_L[0-4]_' marks the start of mode,
    # and '_YYYYMMDDThhMMSS' marks the end.

    def _extract_mode(path: Path) -> str:
        name = path.name
        start_match = re.search(r"solo_L[0-4]_", name)
        end_match = re.search(r"_(\d{8})T(\d{6})", name)
        if start_match and end_match:
            return name[start_match.end(): end_match.start()]
        return ""

    modes = np.vectorize(_extract_mode)(list_paths)

    unique_modes = np.unique(modes)
    _vprint(verbose, 1, f"Found {len(unique_modes)} modes")
    _vprint(verbose, 2, f"Modes found:\n\t" + "\n\t".join(unique_modes))

    # Initialize dictionary with an empty list for each mode
    dict_paths: Dict[str, list[Path]] = {mode: [] for mode in unique_modes}

    # Assign each path to its corresponding mode
    for path, mode in zip(list_paths, modes):
        if mode:
            dict_paths[mode].append(path)

    # Convert lists to numpy arrays for consistency
    dict_paths_array: Dict[str, np.ndarray[Path]] = {
        mode: np.array(paths, dtype=object)
        for mode, paths in dict_paths.items()
    }
    return dict_paths_array



def get_closest_EUIFSI304_paths(
    date_ref: np.datetime64,
    interval: np.timedelta64,
    local_dir: Union[str, Path] = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2"),
    verbose: int = 0
) -> List[Path]:
    pass
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
    # Normalize to millisecond precision
    date_ref = np.datetime64(date_ref, "ms")
    half_int = np.timedelta64(interval, "ms")
    lower = date_ref - half_int
    upper = date_ref + half_int
    assert lower <= upper, "Interval must be non-negative"

    local_dir = Path(local_dir)
    _vprint(verbose, 1, f"Searching EUI paths around {date_ref} ± {interval}")

    # 1) Gather all days that might contain candidates
    days = _find_all_days(lower, upper, verbose)
    _vprint(verbose, 2, f"Checking {len(days)} days from {lower} to {upper}")

    # 2) Collect all FITS files in that window
    all_paths: List[Path] = []
    for day in days:
        _vprint(verbose, 3, f"  Grabbing data for {day}")
        all_paths.extend(_grab_EUI_data(day, local_dir, verbose))

    if not all_paths:
        _vprint(verbose, 1, "No EUI files found in the interval.")
        return []

    all_paths = split_eui_paths_by_mode(all_paths)['eui-fsi304-image']

    # 3) Extract timestamps from filenames
    def extract_dt(p: Path) -> Union[np.datetime64, None]:
        m = re.search(r"(\d{8})T(\d{6})", p.name)
        if not m:
            return None
        dt = datetime.datetime.strptime(m.group(0), "%Y%m%dT%H%M%S")
        return np.datetime64(dt, "ms")

    _vprint(verbose, 2, f"Extracting timestamps from {len(all_paths)} files")
    paths_arr = np.array(all_paths, dtype=object)
    dates = np.array([extract_dt(p) for p in paths_arr], dtype="datetime64[ms]")

    # 4) Compute absolute differences to reference, mask out None
    valid_mask = dates != np.datetime64("NaT", "ms")
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


def _extract_map_time(entry: Union[GenericMap, str, Path], verbose: int = 0) -> np.datetime64:
    if isinstance(entry, GenericMap):
        t = np.datetime64(entry.date.isot)
        _vprint(verbose, 3, f"Map time (GenericMap): {t}")
        return t
    hdr = fits.getheader(entry, ext=1)
    date_key = hdr.get("DATE-AVG") or hdr.get("DATE-OBS") or hdr.get("DATE_BEG")
    if date_key is None:
        raise ValueError(f"No DATE-* keyword found in {entry}")
    t = np.datetime64(date_key)
    _vprint(verbose, 3, f"Map time ({entry}): {t}")
    return t



def interpol2d(image, x, y, fill, order, dst=None):
    """"
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

    map_coordinates(image,
                    coords,
                    order=order,
                    mode='constant',
                    cval=fill, output=dst.ravel(), prefilter=False)
    if return_:
        return dst

# build synthetic FSI raster.
def build_synthetic_raster_from_maps(
    spice_map: GenericMap,
    fsi_maps: Sequence[Union[GenericMap, str, Path]],
    step_times: Sequence[np.datetime64] | None = None,
    threshold_time: np.timedelta64 | None = None,
    order: int = 2,
    verbose: int = 0,
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
    step_times : Sequence[np.datetime64], optional
        Time for each SPICE slit position (length = NAXIS1). Defaults to the
        SPICE map time for all positions.
    threshold_time : np.timedelta64, optional
        Maximum allowed |Δt| between slit time and selected FSI time.
    order : int, optional
        Interpolation order for ``interpol2d`` (default 2).
    """
    if not fsi_maps:
        raise ValueError("fsi_maps must be non-empty")

    _vprint(verbose, 1, "Building synthetic raster from FSI maps")

    ny, nx = spice_map.data.shape

    fsi_entries: list[Union[GenericMap, Path, str]] = []
    fsi_times_list: list[np.datetime64] = []
    for entry in fsi_maps:
        fsi_entries.append(entry)
        fsi_times_list.append(_extract_map_time(entry, verbose=verbose))  
    fsi_times: NDArray[np.datetime64] = np.array(fsi_times_list)
    
    def meta_to_header(meta):
      hdr = fits.Header()
      for k, v in meta.items():
        if len(k) > 8 or k.startswith("_"):
          continue  # skip non-FITS-style keys
        if isinstance(v, str):
          v = v.replace("\n", " ").encode("ascii", "ignore").decode("ascii")
        hdr[k] = v
      return hdr
    WCS3D = WCS(meta_to_header(spice_map.meta))
    
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    coords_spice: SkyCoord
    coords_spice, time_matrix = WCS3D.pixel_to_world(xx*u.pix, yy*u.pix, 0)
    step_times: NDArray[np.datetime64] = (time_matrix.to_datetime()).astype("datetime64")[0]
    _vprint(verbose, 2, "Computed step_times from WCS3D time axis")
    
    data_composed = np.full((ny, nx), np.nan, dtype=float)
    fsi_cache: dict[int, GenericMap] = {}
    for i in range(nx):
        idx = _nearest_imager_index(step_times[i], fsi_times, threshold_time, verbose=verbose)
        entry = fsi_entries[idx]
        if isinstance(entry, GenericMap):
            fsi_map = entry
        else:
            if idx not in fsi_cache:
                _vprint(verbose, 3, f"Loading FSI map idx {idx} from {entry}")
                fsi_cache[idx] = Map(entry)
            fsi_map = fsi_cache[idx]

        coords_col = SkyCoord(
            coords_spice.Tx[:, i],
            coords_spice.Ty[:, i],
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

    hdr = spice_map.meta.copy()
    hdr["SYNRASTR"] = "FSI->SPICE synthetic raster"
    hdr["SRCIMGS"] = len(fsi_maps)
    hdr.setdefault("DATE-AVG", spice_map.date.isot)

    plot_settings = getattr(spice_map, "plot_settings", None)
    plot_settings["norm"] = normit(data_composed)
    
    return GenericMap(data_composed, hdr, plot_settings=plot_settings)

# Parallel version # We have abandoned multiprocessing due to overhead issues.
def build_synthetic_raster_from_maps_parallel(
    spice_map: GenericMap,
    fsi_maps: Sequence[Union[GenericMap, str, Path]],
    step_times: Sequence[np.datetime64] | None = None,
    threshold_time: np.timedelta64 | None = None,
    order: int = 2,
    n_jobs: int = 1,
    verbose: int = 0,
) -> GenericMap:
    """
    Parallel version of build_synthetic_raster_from_maps.

    Splits columns (x dimension) into batches and processes them in threads.
    Uses nearest-in-time FSI map per column; results are stitched back together.
    """
    if not fsi_maps:
        raise ValueError("fsi_maps must be non-empty")

    _vprint(verbose, 1, f"Building synthetic raster in parallel with n_jobs={n_jobs}")

    ny, nx = spice_map.data.shape

    fsi_entries: list[Union[GenericMap, Path, str]] = []
    fsi_times_list: list[np.datetime64] = []
    for entry in fsi_maps:
        fsi_entries.append(entry)
        fsi_times_list.append(_extract_map_time(entry, verbose=verbose))
    fsi_times: NDArray[np.datetime64] = np.array(fsi_times_list)

    def meta_to_header(meta):
        hdr = fits.Header()
        for k, v in meta.items():
            if len(k) > 8 or k.startswith("_"):
                continue
            if isinstance(v, str):
                v = v.replace("\n", " ").encode("ascii", "ignore").decode("ascii")
            hdr[k] = v
        return hdr

    WCS3D = WCS(meta_to_header(spice_map.meta))

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    coords_spice, time_matrix = WCS3D.pixel_to_world(xx * u.pix, yy * u.pix, 0)
    step_times: NDArray[np.datetime64] = (time_matrix.to_datetime()).astype("datetime64")[0]
    _vprint(verbose, 2, "Computed step_times from WCS3D time axis")

    data_composed = np.full((ny, nx), np.nan, dtype=float)
    fsi_cache: dict[int, GenericMap] = {}
    cache_lock = Lock()

    def process_range(x_start: int, x_end: int) -> tuple[int, int, np.ndarray]:
        local = np.full((ny, x_end - x_start), np.nan, dtype=float)
        for local_j, i in enumerate(range(x_start, x_end)):
            idx = _nearest_imager_index(step_times[i], fsi_times, threshold_time, verbose=verbose)
            entry = fsi_entries[idx]
            with cache_lock:
                if isinstance(entry, GenericMap):
                    fsi_map = entry
                else:
                    if idx not in fsi_cache:
                        _vprint(verbose, 3, f"Loading FSI map idx {idx} from {entry}")
                        fsi_cache[idx] = Map(entry)
                    fsi_map = fsi_cache[idx]

            coords_col = SkyCoord(
                coords_spice.Tx[:, i],
                coords_spice.Ty[:, i],
                frame=coords_spice.frame,
            )
            x_fsi, y_fsi = fsi_map.world_to_pixel(coords_col)
            local[:, local_j] = interpol2d(
                fsi_map.data,
                x=np.asarray(x_fsi),
                y=np.asarray(y_fsi),
                order=order,
                fill=np.nan,
            )
        return x_start, x_end, local

    if n_jobs is None or n_jobs < 2:
        _vprint(verbose, 1, "n_jobs < 2; falling back to serial processing")
        x_start, x_end, local = process_range(0, nx)
        data_composed[:, x_start:x_end] = local
    else:
        chunk = max(1, int(np.ceil(nx / n_jobs)))
        ranges = [(s, min(nx, s + chunk)) for s in range(0, nx, chunk)]
        _vprint(verbose, 2, f"Processing {len(ranges)} chunks of size ≤ {chunk}")
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(process_range, s, e) for s, e in ranges]
            for fut in futures:
                x_start, x_end, local = fut.result()
                data_composed[:, x_start:x_end] = local

    hdr = spice_map.meta.copy()
    hdr["SYNRASTR"] = "FSI->SPICE synthetic raster (parallel)"
    hdr["SRCIMGS"] = len(fsi_maps)
    hdr.setdefault("DATE-AVG", spice_map.date.isot)

    plot_settings = getattr(spice_map, "plot_settings", None)
    plot_settings["norm"] = normit(data_composed)

    return GenericMap(data_composed, hdr, plot_settings=plot_settings)

def _process_chunk_multiproc(args: tuple) -> tuple[int, int, np.ndarray]:
    (
        x_start,
        x_end,
        tx,
        ty,
        frame,
        step_times,
        fsi_times,
        fsi_entries,
        threshold_time,
        order,
    ) = args

    ny = tx.shape[0]
    local = np.full((ny, x_end - x_start), np.nan, dtype=float)
    for local_j, i in enumerate(range(x_start, x_end)):
        idx = _nearest_imager_index(step_times[i], fsi_times, threshold_time, verbose=0)
        entry = fsi_entries[idx]
        fsi_map = Map(entry)
        coords_col = SkyCoord(tx[:, i], ty[:, i], frame=frame)
        x_fsi, y_fsi = fsi_map.world_to_pixel(coords_col)
        local[:, local_j] = interpol2d(
            fsi_map.data,
            x=np.asarray(x_fsi),
            y=np.asarray(y_fsi),
            order=order,
            fill=np.nan,
        )
    return x_start, x_end, local

def build_synthetic_raster_multiproc(
    spice_map: GenericMap,
    fsi_maps: Sequence[Union[str, Path]],
    step_times: Sequence[np.datetime64] | None = None,
    threshold_time: np.timedelta64 | None = None,
    order: int = 2,
    n_jobs: int = 1,
    verbose: int = 0,
) -> GenericMap:
    """
    Multiprocessing version of build_synthetic_raster_from_maps.

    Notes: fsi_maps must be file paths (GenericMap is not pickled across processes).
    Each process loads needed maps independently; expect higher memory use.
    """
    if not fsi_maps:
        raise ValueError("fsi_maps must be non-empty")

    if any(isinstance(e, GenericMap) for e in fsi_maps):
        raise ValueError("fsi_maps must be file paths for multiprocessing.")

    _vprint(verbose, 1, f"Building synthetic raster with ProcessPool (n_jobs={n_jobs})")

    ny, nx = spice_map.data.shape

    fsi_entries: list[Union[Path, str]] = [Path(e) for e in fsi_maps]
    fsi_times_list: list[np.datetime64] = [
        _extract_map_time(e, verbose=verbose) for e in fsi_entries
    ]
    fsi_times: NDArray[np.datetime64] = np.array(fsi_times_list)

    def meta_to_header(meta):
        hdr = fits.Header()
        for k, v in meta.items():
            if len(k) > 8 or k.startswith("_"):
                continue
            if isinstance(v, str):
                v = v.replace("\n", " ").encode("ascii", "ignore").decode("ascii")
            hdr[k] = v
        return hdr

    WCS3D = WCS(meta_to_header(spice_map.meta))

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    coords_spice, time_matrix = WCS3D.pixel_to_world(xx * u.pix, yy * u.pix, 0)
    step_times_array: NDArray[np.datetime64] = (time_matrix.to_datetime()).astype("datetime64")[0]
    _vprint(verbose, 2, "Computed step_times from WCS3D time axis")

    tx = coords_spice.Tx
    ty = coords_spice.Ty
    frame = coords_spice.frame

    data_composed = np.full((ny, nx), np.nan, dtype=float)

    if n_jobs is None or n_jobs < 2:
        _vprint(verbose, 1, "n_jobs < 2; falling back to serial processing")
        x_start, x_end, local = _process_chunk_multiproc(
            (0, nx, tx, ty, frame, step_times_array, fsi_times, fsi_entries, threshold_time, order)
        )
        data_composed[:, x_start:x_end] = local
    else:
        chunk = max(1, int(np.ceil(nx / n_jobs)))
        ranges = [(s, min(nx, s + chunk)) for s in range(0, nx, chunk)]
        _vprint(verbose, 2, f"Processing {len(ranges)} chunks of size ≤ {chunk} with processes")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _process_chunk_multiproc,
                    (s, e, tx, ty, frame, step_times_array, fsi_times, fsi_entries, threshold_time, order),
                )
                for s, e in ranges
            ]
            for fut in futures:
                x_start, x_end, local = fut.result()
                data_composed[:, x_start:x_end] = local

    hdr = spice_map.meta.copy()
    hdr["SYNRASTR"] = "FSI->SPICE synthetic raster (multiproc)"
    hdr["SRCIMGS"] = len(fsi_maps)
    hdr.setdefault("DATE-AVG", spice_map.date.isot)

    plot_settings = getattr(spice_map, "plot_settings", None)
    plot_settings["norm"] = normit(data_composed)

    return GenericMap(data_composed, hdr, plot_settings=plot_settings)



