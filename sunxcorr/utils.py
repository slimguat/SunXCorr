"""Utility functions for coalignment processes."""

from typing import Tuple

import numpy as np
import astropy.units as u
from sunpy.map import GenericMap
# from scipy.ndimage import uniform_filter
import sunpy.map.maputils
from reproject import reproject_interp
from typing import Any, Tuple, Union, cast


# ==============================================================
# Helper: build synthetic raster from FSI maps
# ==============================================================
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
    
    WCS3D = WCS(meta_to_header(spice_map.meta))

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    coords_spice, time_matrix = _pixel_world_with_optional_time(WCS3D, xx, yy)
    step_times: NDArray[np.datetime64] = _coerce_step_times(
        time_matrix,
        nx,
        np.datetime64(spice_map.date.isot),
    )
    _vprint(verbose, 2, "Computed step_times from WCS metadata")
    
    data_composed = np.full((ny, nx), np.nan, dtype=float)
    fsi_cache: dict[int, GenericMap] = {}
    for i in range(nx):
        idx = _nearest_imager_index(step_times[i], fsi_times, threshold_time, verbose=verbose)
        entry = fsi_entries[idx]
        if isinstance(entry, GenericMap):
            fsi_map = entry
        else:
            if idx not in fsi_cache:
                _vprint(verbose, 2, f"Loading FSI map idx {idx} from {entry.name}")
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
        input_map,           # or (input_map.data, input_map.wcs)
        target_wcs,
        shape_out=target_shape,
        order=order,
    )

    # Mask outside-footprint pixels as NaN
    reprojected_data = np.where(footprint > 0, reprojected_data, np.nan)

    new_meta = cast(dict, ref_map.meta).copy()
    new_map = sunpy.map.Map(reprojected_data, new_meta, plot_settings=input_map.plot_settings)

    return cast(GenericMap, new_map)


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
    cdelt1 = abs(map_obj.meta.get('CDELT1', 1.0))
    cdelt2 = abs(map_obj.meta.get('CDELT2', 1.0))
    cunit1 = map_obj.meta.get('CUNIT1', 'arcsec')
    cunit2 = map_obj.meta.get('CUNIT2', 'arcsec')
    pixel_scale_x = (cdelt1 * u.Unit(cunit1)).to(u.arcsec)
    pixel_scale_y = (cdelt2 * u.Unit(cunit2)).to(u.arcsec)
    return pixel_scale_x, pixel_scale_y


def arcsec_to_pixels(value_arcsec: u.Quantity | Tuple[u.Quantity, u.Quantity], map_obj: GenericMap) -> Tuple[int, int] | int:
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
        pixels_x = int((value_arcsec / pixel_scale_x).to(u.dimensionless_unscaled).value)
        pixels_y = int((value_arcsec / pixel_scale_y).to(u.dimensionless_unscaled).value)
        return pixels_x, pixels_y


def pixels_to_arcsec(value_pixels: float | Tuple[float, float], map_obj: GenericMap) -> Tuple[u.Quantity, u.Quantity] | u.Quantity:
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
    from help_funcs import no_nan_uniform_filter
    smoothed_data = no_nan_uniform_filter(
        np.asarray(map_obj.data),
        remove_percentile=99,
        size=(bin_y, bin_x)  # Note: numpy is (rows, cols) = (Y, X)
    )
    binned_data = smoothed_data[::bin_y, ::bin_x]
    
    # Update metadata for binned WCS
    new_meta = map_obj.meta.copy()
    new_meta['CDELT1'] = map_obj.meta['CDELT1'] * bin_x
    new_meta['CDELT2'] = map_obj.meta['CDELT2'] * bin_y
    new_meta['CRPIX1'] = map_obj.meta['CRPIX1'] / bin_x
    new_meta['CRPIX2'] = map_obj.meta['CRPIX2'] / bin_y
    new_meta['NAXIS1'] = binned_data.shape[1]
    new_meta['NAXIS2'] = binned_data.shape[0]
    
    from sunpy.map import Map
    # Preserve plot_settings (norm, cmap, etc.) from original map
    plot_settings = getattr(map_obj, 'plot_settings', None)
    return Map(binned_data, new_meta, plot_settings=plot_settings)


def apply_shift_to_map(
    map_obj: GenericMap,
    shift_x_arcsec: u.Quantity,
    shift_y_arcsec: u.Quantity
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
    from slimfunc_correlation_effort import make_corrected_wcs_map
    
    # Convert shifts to pixels using separate pixel scales
    pixel_scale_x, pixel_scale_y = get_pixel_scale_quantity(map_obj)
    dx = (shift_x_arcsec / pixel_scale_x).to(u.dimensionless_unscaled).value
    dy = (shift_y_arcsec / pixel_scale_y).to(u.dimensionless_unscaled).value
    
    # Apply shift via WCS correction
    best_params = {
        'dx': dx,
        'dy': dy,
        'squeeze_x': 1.0,
        'squeeze_y': 1.0
    }
    corrected_map = make_corrected_wcs_map(map_obj, best_params)
    
    return corrected_map


def apply_shift_and_scale_to_map(
    map_obj: GenericMap,
    shift_x_arcsec: u.Quantity,
    shift_y_arcsec: u.Quantity,
    scale_x: float,
    scale_y: float
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
    from slimfunc_correlation_effort import make_corrected_wcs_map
    
    # Convert shifts to pixels using separate pixel scales
    pixel_scale_x, pixel_scale_y = get_pixel_scale_quantity(map_obj)
    dx = (shift_x_arcsec / pixel_scale_x).to(u.dimensionless_unscaled).value
    dy = (shift_y_arcsec / pixel_scale_y).to(u.dimensionless_unscaled).value
    
    # Apply transformation
    best_params = {
        'dx': dx,
        'dy': dy,
        'squeeze_x': scale_x,
        'squeeze_y': scale_y
    }
    corrected_map = make_corrected_wcs_map(map_obj, best_params)
    
    return corrected_map
