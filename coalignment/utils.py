"""Utility functions for coalignment processes."""

from typing import Tuple

import numpy as np
import astropy.units as u
from sunpy.map import GenericMap
# from scipy.ndimage import uniform_filter
import sunpy.map.maputils


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
