"""Preprocessing helpers for preparing SunPy maps prior to cross-correlation.

This module contains routines that expand the target field of view and apply
binning-friendly smoothing while preserving World Coordinate System (WCS)
metadata. They are shared by multiple alignment pipelines so they live outside
of the main `Coaligner` implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Union, cast

import numpy as np
import sunpy.map
from scipy.ndimage import uniform_filter
from sunpy.map import GenericMap


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


def no_nan_uniform_filter(
    data: np.ndarray,
    remove_percentile: float = 100,
    *args,
    **kwargs,
) -> np.ndarray:
  """Apply uniform filter while masking outliers and preserving NaN regions.
  
  Wraps scipy.ndimage.uniform_filter with preprocessing to handle NaN values
  and optionally remove extreme outliers before smoothing.
  
  Parameters
  ----------
  data : np.ndarray
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
  np.ndarray
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
  data_cleaned: np.ndarray = np.where(data > data_percentile, np.nan, data)
  nan_mask: np.ndarray = np.isnan(data_cleaned)
  data_filled: np.ndarray = np.where(nan_mask, 0.0, data_cleaned)
  filtered_data: np.ndarray = uniform_filter(data_filled, *args, **kwargs)
  filtered_data[nan_mask] = np.nan
  return filtered_data
