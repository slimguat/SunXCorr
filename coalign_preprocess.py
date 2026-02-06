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
  """Pad or crop `smap` symmetrically while keeping WCS metadata consistent."""
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
  """Apply `uniform_filter` while masking values above `remove_percentile`."""
  data_percentile: float = cast(float, np.nanpercentile(data, remove_percentile))
  data_cleaned: np.ndarray = np.where(data > data_percentile, np.nan, data)
  nan_mask: np.ndarray = np.isnan(data_cleaned)
  data_filled: np.ndarray = np.where(nan_mask, 0.0, data_cleaned)
  filtered_data: np.ndarray = uniform_filter(data_filled, *args, **kwargs)
  filtered_data[nan_mask] = np.nan
  return filtered_data
