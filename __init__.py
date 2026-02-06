"""Solar Coalignment Library.

A Python library for high-precision coalignment of solar spectroscopic 
rasters with photometric reference images using cross-correlation optimization.
"""

__version__ = '1.0.0'
__author__ = 'Salim Zergua'

from .coaligner import Coaligner
from .slimfunc_correlation_effort import (
    correlation_for_params,
    make_corrected_wcs_map,
    reproject_map_to_reference,
)
from .help_funcs import (
    build_synthetic_raster_from_maps,
    get_EUI_paths,
    get_closest_EUIFSI174_paths,
)

__all__ = [
    'Coaligner',
    'correlation_for_params',
    'make_corrected_wcs_map',
    'reproject_map_to_reference',
    'build_synthetic_raster_from_maps',
    'get_EUI_paths',
    'get_closest_EUIFSI174_paths',
]
