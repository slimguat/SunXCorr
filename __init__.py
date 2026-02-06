"""Solar Coalignment Library.

A Python library for high-precision coalignment of radince maps raster scan map and normal single exposure images using cross-correlation optimization. 
This packag is proven for cross-correlating SOLAR ORBITER/SPICE with SOLAR ORBITER/FSI but is designed to be flexible for other solar image alignment tasks.
"""

__version__ = '1.0.0'
__author__ = 'Slimane Mzerguat'

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
