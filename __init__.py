"""Solar Coalignment Library.

A Python library for high-precision coalignment of radince maps raster scan map and normal single exposure images using cross-correlation optimization. 
This packag is proven for cross-correlating SOLAR ORBITER/SPICE with SOLAR ORBITER/FSI but is designed to be flexible for other solar image alignment tasks.
"""

__version__ = '1.0.0'
__author__ = 'Slimane Mzerguat'
# Optional lazy imports: keep package import lightweight for test collection
try:
    from .coaligner import Coaligner  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    Coaligner = None  # type: ignore

try:
    from .slimfunc_correlation_effort import (
        correlation_for_params,
        make_corrected_wcs_map,
        reproject_map_to_reference,
    )
except Exception:  # pragma: no cover
    correlation_for_params = None  # type: ignore
    make_corrected_wcs_map = None  # type: ignore
    reproject_map_to_reference = None  # type: ignore

try:
    from .help_funcs import (
        build_synthetic_raster_from_maps,
        get_EUI_paths,
        get_closest_EUIFSI174_paths,
    )
except Exception:  # pragma: no cover
    build_synthetic_raster_from_maps = None  # type: ignore
    get_EUI_paths = None  # type: ignore
    get_closest_EUIFSI174_paths = None  # type: ignore

__all__ = [
    name for name in (
        'Coaligner',
        'correlation_for_params',
        'make_corrected_wcs_map',
        'reproject_map_to_reference',
        'build_synthetic_raster_from_maps',
        'get_EUI_paths',
        'get_closest_EUIFSI174_paths',
    )
    if globals().get(name) is not None
]
