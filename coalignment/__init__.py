"""`coalignment` package shim.

Re-export public symbols from the `sunxcorr` package so existing imports
continue to work while `sunxcorr` is adopted as the canonical package name.
"""

from sunxcorr.base_node import CoalignmentNode  # noqa: F401
from sunxcorr.orchestrator import Orchestrator  # noqa: F401
from sunxcorr.process_result import ProcessResult  # noqa: F401
from sunxcorr.single_map_process import SingleMapProcess  # noqa: F401
from sunxcorr.synthetic_raster_process import SyntheticRasterProcess  # noqa: F401

__all__ = [
    "CoalignmentNode",
    "Orchestrator",
    "SingleMapProcess",
    "SyntheticRasterProcess",
    "ProcessResult",
]

__version__ = "2.0.0"
