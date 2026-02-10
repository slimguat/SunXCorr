"""SunXCorr package.

Public API re-exports for the `sunxcorr` package. This initializer
exposes the main classes used by downstream code.
"""

from __future__ import annotations

from .base_node import CoalignmentNode  # noqa: F401
from .orchestrator import Orchestrator  # noqa: F401
from .process_result import ProcessResult  # noqa: F401
from .single_map_process import SingleMapProcess  # noqa: F401
from .synthetic_raster_process import SyntheticRasterProcess  # noqa: F401

__all__ = [
    "CoalignmentNode",
    "Orchestrator",
    "SingleMapProcess",
    "SyntheticRasterProcess",
    "ProcessResult",
]

__version__ = "2.0.0"
__author__ = "Slimane MZERGUAT"
