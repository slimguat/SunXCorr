"""Coalignment package - flexible tree-based coalignment framework."""

from .base_node import CoalignmentNode
from .orchestrator import Orchestrator
from .single_map_process import SingleMapProcess
from .synthetic_raster_process import SyntheticRasterProcess
from .process_result import ProcessResult

__all__ = [
    'CoalignmentNode',
    'Orchestrator',
    'SingleMapProcess',
    'SyntheticRasterProcess',
    'ProcessResult',
]

__version__ = '2.0.0'
