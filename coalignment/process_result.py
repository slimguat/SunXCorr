"""Data structures for coalignment process results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import astropy.units as u
from sunpy.map import GenericMap


@dataclass
class ProcessResult:
    """Result from a single coalignment process execution.
    
    Attributes
    ----------
    process_id : str
        Unique identifier for the process instance
    process_name : str
        Human-readable name of the process
    input_map : GenericMap
        Map that was processed
    output_map : GenericMap
        Corrected map produced by the process
    shift_arcsec : Tuple[u.Quantity, u.Quantity]
        Shift in arcseconds (dx, dy)
    shift_pixels : Tuple[float, float]
        Shift in pixels (dx, dy)
    scale_factors : Tuple[float, float]
        Scale factors (sx, sy) - (1.0, 1.0) if no scaling
    correlation_peak : float
        Best correlation achieved
    search_space_explored : int
        Number of candidates evaluated
    iteration_count : int
        Number of optimization iterations
    execution_time : float
        Elapsed time in seconds
    debug_pdf_path : Path | None
        Path to debug PDF (if generated)
    animation_path : Path | None
        Path to comparison GIF (if generated)
    """
    
    process_id: str
    process_name: str
    input_map: GenericMap
    output_map: GenericMap
    shift_arcsec: Tuple[u.Quantity, u.Quantity]
    shift_pixels: Tuple[float, float]
    scale_factors: Tuple[float, float]
    correlation_peak: float
    search_space_explored: int = 0
    iteration_count: int = 0
    execution_time: float = 0.0
    debug_pdf_path: Path | None = None
    animation_path: Path | None = None
    reference_reprojected: GenericMap | None = None  # Store reprojected reference for potential debugging
