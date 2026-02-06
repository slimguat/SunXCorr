"""High-level coalignment orchestrator for SPICE/SPICE-like SunPy maps."""

from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple, Union, cast
from collections import defaultdict
from itertools import count

from multiprocessing import Manager, Process, Queue
from multiprocessing.managers import BaseManager
from pathlib import Path
import heapq

from matplotlib.backends.backend_pdf import PdfPages
import astropy.units as u
import math
import numpy as np
import os
import sunpy.map
from sunpy.map import GenericMap, Map

from coalign_debug import DebugPlotContext, create_debug_context
from coalign_helpers import (
  build_shift_structures,
  clamp_point,
  clamp_point_nd,
  compute_gradient_step,
  compute_gradient_step_nd,
  first_unvisited,
  gather_neighbors,
  phase_label,
  point_in_bounds,
  point_in_bounds_nd,
  update_phase_state,
)
from coalign_preprocess import enlarge_map_by_padding, no_nan_uniform_filter
from coalign_workers import (
  corr_worker_loop,
  make_worker_payload_key,
  store_shared_payload,
)
from help_funcs import (
  _vprint,
  build_synthetic_raster_from_maps,
  get_closest_EUIFSI174_paths,
  get_EUI_paths,
)
from slimfunc_correlation_effort import (
  correlation_for_params,
  make_corrected_wcs_map,
  reproject_map_to_reference,
)


class Coaligner:
  """Three-phase solar image coalignment pipeline with adaptive optimization.

  The Coaligner implements a sophisticated multi-phase approach to align solar images
  (typically SPICE rasters with EUI/FSI reference maps) through progressive refinement:
  
  1. **Binning Phase**: Coarse alignment using spatially binned maps for speed
  2. **One-Map Phase**: Fine alignment using full-resolution single reference map  
  3. **Synthetic Raster Phase**: Optimal alignment by generating synthetic rasters
     from multiple FSI images spanning the raster observation time
  
  Each phase uses a discrete gradient-based stochastic optimizer with persistent
  worker processes for efficient parallel correlation evaluation.

  Key Features
  ------------
  - Persistent worker pool for parallel correlation computation
  - Adaptive debug plotting (2 or 3 axes depending on scale optimization)
  - Blink animation GIFs showing before/after correction per phase
  - Automatic synthetic raster generation from FSI image sequences
  - Plateau-based step size adaptation with graceful convergence

  Attributes
  ----------
  map_to_coalign : GenericMap
      The target map to be aligned (typically SPICE raster)
  reference_map : GenericMap
      Single reference map for phases 1 and 2 (typically EUI FSI 174)
  list_of_reference_maps : list of GenericMap, optional
      Multiple FSI images for synthetic raster generation (phase 3)
  procedures : dict
      Tracking which phases have been executed
  results : dict
      Results dictionary for each phase containing best parameters and correlation
  best_params : dict, optional
      Final WCS parameters after all executed phases
  
  Notes
  -----
  The coalignment is applied progressively: each phase refines the results from
  the previous phase. The workflow typically follows:
  
  1. Initialize Coaligner with SPICE raster and FSI reference
  2. Call `run_binning_xcorr()` for coarse alignment
  3. Call `run_one_map_xcorr()` for refined alignment
  4. Call `run_synthetic_raster_xcorr()` for final high-precision alignment
  5. Access corrected WCS parameters via `best_params` attribute
  
  The optimizer uses correlation as the objective function, with adaptive step
  sizes that shrink when correlation plateaus. This ensures convergence while
  avoiding premature stopping in noisy correlation landscapes.
  
  Examples
  --------
  >>> from coaligner import Coaligner
  >>> import sunpy.map
  >>> 
  >>> # Load maps
  >>> spice_map = sunpy.map.Map('spice_raster.fits')
  >>> fsi_map = sunpy.map.Map('eui_fsi_174.fits')
  >>> 
  >>> # Initialize coaligner  
  >>> aligner = Coaligner(
  ...     map_to_coalign=spice_map,
  ...     reference_map=fsi_map,
  ...     verbose=1,
  ...     n_jobs=8
  ... )
  >>> 
  >>> # Run three-phase alignment
  >>> aligner.run_binning_xcorr()
  >>> aligner.run_one_map_xcorr()
  >>> aligner.run_synthetic_raster_xcorr()
  >>> 
  >>> # Access corrected WCS
  >>> corrected_wcs = aligner.best_params
  >>> print(f"Final correlation: {aligner.results['synthetic_raster']['best_corr']}")

  See Also
  --------
  coalign_debug.DebugPlotContext : Debug visualization context
  coalign_helpers.compute_gradient_step_nd : Gradient computation for optimizer
  coalign_workers.corr_worker_loop : Persistent correlation worker
  """

  def __init__(
    self,
    map_to_coalign: GenericMap,
    reference_map: GenericMap,
    list_of_reference_maps: Iterable[GenericMap] | None = None,
    verbose: int = 0,
    n_jobs: int = cast(int, os.cpu_count()) - 1,
    n_neighbors: int | None = None,
  ) -> None:
    """Initialize the coalignment pipeline with input maps and settings.

    Parameters
    ----------
    map_to_coalign : GenericMap
        Target map to be coaligned, typically a SPICE raster. Must have valid
        WCS metadata (CRVAL, CDELT, CRPIX, PC matrix).
    reference_map : GenericMap
        Reference map used for phases 1 and 2, typically EUI FSI 174 Angstrom.
        Should have compatible field of view with the target map.
    list_of_reference_maps : Iterable of GenericMap, optional
        Sequence of FSI images for synthetic raster generation (phase 3). 
        Images should span the temporal range of the SPICE raster observation.
        If None, phase 3 will attempt to auto-load FSI images from disk.
    verbose : int, default=0
        Verbosity level:
        - 0: Silent (errors only)
        - 1: Phase summaries
        - 2: Iteration details  
        - 3: Debug information
    n_jobs : int, default=cpu_count()-1
        Number of parallel worker processes for correlation evaluation.
        Workers persist across all phases for efficiency.
    n_neighbors : int, optional
        Number of neighbor candidates to evaluate per iteration. If None,
        defaults to `n_jobs * 10`. Higher values improve exploration but
        increase iteration time.

    Raises
    ------
    ValueError
        If input maps lack required WCS metadata.
    TypeError
        If map_to_coalign or reference_map are not GenericMap instances.

    Notes
    -----
    Upon initialization, the coaligner:
    - Wraps maps with sunpy.map.Map() for consistency
    - Allocates internal data structures for three phases
    - Sets default optimization parameters (can be customized via attributes)
    - Does NOT start worker processes (started on first phase execution)
    
    The default parameters are tuned for SPICE/FSI coalignment but can be
    adjusted by modifying the `xcorr_binned_kwargs`, `xcorr_one_map_kwargs`,
    and `synthetic_kwargs` attributes before running phases.

    Examples
    --------
    Basic initialization:
    
    >>> aligner = Coaligner(
    ...     map_to_coalign=spice_map,
    ...     reference_map=fsi_map,
    ...     verbose=1
    ... )
    
    Customize optimization parameters:
    
    >>> aligner = Coaligner(spice_map, fsi_map, n_jobs=16)
    >>> aligner.xcorr_binned_kwargs['shift_range'] = (30, 30)  # wider search
    >>> aligner.synthetic_kwargs['plateau_iters'] = 5  # more patient convergence
    """
    self.map_to_coalign = cast(GenericMap, sunpy.map.Map(map_to_coalign))
    self.reference_map = cast(GenericMap, sunpy.map.Map(reference_map))
    self.list_of_reference_maps: List[GenericMap | Path | str] | None = (
      list(list_of_reference_maps) if list_of_reference_maps else None
    )
    self.n_jobs = n_jobs
    self.procedures: dict[Literal['binning', 'one_map', 'synthetic_raster'], bool] = {
      'binning': False,
      'one_map': False,
      'synthetic_raster': False,
    }
    self.results: dict[Literal['binning', 'one_map', 'synthetic_raster'], dict[Any, Any] | None] = {
      "binning": None,
      "one_map": None,
      "synthetic_raster": None,
    }
    self.best_params: dict[
      Literal['CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'],
      float,
    ] | None = None
    self.target_FOV_expension_arcsec: u.Quantity = 700 * u.arcsec
    self.bin_kernel_arcsec: u.Quantity = 50 * u.arcsec
    self.bin_kernel: Tuple[int, int] | None = None
    self.verbose = verbose
    n_neighbors = (n_neighbors if n_neighbors is not None else self.n_jobs * 10)
    self.xcorr_binned_kwargs = {
      "dx0": 0.0,
      "dy0": 0.0,
      "shift_range": (20, 20),
      "max_corr": 0.7,
      "scale_x": 1.0,
      "scale_y": 1.0,
      "plateau_iters": 3,
      "target_data": None,
      "reference_data": None,
      "binned_crpix1": None,
      "binned_crpix2": None,
      "extended_target_map": None,
      "reprojected_reference_on_extended_target_map": None,
      "n_neighbors": n_neighbors,
      "pad_x": 0,
      "pad_y": 0,
      "kernel": None,
    }
    self.xcorr_one_map_kwargs = {
      "dx0": None,
      "dy0": None,
      "shift_range": None,
      "max_corr": -1.0,
      "scale_x": 1.0,
      "scale_y": 1.0,
      "corr_atol": 1e-4,
      "corr_rtol": 1e-3,
      "plateau_iters": 3,
      "target_data": None,
      "reference_data": None,
      "crpix1": None,
      "crpix2": None,
      "extended_target_map": None,
      "reprojected_reference_on_extended_target_map": None,
      "n_neighbors": n_neighbors,
      "pad_x": 0,
      "pad_y": 0,
    }
    self.synthetic_auto_run: bool = True
    self.synthetic_reference_time_window: np.timedelta64 = np.timedelta64(1, 'D')
    self.synthetic_kwargs: Dict[str, Any] = {
      "dx0": 0.0,
      "dy0": 0.0,
      "sx0": 1.0,
      "sy0": 1.0,
      "shift_range": (40, 40),
      "scale_range": (0.7, 1.3),
      "scale_step_x": 1e-3,
      "scale_step_y": 1e-3,
      "max_corr": -1.0,
      "corr_atol": 1e-4,
      "corr_rtol": 1e-3,
      "plateau_iters": 3,
      "n_neighbors": n_neighbors,
      "n_jobs": self.n_jobs,
      "center": None,
      "reference_data": None,
      "target_data": None,
      "crpix1": None,
      "crpix2": None,
      "extended_target_map": None,
      "reprojected_reference_on_extended_target_map": None,
      "reference_local_dir": None,
      "reference_channel_keyword": "fsi174",
      "reference_exclude_tokens": ("short",),
    }
    
    self._corr_workers: List[Process] = []
    self._task_queue: Queue | None = None
    self._result_queue: Queue | None = None
    self._worker_payload_key: Tuple[Any, ...] | None = None
    self._next_job_id = 0
    self._manager: BaseManager | None = None
    self._shared_payloads: Dict[str, Dict[str, Any]] | None = None
    self._current_payload_id: str | None = None
    self._latest_binning_base_map: GenericMap | None = self.map_to_coalign
    self._latest_one_map_base_map: GenericMap | None = None
    self._shared_debug_writer: PdfPages | None = None
    self._shared_debug_pdf_path: Path | None = None
    self._debug_session_depth: int = 0
    
  def _compute_kernel_size(self, meta: Mapping[str, Any]) -> Tuple[int, int]:
    """Return the binning kernel (rows, cols) derived from the map metadata."""
    cdelt1 = float(meta["CDELT1"])
    cdelt2 = float(meta["CDELT2"])
    cunit1 = str(meta["CUNIT1"])
    cunit2 = str(meta["CUNIT2"])
    kernel_rows = max(
      1,
      int(
        self.bin_kernel_arcsec.to(u.arcsec)
        / (u.Quantity(abs(cdelt2), cunit2)).to(u.arcsec)
      ),
    )
    kernel_cols = max(
      1,
      int(
        self.bin_kernel_arcsec.to(u.arcsec)
        / (u.Quantity(abs(cdelt1), cunit1)).to(u.arcsec)
      ),
    )
    return (kernel_rows, kernel_cols)

  def _release_payload_data(self) -> None:
    """Remove the current shared payload from the manager dictionary."""
    if self._shared_payloads is not None and self._current_payload_id is not None:
      self._shared_payloads.pop(self._current_payload_id, None)
    self._current_payload_id = None

  def _shutdown_corr_workers(self) -> None:
    """Shut down all worker processes and reset shared worker state."""
    if not self._corr_workers:
      self._release_payload_data()
      if self._manager is not None:
        self._manager.shutdown()
        self._manager = None
        self._shared_payloads = None
      self._worker_payload_key = None
      self._task_queue = None
      self._result_queue = None
      self._next_job_id = 0
      return
    if self._task_queue is not None:
      for _ in self._corr_workers:
        self._task_queue.put(None)
    for proc in self._corr_workers:
      proc.join()
    self._corr_workers.clear()
    self._task_queue = None
    self._result_queue = None
    self._worker_payload_key = None
    self._next_job_id = 0
    self._release_payload_data()
    if self._manager is not None:
      self._manager.shutdown()
      self._manager = None
      self._shared_payloads = None

  def _ensure_corr_workers(
    self,
    ref_img: np.ndarray,
    target_img: np.ndarray,
    center: Tuple[float, float] | None,
  ) -> None:
    """Spin up or reuse worker processes for batched correlation evaluations."""
    if self.n_jobs <= 1:
      if self._corr_workers:
        self._shutdown_corr_workers()
      return

    payload_key = make_worker_payload_key(ref_img, target_img, center)
    reuse_existing = (
      self._worker_payload_key == payload_key
      and bool(self._corr_workers)
      and self._task_queue is not None
      and self._result_queue is not None
    )

    if not reuse_existing:
      self._shutdown_corr_workers()

    if self._manager is None:
      self._manager = Manager()
      self._shared_payloads = cast(Dict[str, Dict[str, Any]], self._manager.dict())

    payload_id = store_shared_payload(
      self._shared_payloads,
      self._current_payload_id,
      ref_img,
      target_img,
      center,
    )
    self._current_payload_id = payload_id
    if payload_id is None:
      raise RuntimeError("Unable to store shared payload for correlation workers.")

    if reuse_existing:
      return

    self._task_queue = Queue()
    self._result_queue = Queue()
    self._next_job_id = 0
    for _ in range(self.n_jobs):
      proc = Process(
        target=corr_worker_loop,
        args=(self._task_queue, self._result_queue, self._shared_payloads, payload_id),
      )
      proc.daemon = True
      proc.start()
      self._corr_workers.append(proc)
    self._worker_payload_key = payload_key

  def _close_debug_writer(self) -> None:
    """Close the shared debug PDF writer if it is active."""
    if self._shared_debug_writer is not None:
      try:
        self._shared_debug_writer.close()
      except Exception:
        pass
    self._shared_debug_writer = None
    self._shared_debug_pdf_path = None

  def _push_debug_session(self) -> None:
    """Increment the nesting depth for shared debug PDF sessions."""
    self._debug_session_depth += 1

  def _pop_debug_session(self) -> None:
    """Decrement the session depth and finalize the PDF when leaving the outer scope."""
    if self._debug_session_depth == 0:
      return
    self._debug_session_depth -= 1
    if self._debug_session_depth == 0:
      self._close_debug_writer()

  def _create_debug_context(self, shift_x: int, shift_y: int, phase_key: Literal['binning', 'one_map']) -> DebugPlotContext:
    """Return a debug context that writes into the shared PDF file."""
    debug_dir = Path("data_storage") / "xcorr"
    debug_dir.mkdir(parents=True, exist_ok=True)
    max_span = max(shift_x, shift_y)
    base_radius = 0.4
    base_dpi = 300
    if phase_key == 'one_map' or max_span > 20:
      computed_radius = max_span * 0.015
      base_radius = min(0.9, max(0.35, computed_radius))
      base_dpi = min(320, 140 + int(max_span * 1.8))
    if self._shared_debug_writer is None or self._shared_debug_pdf_path is None:
      ctx = create_debug_context(
        shift_x,
        shift_y,
        debug_dir,
        close_writer_on_exit=False,
        point_radius_data=base_radius,
        dpi=base_dpi,
      )
      self._shared_debug_writer = ctx.pdf_writer
      self._shared_debug_pdf_path = ctx.pdf_path
      return ctx
    return create_debug_context(
      shift_x,
      shift_y,
      debug_dir,
      pdf_writer=self._shared_debug_writer,
      pdf_path=self._shared_debug_pdf_path,
      close_writer_on_exit=False,
      point_radius_data=base_radius,
      dpi=base_dpi,
    )

  def close(self) -> None:
    """Explicitly release worker processes and shared resources."""
    self._shutdown_corr_workers()
    self._close_debug_writer()

  def __del__(self) -> None:  # pragma: no cover - defensive cleanup
    """Ensure worker processes are torn down when the object is GC'ed."""
    try:
      self._shutdown_corr_workers()
      self._close_debug_writer()
    except Exception:
      pass

  def run_global_xcorr(self) -> None:
    """Execute the binned search followed by the one-map refinement."""
    session_managed_here = False
    if abs(self.verbose) >= 3:
      self._push_debug_session()
      session_managed_here = True
    try:
      self.run_binned_xcorr()
      self.run_one_map_xcorr()
      if self.synthetic_auto_run:
        self.run_synthetic_raster_xcorr()
    finally:
      if session_managed_here:
        self._pop_debug_session()

  def run_binned_cross_correlation(self) -> None:
    """Alias for run_binned_xcorr(). Use run_binned_xcorr() instead."""
    self.run_binned_xcorr()

  def run_binned_xcorr(self) -> None:
    """Execute Phase 1: Coarse coalignment using spatially binned maps.
    
    This phase performs a coarse 2D shift search using heavily binned (typically
    50 arcsec/pixel) versions of the target and reference maps. Binning provides:
    
    - **Speed**: ~100x faster than full resolution
    - **Robustness**: Reduces noise and small-scale mismatches
    - **Wide search**: Can efficiently explore large shift ranges (±20-40 pixels)
    
    The phase automatically:
    1. Bins both maps to ~50 arcsec resolution
    2. Expands the target FOV by ~700 arcsec for shift accommodation  
    3. Reprojects reference onto the extended target grid
    4. Runs 2D discrete gradient optimizer (shift only, no scale)
    5. Generates debug PDF and blink GIF showing before/after
    
    Results are stored in `self.results['binning']` and used as initialization
    for Phase 2 (one-map) alignment.

    Raises
    ------
    RuntimeError
        If worker pool fails to initialize or maps cannot be reprojected.
    ValueError
        If maps have incompatible WCS or insufficient overlap.

    Notes
    -----
    Default binning kernel is computed to achieve `bin_kernel_arcsec` (default
    50 arcsec) resolution. You can customize parameters before calling:
    
    >>> aligner.xcorr_binned_kwargs['shift_range'] = (40, 40)  # wider search
    >>> aligner.xcorr_binned_kwargs['plateau_iters'] = 5  # more patience
    >>> aligner.run_binned_xcorr()
    
    The optimizer uses a discrete lattice of neighbors ordered by anisotropic
    distance, evaluates correlations in parallel, and moves via correlation-
    weighted gradient descent. Step sizes shrink when correlation plateaus.
    
    Typical execution time: 10-30 seconds on 8 cores.

    Examples
    --------
    >>> aligner = Coaligner(spice_map, fsi_map, verbose=1, n_jobs=8)
    >>> aligner.run_binned_xcorr()
    Phase 1 (binning) complete: dx=12.3, dy=-5.7, corr=0.847
    >>> print(aligner.results['binning']['best_corr'])
    0.847

    See Also
    --------
    prepare_binned_data : Prepares binned maps (called automatically)
    run_one_map_xcorr : Phase 2 refinement
    _run_shift_search : Core 2D shift optimizer
    """
    self.prepare_binned_data()
    search_result, best_point = self._run_shift_search("binning")
    corrected_payload = self._build_corrected_target_map(
      target_map=self._latest_binning_base_map or self.map_to_coalign,
      extended_target_map=self.xcorr_binned_kwargs.get('extended_target_map'),
      pad_x=int(self.xcorr_binned_kwargs.get('pad_x') or 0),
      pad_y=int(self.xcorr_binned_kwargs.get('pad_y') or 0),
      kernel=cast(Tuple[int, int], self.xcorr_binned_kwargs.get('kernel') or self.bin_kernel or (1, 1)),
      best_shift=best_point,
    )
    self._finalize_phase_result("binning", search_result, corrected_payload)
  
  def _run_shift_scale_search(
    self,
    cfg: Dict[str, Any],
    phase_key: Literal['binning', 'one_map', 'synthetic_raster'],
    label: str,
    shift_x: int,
    shift_y: int,
  ) -> Tuple[Dict[str, Any], Tuple[int, int, int, int]]:
    """Search the joint shift/scale grid and return the best parameters."""
    ref_img = np.asarray(cfg['reference_data'], dtype=np.float64)
    target_img = np.asarray(cfg['target_data'], dtype=np.float64)
    scale_range_cfg = cfg.get('scale_range')
    if not isinstance(scale_range_cfg, Sequence) or len(scale_range_cfg) != 2:
      raise ValueError("scale_range must be a length-2 sequence when searching shift+scale.")
    def _is_pair_sequence(value: Any) -> bool:
      return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    def _as_bounds(pair: Sequence[Any]) -> Tuple[float, float]:
      if len(pair) != 2:
        raise ValueError("scale_range pairs must contain exactly two values.")
      lower = float(pair[0])
      upper = float(pair[1])
      if lower > upper:
        lower, upper = upper, lower
      if math.isclose(lower, upper):
        raise ValueError("scale_range bounds must span a non-zero interval.")
      if lower <= 0.0 or upper <= 0.0:
        raise ValueError("scale_range bounds must be positive.")
      return lower, upper

    first_entry, second_entry = scale_range_cfg[0], scale_range_cfg[1]
    if _is_pair_sequence(first_entry) and _is_pair_sequence(second_entry):
      scale_bounds_x = _as_bounds(cast(Sequence[Any], first_entry))
      scale_bounds_y = _as_bounds(cast(Sequence[Any], second_entry))
    elif not _is_pair_sequence(first_entry) and not _is_pair_sequence(second_entry):
      scale_bounds_x = _as_bounds(cast(Sequence[Any], scale_range_cfg))
      scale_bounds_y = scale_bounds_x
    else:
      raise ValueError("scale_range must be either (min,max) or ((min,max),(min,max)).")

    def _scalar(value: Any, fallback: float) -> float:
      if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return float(value[0])
      return float(value if value is not None else fallback)

    scale_step_x = _scalar(cfg.get('scale_step_x', cfg.get('scale_step')), 1e-3)
    scale_step_y = _scalar(cfg.get('scale_step_y', cfg.get('scale_step')), 1e-3)
    if scale_step_x <= 0.0 or scale_step_y <= 0.0:
      raise ValueError("scale_step values must be positive.")
    def _default_pivot(bounds: Tuple[float, float], preferred: float = 1.0) -> float:
      lower, upper = bounds
      if preferred < lower:
        return lower
      if preferred > upper:
        return upper
      return preferred

    pivot_scale_x = _default_pivot(scale_bounds_x)
    pivot_scale_y = _default_pivot(scale_bounds_y)

    def _axis_radius(bounds: Tuple[float, float], pivot: float, step: float) -> int:
      lower, upper = bounds
      pivot = _default_pivot(bounds, pivot)
      radius = max(abs(upper - pivot), abs(pivot - lower))
      if radius <= 0.0:
        raise ValueError("scale_range bounds must extend away from the pivot value.")
      return max(1, int(math.ceil(radius / step)))

    sx_range = _axis_radius(scale_bounds_x, pivot_scale_x, scale_step_x)
    sy_range = _axis_radius(scale_bounds_y, pivot_scale_y, scale_step_y)

    raw_sx0 = cfg.get('sx0')
    raw_sy0 = cfg.get('sy0')
    sx0_factor = float(raw_sx0) if isinstance(raw_sx0, (int, float)) and raw_sx0 > 0 else pivot_scale_x
    sy0_factor = float(raw_sy0) if isinstance(raw_sy0, (int, float)) and raw_sy0 > 0 else pivot_scale_y
    sx0 = int(np.clip(round((sx0_factor - pivot_scale_x) / scale_step_x), -sx_range, sx_range))
    sy0 = int(np.clip(round((sy0_factor - pivot_scale_y) / scale_step_y), -sy_range, sy_range))

    dr0 = np.array([
      float(cfg.get('dx0') or 0.0),
      float(cfg.get('dy0') or 0.0),
    ], dtype=float)

    n_neighbors: int = max(1, int(cfg.get('n_neighbors') or self.n_jobs * 10))
    n_jobs: int = max(1, int(cfg.get('n_jobs') or self.n_jobs))
    max_corr_goal: float = float(cfg.get('max_corr', -1.0))
    force_plateau = max_corr_goal < 0.0
    if force_plateau:
      max_corr_goal = float('inf')
    corr_atol: float = float(cfg.get('corr_atol', 1e-4))
    corr_rtol: float = float(cfg.get('corr_rtol', 1e-3))
    plateau_iters: int = int(cfg.get('plateau_iters', 6))

    center_override = cfg.get('center')
    center_pix: Tuple[float, float] | None = None
    if isinstance(center_override, Sequence) and len(center_override) >= 2:
      center_pix = (float(center_override[0]), float(center_override[1]))
    elif phase_key == 'binning':
      if cfg.get('binned_crpix1') is not None and cfg.get('binned_crpix2') is not None:
        center_pix = (float(cfg['binned_crpix1']), float(cfg['binned_crpix2']))
    elif cfg.get('crpix1') is not None and cfg.get('crpix2') is not None:
      center_pix = (float(cfg['crpix1']), float(cfg['crpix2']))

    self._ensure_corr_workers(ref_img, target_img, center_pix)
    effective_workers = n_jobs if n_jobs > 1 else 1
    _vprint(
      self.verbose,
      2,
      (
        f"Using {effective_workers} worker{'s' if effective_workers>1 else ''} computing up to "
        f"{n_neighbors} neighbours each iteration in shift+scale space..."
      ),
    )

    debug_plots = abs(self.verbose) >= 3
    debug_ctx: DebugPlotContext | None = None
    debug_pdf_path: Path | None = None
    session_started_here = False
    debug_phase_label = phase_key if phase_key != 'synthetic_raster' else 'one_map'

    if debug_plots:
      if self._debug_session_depth == 0:
        self._push_debug_session()
        session_started_here = True
      debug_ctx = self._create_debug_context(shift_x, shift_y, cast(Literal['binning', 'one_map'], debug_phase_label))
      debug_pdf_path = self._shared_debug_pdf_path
      if debug_ctx is not None:
        debug_ctx.reset_scatter_size()
      if debug_pdf_path is not None:
        _vprint(self.verbose,2,f"Saving cross-correlation search progression plots to {debug_pdf_path}")

    axis_ranges = (shift_x, shift_y, sx_range, sy_range)
    total_points = 1
    for rng in axis_ranges:
      total_points *= (2 * int(rng) + 1)
    _vprint(self.verbose,2,f"Total shift+scale points: {total_points}...")

    dims = len(axis_ranges)
    def _stream_points(start_point: Tuple[int, ...]):
      visited: set[Tuple[int, ...]] = {start_point}
      heap: list[Tuple[int, int, Tuple[int, ...]]] = []
      ticket = count()

      def _push(candidate: Tuple[int, ...]) -> None:
        if candidate in visited:
          return
        if not point_in_bounds_nd(candidate, axis_ranges):
          return
        visited.add(candidate)
        dist = max(abs(candidate[idx] - start_point[idx]) for idx in range(dims))
        heapq.heappush(heap, (dist, next(ticket), candidate))

      for axis_idx in range(dims):
        for delta in (-1, 1):
          neighbor = list(start_point)
          neighbor[axis_idx] += delta
          _push(tuple(neighbor))

      while heap:
        _, _, point = heapq.heappop(heap)
        yield point
        for axis_idx in range(dims):
          for delta in (-1, 1):
            neighbor = list(point)
            neighbor[axis_idx] += delta
            _push(tuple(neighbor))

    def _gather_neighbors_lazy(center: Tuple[int, ...], limit: int) -> List[Tuple[int, ...]]:
      if limit <= 0:
        return []
      neighbors: List[Tuple[int, ...]] = []
      for candidate in _stream_points(center):
        if candidate in cache:
          continue
        neighbors.append(candidate)
        if len(neighbors) >= limit:
          break
      return neighbors

    def _first_unvisited_lazy(reference_point: Tuple[int, ...]) -> Tuple[int, ...] | None:
      if reference_point not in cache:
        return reference_point
      for candidate in _stream_points(reference_point):
        if candidate not in cache:
          return candidate
      return None

    cache: Dict[Tuple[int, int, int, int], float] = {}
    history: List[Dict[str, float]] = []
    evaluated_count = 0
    center_trace: List[Tuple[int, int, int, int]] = []

    task_queue = self._task_queue
    result_queue = self._result_queue
    has_workers = bool(
      self._corr_workers
      and task_queue
      and result_queue
      and self._shared_payloads is not None
      and self._current_payload_id is not None
    )

    def _clamp_scale(value: float, bounds: Tuple[float, float]) -> float:
      lower, upper = bounds
      if value < lower:
        return lower
      if value > upper:
        return upper
      return value

    def _decode_point(point: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
      sx_val = _clamp_scale(pivot_scale_x + point[2] * scale_step_x, scale_bounds_x)
      sy_val = _clamp_scale(pivot_scale_y + point[3] * scale_step_y, scale_bounds_y)
      return float(point[0]), float(point[1]), float(sx_val), float(sy_val)

    def record_corr(point: Tuple[int, int, int, int], corr_val: float) -> None:
      dx_val, dy_val, sx_val, sy_val = _decode_point(point)
      cache[point] = corr_val
      history.append({
        "dx": dx_val,
        "dy": dy_val,
        "sx": sx_val,
        "sy": sy_val,
        "sx_index": float(point[2]),
        "sy_index": float(point[3]),
        "corr": float(corr_val),
      })
      if debug_ctx is not None:
        debug_ctx.add_point(int(point[0]), int(point[1]), corr_val)

    def evaluate_points(points: List[Tuple[int, int, int, int]]) -> List[Tuple[Tuple[int, int, int, int], float]]:
      results: List[Tuple[Tuple[int, int, int, int], float]] = []
      if not points:
        return results

      worker_points: List[Tuple[int, int, int, int]] = []
      job_map: Dict[int, List[Tuple[int, int, int, int]]] = {}
      for point in points:
        if point in cache:
          results.append((point, cache[point]))
          continue
        if has_workers and task_queue and result_queue:
          worker_points.append(point)
          continue
        dx_val, dy_val, sx_val, sy_val = _decode_point(point)
        corr_val = correlation_for_params(
          ref_img,
          target_img,
          dx_val,
          dy_val,
          sx_val,
          sy_val,
          center=center_pix,
        )
        record_corr(point, corr_val)
        results.append((point, corr_val))

      if worker_points and has_workers and task_queue and result_queue:
        batch_size = max(1, int(math.ceil(len(worker_points) / max(1, effective_workers))))

        def _chunk_points(seq: List[Tuple[int, int, int, int]], size: int) -> Iterable[List[Tuple[int, int, int, int]]]:
          for idx in range(0, len(seq), size):
            yield seq[idx : idx + size]

        pending_jobs: set[int] = set()
        for chunk in _chunk_points(worker_points, batch_size):
          job_id = self._next_job_id
          self._next_job_id += 1
          dxs = [float(pt[0]) for pt in chunk]
          dys = [float(pt[1]) for pt in chunk]
          sxs = [pivot_scale_x + pt[2] * scale_step_x for pt in chunk]
          sys = [pivot_scale_y + pt[3] * scale_step_y for pt in chunk]
          payload_ids = [self._current_payload_id] * len(chunk)
          task_queue.put((job_id, payload_ids, dxs, dys, sxs, sys))
          pending_jobs.add(job_id)
          job_map[job_id] = list(chunk)

        while pending_jobs:
          job_id, batch = result_queue.get()
          pending_jobs.discard(job_id)
          chunk_points = job_map.pop(job_id, [])
          for (dx_val, dy_val, corr_val), point in zip(batch, chunk_points):
            record_corr(point, corr_val)
            results.append((point, corr_val))

      return results

    try:
      current_center = clamp_point_nd((int(round(dr0[0])), int(round(dr0[1])), sx0, sy0), axis_ranges)
      if current_center not in cache:
        evaluate_points([current_center])
      center_trace.append(current_center)
      best_point, best_corr = max(cache.items(), key=lambda item: item[1])
      global_best_corr = best_corr
      search_phase = 2 if force_plateau else 1
      plateau_counter = 0
      last_best_increase = best_corr
      iteration_number = 0
      stall_counts: Dict[Tuple[int, int, int, int], int] = defaultdict(int)

      while True:
        if len(cache) >= total_points:
          _vprint(self.verbose,2,"All shift+scale positions evaluated; stopping.")
          break
        iteration_number += 1
        _vprint(
          self.verbose,
          2,
          (
            f"Iteration {iteration_number}: current center {current_center}, cache size {len(cache)}/{total_points}, "
            f"phase {search_phase}..."
          ),
        )
        neighbor_limit = min(n_neighbors, total_points - len(cache))
        new_points = _gather_neighbors_lazy(current_center, neighbor_limit)
        evaluated = evaluate_points(new_points)
        evaluated_count = len(cache)

        local_points = [(current_center, cache[current_center])]
        local_points.extend(evaluated)

        if debug_ctx is not None:
          debug_ctx.render_iteration((current_center[0], current_center[1]), search_phase)

        best_point, best_corr = max(cache.items(), key=lambda item: item[1])
        if best_corr > global_best_corr:
          global_best_corr = best_corr
        _vprint(
          self.verbose,
          2,
          (
            f"Best local corr: {cache[current_center]:.5f}, best overall corr: {global_best_corr:.5f} at "
            f"shift+scale {best_point}."
          ),
        )
        search_phase, plateau_counter, last_best_increase, plateau_met, phase_switched = update_phase_state(
          search_phase,
          best_corr,
          last_best_increase,
          plateau_counter,
          max_corr_goal,
          corr_atol,
          corr_rtol,
          plateau_iters,
        )
        if phase_switched and not force_plateau:
          _vprint(self.verbose,2,f"Switching to plateau phase at corr={best_corr:.5f}")
        if plateau_met:
          _vprint(self.verbose,2,"Plateau condition met; stopping.")
          break

        _vprint(self.verbose,2,"Computing next step ...")
        step_vec = compute_gradient_step_nd(local_points, current_center, best_point)
        candidate_center = clamp_point_nd(
          tuple(current_center[idx] + int(step_vec[idx]) for idx in range(len(axis_ranges))),
          axis_ranges,
        )
        _vprint(self.verbose,2,"Finding next candidate center from gradient step ...")
        while candidate_center in cache and candidate_center != current_center:
          stall_counts[candidate_center] += 1
          if stall_counts[candidate_center] >= 2:
            fallback_local = _first_unvisited_lazy(candidate_center)
            _vprint(
              self.verbose,
              2,
              (
                f"Stalling at {candidate_center} with corr {cache[candidate_center]:.5f}; "
                f"trying fallback local {fallback_local}..."
              ),
            )
            if fallback_local is not None:
              candidate_center = fallback_local
              _vprint(self.verbose,2,f"Falling back to {candidate_center} for next candidate center.")
              break
            _vprint(self.verbose,2,f"No fallback found for stalled candidate {candidate_center}; breaking out of stall loop.")
            break
          next_candidate = clamp_point_nd(
            tuple(candidate_center[idx] + int(step_vec[idx]) for idx in range(len(axis_ranges))),
            axis_ranges,
          )
          if next_candidate == candidate_center:
            break
          candidate_center = next_candidate
          if candidate_center == current_center or not point_in_bounds_nd(candidate_center, axis_ranges):
            break

        if (
          candidate_center == current_center
          or not point_in_bounds_nd(candidate_center, axis_ranges)
          or candidate_center in cache
        ):
          fallback = _first_unvisited_lazy(current_center)
          if fallback is None:
            _vprint(self.verbose,2,"No remaining unvisited positions; stopping.")
            break
          candidate_center = fallback

        current_center = candidate_center
        center_trace.append(current_center)
        if current_center not in cache:
          evaluate_points([current_center])

      dx_val, dy_val, sx_val, sy_val = _decode_point(best_point)
      best_summary: Dict[str, float] = {
        "dx": dx_val,
        "dy": dy_val,
        "sx": sx_val,
        "sy": sy_val,
        "sx_index": float(best_point[2]),
        "sy_index": float(best_point[3]),
        "corr": float(best_corr),
      }

      result_payload: Dict[str, Any] = {
        "best": best_summary,
        "history": history,
        "centers": center_trace,
        "evaluated": evaluated_count,
        "total": total_points,
        "phase": "plateau" if search_phase == 2 else "search",
      }
      ref_map_obj = cfg.get('reprojected_reference_on_extended_target_map')
      target_map_obj = cfg.get('extended_target_map')
      if debug_ctx is not None and cache:
        # Render history plot (adaptive based on whether scale varies)
        history_array = np.array([[i, h['dx'], h['dy'], h['sx'], h['sy'], h['corr']] for i, h in enumerate(history)], dtype=float)
        debug_ctx.render_history_plot(history_array)
        
        # Build best_params for corrected map visualization
        best_params = {
          "dx": dx_val,
          "dy": dy_val,
          "squeeze_x": sx_val,
          "squeeze_y": sy_val,
        }
        
        # Build corrected map for visualization
        corrected_for_viz = None
        if isinstance(target_map_obj, GenericMap):
          try:
            corrected_for_viz = make_corrected_wcs_map(target_map_obj, best_params, verbose=0)
          except Exception:
            pass
        
        debug_ctx.render_alignment_overlay(
          ref_img,
          target_img,
          (best_point[0], best_point[1]),
          ref_map=ref_map_obj if isinstance(ref_map_obj, GenericMap) else None,
          target_map=target_map_obj if isinstance(target_map_obj, GenericMap) else None,
          corrected_map=corrected_for_viz,
        )
        
        # Render comparison animation if we have all maps
        if corrected_for_viz is not None and isinstance(ref_map_obj, GenericMap):
          _vprint(self.verbose, 2, "Rendering comparison animation for shift+scale search...")
          debug_ctx.render_comparison_animation(ref_map_obj, target_map_obj, corrected_for_viz, phase_name=phase_key)
      self._release_payload_data()
      _vprint(
        self.verbose,
        1,
        (
          f"[{label}] Best correlation found: {best_corr:.6f} at shift ({dx_val:.2f}, {dy_val:.2f}) "
          f"and scales ({sx_val:.5f}, {sy_val:.5f})."
        ),
      )
      return (result_payload, best_point)
    finally:
      if debug_ctx is not None:
        debug_ctx.close()
        if debug_pdf_path is not None:
          _vprint(self.verbose, 2, f"Saved shift+scale xcorr debug trace to {debug_pdf_path}")
      if session_started_here:
        self._pop_debug_session()

  def run_one_map_cross_correlation(
    self,
    seed_shift: Tuple[float, float] | None = None,
  ) -> None:
    """Alias for run_one_map_xcorr(). Use run_one_map_xcorr() instead."""
    self.run_one_map_xcorr(seed_shift=seed_shift)

  def run_one_map_xcorr(
    self,
    seed_shift: Tuple[float, float] | None = None,
  ) -> None:
    """Execute Phase 2: Fine coalignment using full-resolution single reference map.
    
    This phase refines the coarse alignment from Phase 1 by:
    
    - Using full-resolution maps (no binning)
    - Starting from Phase 1 results (or custom seed_shift)
    - Searching a smaller shift range with finer steps
    - Optimizing 2D shift only (no scale adjustment yet)
    
    The phase automatically:
    1. Uses the corrected map from Phase 1 as the starting point
    2. Expands target FOV to accommodate remaining misalignment
    3. Reprojects reference onto the extended target grid
    4. Runs 2D discrete gradient optimizer at full resolution
    5. Generates debug PDF and blink GIF showing before/after
    
    Results are stored in `self.results['one_map']` and used as initialization
    for Phase 3 (synthetic raster) alignment.

    Parameters
    ----------
    seed_shift : tuple of float, optional
        Custom (dx, dy) shift to use as starting point instead of Phase 1 results.
        Useful for testing or when Phase 1 was not run. If None, uses results
        from `run_binned_xcorr()`.

    Raises
    ------
    RuntimeError
        If Phase 1 was not run and seed_shift is not provided.
    ValueError
        If maps have insufficient overlap after shift correction.

    Notes
    -----
    This phase typically refines the Phase 1 shift by a few pixels. The default
    search range is ±10 pixels, which is usually sufficient. Customize parameters:
    
    >>> aligner.xcorr_one_map_kwargs['shift_range'] = (15, 15)  # wider search
    >>> aligner.run_one_map_xcorr()
    
    The optimizer uses the same discrete gradient approach as Phase 1 but operates
    at full map resolution, making it ~100x slower but more accurate.
    
    Typical execution time: 2-5 minutes on 8 cores for 1024x1024 maps.

    Examples
    --------
    Standard workflow (after Phase 1):
    
    >>> aligner.run_binned_xcorr()
    >>> aligner.run_one_map_xcorr()
    Phase 2 (one_map) complete: dx=0.8, dy=-1.2, corr=0.923
    
    Custom starting point:
    
    >>> aligner.run_one_map_xcorr(seed_shift=(10.0, -5.0))

    See Also
    --------
    prepare_one_map_data : Prepares full-resolution maps (called automatically)
    run_binned_xcorr : Phase 1 coarse alignment (should run first)
    run_synthetic_raster_xcorr : Phase 3 refinement with scale optimization
    _run_shift_search : Core 2D shift optimizer
    """
    self.prepare_one_map_data(seed_shift=seed_shift)
    search_result, best_point = self._run_shift_search("one_map")
    corrected_payload = self._build_corrected_target_map(
      target_map=self._latest_one_map_base_map or self.map_to_coalign,
      extended_target_map=self.xcorr_one_map_kwargs.get('extended_target_map'),
      pad_x=int(self.xcorr_one_map_kwargs.get('pad_x') or 0),
      pad_y=int(self.xcorr_one_map_kwargs.get('pad_y') or 0),
      kernel=(1, 1),
      best_shift=best_point,
    )
    self._finalize_phase_result("one_map", search_result, corrected_payload)


  def prepare_binned_data(self) -> None:
    """Generate padded, filtered, and binned arrays used by xcorr search."""
    _vprint(self.verbose,1,"Preparing binned data for coalignement...")
    meta = self.map_to_coalign.meta
    if meta is None:
        raise ValueError("map_to_coalign.meta is None (unexpected).")

    # Optional: cast for Pylance so subscripting is typed
    meta = cast(Mapping[str, Any], meta)

    cdelt1 = float(meta["CDELT1"])
    cdelt2 = float(meta["CDELT2"])
    cunit1 = str(meta["CUNIT1"])
    cunit2 = str(meta["CUNIT2"])

    kernel_size = self._compute_kernel_size(meta)
    self.bin_kernel = kernel_size
    self.xcorr_binned_kwargs['kernel'] = kernel_size
    self._latest_binning_base_map = self.map_to_coalign

    target_x_extension_pixel = int(
      self.target_FOV_expension_arcsec.to(u.arcsec)
      / (u.Quantity(abs(cdelt1), cunit1)).to(u.arcsec),
    )
    target_y_extension_pixel = int(
      self.target_FOV_expension_arcsec.to(u.arcsec)
      / (u.Quantity(abs(cdelt2), cunit2)).to(u.arcsec),
    )
    self.xcorr_binned_kwargs['pad_x'] = target_x_extension_pixel
    self.xcorr_binned_kwargs['pad_y'] = target_y_extension_pixel

    extended_target_map: GenericMap = cast(
      GenericMap,
      enlarge_map_by_padding(
        self.map_to_coalign,
        pad_x=target_x_extension_pixel,
        pad_y=target_y_extension_pixel,
      ),
    )
    reprojected_reference_on_extended_target_map: GenericMap = reproject_map_to_reference(
      extended_target_map,
      self.reference_map,
    )
    extended_target_map = cast(
      GenericMap,
      sunpy.map.Map(
        no_nan_uniform_filter(
          np.asarray(extended_target_map.data),
          remove_percentile=99,
          size=kernel_size,
        ),
        extended_target_map.meta,
        plot_settings=extended_target_map.plot_settings,
      ),
    )

    binned_target_matrix: np.ndarray = extended_target_map.data.copy()[::kernel_size[0], ::kernel_size[1]]
    binned_reference_matrix: np.ndarray = (
      reprojected_reference_on_extended_target_map.data.copy()[::kernel_size[0], ::kernel_size[1]]
    )
    crpix1_binned: float = float(meta['crpix1']) / float(kernel_size[1])
    crpix2_binned: float = float(meta['crpix2']) / float(kernel_size[0])
    self.xcorr_binned_kwargs['binned_crpix1'] = crpix1_binned
    self.xcorr_binned_kwargs['binned_crpix2'] = crpix2_binned
    self.xcorr_binned_kwargs['reference_data'] = binned_reference_matrix
    self.xcorr_binned_kwargs['target_data'] = binned_target_matrix
    self.xcorr_binned_kwargs['extended_target_map'] = extended_target_map
    self.xcorr_binned_kwargs['reprojected_reference_on_extended_target_map'] = reprojected_reference_on_extended_target_map

  def prepare_one_map_data(self, seed_shift: Tuple[float, float] | None = None) -> None:
    """Generate the full-resolution arrays used by the one-map refinement."""
    _vprint(self.verbose,1,"Preparing reprojected reference map for one-map coalignement...")
    base_map: GenericMap | None = None
    binning_result = self.results.get("binning")
    using_corrected_binning_map = False
    if binning_result and isinstance(binning_result, dict):
      corrected_map = binning_result.get("corrected_target_map")
      if isinstance(corrected_map, GenericMap):
        base_map = corrected_map
        using_corrected_binning_map = True
    if base_map is None:
      base_map = self.map_to_coalign

    meta = base_map.meta
    if meta is None:
        raise ValueError("Base map metadata is None (unexpected).")

    meta = cast(Mapping[str, Any], meta)
    kernel_size = self._compute_kernel_size(meta)
    self.bin_kernel = kernel_size
    self._latest_one_map_base_map = base_map

    coarse_best: Tuple[float, float] | None = None
    coarse_source: Literal['manual', 'binning', 'default'] | None = None
    if seed_shift is not None:
      coarse_best = (float(seed_shift[0]), float(seed_shift[1]))
      coarse_source = 'manual'
    elif self.results["binning"] and self.results["binning"].get("best"):
      best = cast(Mapping[str, float], self.results["binning"].get("best"))
      coarse_best = (float(best.get("dx", 0.0)), float(best.get("dy", 0.0)))
      coarse_source = 'binning'

    if coarse_best is None:
      _vprint(
        self.verbose,
        1,
        "No coarse binning result or explicit seed shift provided; defaulting one-map seed to (0, 0).",
      )
      coarse_best = (0.0, 0.0)
      coarse_source = 'default'

    if coarse_source == 'binning' and using_corrected_binning_map:
      _vprint(
        self.verbose,
        2,
        "Using binned corrected map as the one-map base; resetting seed shift to (0, 0).",
      )
      coarse_best = (0.0, 0.0)

    dx_unbinned = float(coarse_best[0]) * float(kernel_size[1])
    dy_unbinned = float(coarse_best[1]) * float(kernel_size[0])
    self.xcorr_one_map_kwargs["dx0"] = dx_unbinned
    self.xcorr_one_map_kwargs["dy0"] = dy_unbinned

    if self.xcorr_one_map_kwargs.get("shift_range") is None:
      self.xcorr_one_map_kwargs["shift_range"] = (
        max(5, int(2 * kernel_size[1])),
        max(5, int(2 * kernel_size[0])),
      )

    target_x_extension_pixel = kernel_size[1] * 2
    target_y_extension_pixel = kernel_size[0] * 2
    self.xcorr_one_map_kwargs['pad_x'] = target_x_extension_pixel
    self.xcorr_one_map_kwargs['pad_y'] = target_y_extension_pixel

    extended_target_map: GenericMap = cast(
      GenericMap,
      enlarge_map_by_padding(
        base_map,
        pad_x=target_x_extension_pixel,
        pad_y=target_y_extension_pixel,
      ),
    )
    reprojected_reference_on_extended_target_map: GenericMap = reproject_map_to_reference(
      extended_target_map,
      self.reference_map,
    )
    filter_size = (max(1, kernel_size[0] // 2), max(1, kernel_size[1] // 2))
    extended_target_map = cast(
      GenericMap,
      sunpy.map.Map(
        no_nan_uniform_filter(
          np.asarray(extended_target_map.data),
          remove_percentile=99,
          size=filter_size,
        ),
        extended_target_map.meta,
        plot_settings=extended_target_map.plot_settings,
      ),
    )
    meta_extended = cast(Mapping[str, Any], extended_target_map.meta)
    crpix1 = meta_extended['crpix1']
    crpix2 = meta_extended['crpix2']
    self.xcorr_one_map_kwargs['crpix1'] = crpix1
    self.xcorr_one_map_kwargs['crpix2'] = crpix2
    self.xcorr_one_map_kwargs['reference_data'] = reprojected_reference_on_extended_target_map.data
    self.xcorr_one_map_kwargs['target_data'] = extended_target_map.data
    self.xcorr_one_map_kwargs['extended_target_map'] = extended_target_map
    self.xcorr_one_map_kwargs['reprojected_reference_on_extended_target_map'] = reprojected_reference_on_extended_target_map

  def _extract_meta_datetime(self, meta: Mapping[str, Any], keys: Sequence[str]) -> np.datetime64 | None:
    for key in keys:
      value = meta.get(key)
      if value is None:
        continue
      try:
        dt_value = value if isinstance(value, np.datetime64) else np.datetime64(value)
      except Exception:
        try:
          dt_value = np.datetime64(str(value))
        except Exception:
          continue
      return np.datetime64(dt_value, 'ms')
    return None

  def _resolve_synthetic_base_map(self) -> GenericMap:
    preferred_keys: Tuple[str, ...] = ('one_map', 'binning')
    for key in preferred_keys:
      phase_result = self.results.get(key)
      if isinstance(phase_result, dict):
        corrected = phase_result.get('corrected_target_map')
        if isinstance(corrected, GenericMap):
          return corrected
    return self.map_to_coalign

  def _gather_synthetic_reference_sources(self, base_map: GenericMap) -> List[GenericMap | Path | str]:
    if self.list_of_reference_maps:
      return list(self.list_of_reference_maps)

    meta = cast(Mapping[str, Any], base_map.meta)
    date_beg = self._extract_meta_datetime(meta, ('DATE-BEG', 'DATE_BEG', 'DATE-OBS', 'DATE_OBS'))
    date_end = self._extract_meta_datetime(meta, ('DATE-END', 'DATE_END', 'DATE-OBS', 'DATE_OBS'))
    if date_beg is None or date_end is None:
      return []
    interval = self.synthetic_reference_time_window
    date_avg = self._extract_meta_datetime(meta, ('DATE-AVG', 'DATE_AVG', 'DATE-OBS', 'DATE_OBS')) or date_beg
    local_dir_setting = self.synthetic_kwargs.get('reference_local_dir')
    local_dir = Path(local_dir_setting) if local_dir_setting else Path('.')
    verbose_level = max(0, int(self.verbose))

    try:
      candidate_paths = get_EUI_paths(date_min=date_beg, date_max=date_end, local_dir=local_dir, verbose=verbose_level)
    except Exception as exc:
      _vprint(self.verbose, -1, f"Failed to scan EUI directory {local_dir}: {exc}")
      candidate_paths = []

    channel_token = str(self.synthetic_kwargs.get('reference_channel_keyword') or '').lower()
    exclude_tokens = tuple(str(tok).lower() for tok in self.synthetic_kwargs.get('reference_exclude_tokens', ()))
    filtered: List[Path] = []
    for path in candidate_paths:
      name = path.name.lower()
      if channel_token and channel_token not in name:
        continue
      if any(token and token in name for token in exclude_tokens):
        continue
      filtered.append(Path(path))

    try:
      closest = get_closest_EUIFSI174_paths(
        date_avg,
        interval=interval,
        local_dir=local_dir,
        verbose=verbose_level,
      )
    except Exception as exc:
      _vprint(self.verbose, -1, f"Unable to locate closest FSI174 files: {exc}")
      closest = []

    for candidate in closest:
      if candidate not in filtered:
        filtered.append(candidate)

    deduped: List[GenericMap | Path | str] = list(dict.fromkeys(filtered))
    return deduped

  def _build_synthetic_raster_from_fsi(self, base_map: GenericMap) -> Tuple[GenericMap, List[GenericMap | Path | str]]:
    sources = self._gather_synthetic_reference_sources(base_map)
    if not sources:
      raise RuntimeError("No FSI inputs available to build the synthetic raster.")
    verbose_level = max(0, int(self.verbose))
    synthetic_map = build_synthetic_raster_from_maps(
      base_map,
      sources,
      verbose=verbose_level,
    )
    return synthetic_map, sources

  def run_synthetic_raster_xcorr(self) -> None:
    """Execute Phase 3: Optimal coalignment using synthetic raster from FSI sequence.
    
    This is the most sophisticated phase, addressing temporal mismatch between
    SPICE rasters (which scan over minutes) and snapshot FSI images. The phase:
    
    1. **Loads FSI Image Sequence**: Automatically finds all FSI 174 images that
       bracket the SPICE raster observation time (typically 5-20 images)
    2. **Generates Synthetic Raster**: Interpolates FSI images onto each SPICE
       raster slit position at the precise observation time of that slit
    3. **4D Optimization**: Searches for best shift (dx, dy) **and scale** (sx, sy)
       factors simultaneously
    4. **Final Correction**: Applies the optimal transformation to the target WCS
    
    This phase produces the highest-quality alignment because:
    
    - Synthetic raster perfectly matches SPICE observation cadence
    - Scale factors correct for residual plate scale errors
    - Temporal evolution (e.g., loop motion) is properly accounted for
    
    The phase automatically:
    - Uses Phase 2 results as starting point (or Phase 1 if Phase 2 wasn't run)
    - Loads FSI images from disk based on SPICE time range
    - Builds synthetic raster with sub-pixel interpolation
    - Runs 4D discrete gradient optimizer (dx, dy, sx, sy)
    - Generates debug PDF and blink GIF showing before/after
    
    Results are stored in `self.results['synthetic_raster']` and `self.best_params`
    contains the final corrected WCS parameters.

    Raises
    ------
    RuntimeError
        If FSI images cannot be loaded or synthetic raster generation fails.
    ValueError
        If optimization parameters are invalid or maps have incompatible shapes.
    FileNotFoundError
        If FSI image directory is not accessible.

    Notes
    -----
    **FSI Image Discovery**:
    The method searches for FSI 174 images in the directory specified by
    `synthetic_kwargs['reference_local_dir']` (default: current directory).
    Images are filtered by:
    - Channel keyword (default: "fsi174")
    - Time range: SPICE DATE-BEG to DATE-END ± time window (default: ±1 day)
    - Exclusion tokens (default: excludes "short" exposures)
    
    **Scale Optimization**:
    Scale factors typically converge to 0.998-1.002, correcting for minor plate
    scale mismatches between instruments. Search range defaults to 0.7-1.3 with
    1e-3 step size.
    
    **Performance**:
    This is the slowest phase due to 4D search space. Typical execution time:
    5-15 minutes on 8 cores for moderate-size rasters. Can be sped up by:
    - Reducing `n_neighbors` (default: 80, min recommended: 40)
    - Tightening `scale_range` if plate scales are well calibrated
    - Using more cores via `n_jobs`

    Examples
    --------
    Full three-phase pipeline:
    
    >>> aligner = Coaligner(spice_map, fsi_map, verbose=1, n_jobs=16)
    >>> aligner.run_binned_xcorr()
    >>> aligner.run_one_map_xcorr()
    >>> aligner.run_synthetic_raster_xcorr()
    Phase 3 (synthetic_raster) complete: dx=0.3, dy=-0.5, sx=1.0012, sy=0.9998, corr=0.962
    >>> print(aligner.best_params)
    {'CRVAL1': -150.234, 'CRVAL2': 280.567, 'CDELT1': 1.002, ...}
    
    Customize FSI loading:
    
    >>> aligner.synthetic_kwargs['reference_local_dir'] = '/data/eui/fsi174'
    >>> aligner.synthetic_kwargs['reference_channel_keyword'] = 'fsi174'
    >>> aligner.synthetic_reference_time_window = np.timedelta64(2, 'h')
    >>> aligner.run_synthetic_raster_xcorr()
    
    Adjust scale search:
    
    >>> aligner.synthetic_kwargs['scale_range'] = (0.95, 1.05)  # tighter bounds
    >>> aligner.synthetic_kwargs['scale_step_x'] = 5e-4  # finer steps
    >>> aligner.run_synthetic_raster_xcorr()

    See Also
    --------
    _build_synthetic_raster_from_fsi : FSI-to-synthetic-raster conversion
    _run_shift_scale_search : Core 4D optimizer
    build_synthetic_raster_from_maps : Underlying interpolation engine (help_funcs)
    run_one_map_xcorr : Phase 2 refinement (should run first)
    """
    _vprint(self.verbose, 1, "Running synthetic raster cross-correlation phase...")
    base_map = self._resolve_synthetic_base_map()
    synthetic_map, sources = self._build_synthetic_raster_from_fsi(base_map)
    synth_data = np.asarray(synthetic_map.data, dtype=np.float64)
    target_data = np.asarray(base_map.data, dtype=np.float64)
    meta = cast(Mapping[str, Any], base_map.meta)
    center = self.synthetic_kwargs.get('center')
    if center is None:
      crpix1 = float(meta.get('crpix1', target_data.shape[1] / 2.0))
      crpix2 = float(meta.get('crpix2', target_data.shape[0] / 2.0))
      center = (crpix1, crpix2)
    elif isinstance(center, Sequence) and len(center) >= 2:
      center = (float(center[0]), float(center[1]))
    else:
      raise ValueError("synthetic_kwargs.center must be a 2-element sequence if provided.")

    kwargs = self.synthetic_kwargs
    synthetic_n_jobs = int(kwargs.get('n_jobs') or self.n_jobs or 1)
    shift_range = kwargs.get('shift_range')
    if not isinstance(shift_range, Sequence) or len(shift_range) != 2:
      raise ValueError("synthetic_kwargs must define a shift_range tuple for discrete search.")
    shift_x, shift_y = int(shift_range[0]), int(shift_range[1])

    kwargs['reference_data'] = synth_data
    kwargs['target_data'] = target_data
    kwargs['center'] = center
    kwargs['crpix1'] = float(center[0])
    kwargs['crpix2'] = float(center[1])
    kwargs['n_jobs'] = synthetic_n_jobs
    # Add map objects for blink comparison rendering
    kwargs['reprojected_reference_on_extended_target_map'] = synthetic_map
    kwargs['extended_target_map'] = base_map

    search_result, _best_point = self._run_shift_scale_search(
      kwargs,
      'synthetic_raster',
      'synthetic',
      shift_x,
      shift_y,
    )

    best_summary = cast(Dict[str, float], search_result.get('best') or {})
    best_params = {
      "dx": float(best_summary.get('dx', 0.0)),
      "dy": float(best_summary.get('dy', 0.0)),
      "squeeze_x": float(best_summary.get('sx', 1.0)),
      "squeeze_y": float(best_summary.get('sy', 1.0)),
      "corr": float(best_summary.get('corr', 0.0)),
    }

    corrected_map = make_corrected_wcs_map(
      base_map,
      best_params,
      verbose=self.verbose,
    )

    def _describe_source(entry: GenericMap | Path | str) -> str:
      if isinstance(entry, Path):
        return str(entry)
      if isinstance(entry, str):
        return entry
      meta = getattr(entry, 'meta', {})
      if isinstance(meta, Mapping):
        filename = meta.get('filename') or meta.get('FILENAME')
        if filename:
          return str(filename)
      return entry.__class__.__name__

    search_result.update({
      "best_params": best_params,
      "synthetic_map": synthetic_map,
      "corrected_target_map": corrected_map,
      "source_count": len(sources),
      "sources": [_describe_source(src) for src in sources],
    })
    self.results['synthetic_raster'] = search_result
    self.procedures['synthetic_raster'] = True
    self.best_params = best_params
    _vprint(
      self.verbose,
      1,
      (
        f"[synthetic] corr={best_params.get('corr', 0.0):.6f} at "
        f"dx={best_params.get('dx', 0.0):.2f}, dy={best_params.get('dy', 0.0):.2f}, "
        f"sx={best_params.get('squeeze_x', 1.0):.5f}, sy={best_params.get('squeeze_y', 1.0):.5f}."
      ),
    )

  def _build_corrected_target_map(
    self,
    *,
    target_map: GenericMap,
    extended_target_map: GenericMap | None,
    pad_x: int,
    pad_y: int,
    kernel: Tuple[int, int],
    best_shift: Tuple[int, int],
  ) -> Tuple[
    GenericMap | None,
    GenericMap | None,
    Dict[str, float] | None,
    Tuple[float, float] | None,
  ]:
    """Rebuild the corrected map for the provided phase inputs."""
    working_map: GenericMap | None = extended_target_map if isinstance(extended_target_map, GenericMap) else None
    if working_map is None:
      working_map = cast(GenericMap, enlarge_map_by_padding(target_map, pad_x=pad_x, pad_y=pad_y))
    if working_map is None:
      return (None, None, None, None)

    kernel_rows = max(1, int(kernel[0]))
    kernel_cols = max(1, int(kernel[1]))
    dx_unbinned = float(best_shift[0]) * float(kernel_cols)
    dy_unbinned = float(best_shift[1]) * float(kernel_rows)
    best_params = {
      "dx": dx_unbinned,
      "dy": dy_unbinned,
      "squeeze_x": 1.0,
      "squeeze_y": 1.0,
    }

    corrected_extended = make_corrected_wcs_map(
      working_map,
      best_params,
      verbose=self.verbose,
    )

    corrected_cropped: GenericMap | None = corrected_extended
    if pad_x or pad_y:
      corrected_cropped = enlarge_map_by_padding(
        corrected_extended,
        pad_x=-pad_x,
        pad_y=-pad_y,
      )
      _vprint(self.verbose, 1, "coordinates have been corrected ...")
      _vprint(self.verbose, 1, f"CRVAL1: {u.Quantity(target_map.meta.get('CRVAL1'),target_map.meta.get('CUNIT1')).to(u.arcsec).to_value():.2f} -> {u.Quantity(corrected_cropped.meta.get('CRVAL1'),corrected_cropped.meta.get('CUNIT1')).to(u.arcsec).to_value():.2f}")
      _vprint(self.verbose, 1, f"CRVAL2: {u.Quantity(target_map.meta.get('CRVAL2'),target_map.meta.get('CUNIT2')).to(u.arcsec).to_value():.2f} -> {u.Quantity(corrected_cropped.meta.get('CRVAL2'),corrected_cropped.meta.get('CUNIT2')).to(u.arcsec).to_value():.2f}")
      _vprint(self.verbose, 1, f"CDELT1: {u.Quantity(target_map.meta.get('CDELT1'),target_map.meta.get('CUNIT1')).to(u.arcsec).to_value():.2f} -> {u.Quantity(corrected_cropped.meta.get('CDELT1'),corrected_cropped.meta.get('CUNIT1')).to(u.arcsec).to_value():.2f}")
      _vprint(self.verbose, 1, f"CDELT2: {u.Quantity(target_map.meta.get('CDELT2'),target_map.meta.get('CUNIT2')).to(u.arcsec).to_value():.2f} -> {u.Quantity(corrected_cropped.meta.get('CDELT2'),corrected_cropped.meta.get('CUNIT2')).to(u.arcsec).to_value():.2f}")
      _vprint(self.verbose, 1, f"PC1_1 : {target_map.meta.get('PC1_1', 1.0):.2f} -> {corrected_cropped.meta.get('PC1_1', 1.0):.2f}")
      _vprint(self.verbose, 1, f"PC1_2 : {target_map.meta.get('PC1_2', 0.0):.2f} -> {corrected_cropped.meta.get('PC1_2', 0.0):.2f}")
      _vprint(self.verbose, 1, f"PC2_1 : {target_map.meta.get('PC2_1', 0.0):.2f} -> {corrected_cropped.meta.get('PC2_1', 0.0):.2f}")
      _vprint(self.verbose, 1, f"PC2_2 : {target_map.meta.get('PC2_2', 1.0):.2f} -> {corrected_cropped.meta.get('PC2_2', 1.0):.2f}")    
    return (corrected_cropped, corrected_extended, best_params, (dx_unbinned, dy_unbinned))

  def _finalize_phase_result(
    self,
    phase_key: Literal['binning', 'one_map'],
    search_result: Dict[str, Any],
    corrected_payload: Tuple[
      GenericMap | None,
      GenericMap | None,
      Dict[str, float] | None,
      Tuple[float, float] | None,
    ],
  ) -> None:
    """Attach corrected map artifacts to the phase result and persist it."""
    corrected_cropped, corrected_extended, best_params, unbinned_shift = corrected_payload
    best_summary = search_result.get("best")
    if isinstance(best_summary, dict) and unbinned_shift is not None:
      best_summary["dx_unbinned"] = float(unbinned_shift[0])
      best_summary["dy_unbinned"] = float(unbinned_shift[1])
    if corrected_cropped is not None:
      search_result["corrected_target_map"] = corrected_cropped
    if corrected_extended is not None:
      search_result["corrected_extended_target_map"] = corrected_extended
    if best_params is not None:
      search_result["best_params"] = best_params
      self.best_params = best_params
    self.results[phase_key] = search_result
    self.procedures[phase_key] = True
    
    # Render blink comparison if debug session is active
    if self._shared_debug_writer is not None and corrected_extended is not None:
      cfg = self.xcorr_binned_kwargs if phase_key == 'binning' else self.xcorr_one_map_kwargs
      ref_map = cfg.get('reprojected_reference_on_extended_target_map')
      target_map = cfg.get('extended_target_map')
      if isinstance(ref_map, GenericMap) and isinstance(target_map, GenericMap):
        from coalign_debug import DebugPlotContext
        # Create temporary debug context for animation rendering
        temp_ctx = DebugPlotContext(
          pdf_writer=self._shared_debug_writer,
          fig=None,
          ax=None,
          color_mappable=None,
          debug_points=[],
          plotted_points=set(),
          pdf_path=self._shared_debug_pdf_path,
          owns_writer=False,
        )
        _vprint(self.verbose, 2, f"Rendering comparison animation for {phase_key} phase...")
        temp_ctx.render_comparison_animation(ref_map, target_map, corrected_extended, phase_name=phase_key)

  def _run_shift_search(self, phase_key: Literal['binning', 'one_map']) -> Tuple[Dict[str, Any], Tuple[int, int]]:
    """Search the shift grid for the requested phase and return the best result."""
    cfg = self.xcorr_binned_kwargs if phase_key == 'binning' else self.xcorr_one_map_kwargs
    label = 'binned' if phase_key == 'binning' else 'one-map'
    _vprint(self.verbose,1,f"Running cross-correlation on {label} data...")
    if cfg['reference_data'] is None or cfg['target_data'] is None:
      raise ValueError(
        f"Both reference_data and target_data must be prepared before running the {label} search.",
      )

    ref_img = np.asarray(cfg['reference_data'], dtype=np.float64)
    target_img = np.asarray(cfg['target_data'], dtype=np.float64)
    shift_range = cfg.get('shift_range')
    if shift_range is None:
      raise ValueError(f"shift_range must be configured before running the {label} search.")
    shift_x, shift_y = int(shift_range[0]), int(shift_range[1])
    scale_range_cfg = cfg.get('scale_range')
    if scale_range_cfg:
      return self._run_shift_scale_search(
        cfg,
        phase_key,
        label,
        shift_x,
        shift_y,
      )
    dr0 = np.array([
      float(cfg.get('dx0') or 0.0),
      float(cfg.get('dy0') or 0.0),
    ], dtype=float)
    scale_x_value: float = float(cfg.get('scale_x', 1.0))
    scale_y_value: float = float(cfg.get('scale_y', 1.0))
    n_neighbors: int = max(1, int(cfg.get('n_neighbors') or self.n_jobs * 10))
    n_jobs: int = max(1, int(self.n_jobs))
    max_corr_goal: float = float(cfg.get('max_corr', 0.7 if phase_key == 'binning' else -1.0))
    force_plateau = max_corr_goal < 0.0
    if force_plateau:
      max_corr_goal = float('inf')
    corr_atol: float = float(cfg.get('corr_atol', 1e-4))
    corr_rtol: float = float(cfg.get('corr_rtol', 1e-3))
    plateau_iters: int = int(cfg.get('plateau_iters', 3))
    _vprint(
      self.verbose,
      2,
      f"Initial shift guess: {dr0} Pix, shift search range: ({shift_x},{shift_y})Pix, "
      f"max_corr_goal: {max_corr_goal:04.2f}, corr_atol: {corr_atol:g}, corr_rtol: {corr_rtol:g}, "
      f"plateau_iters: {plateau_iters:02d}",
    )
    center_pix: Tuple[float, float] | None = None
    if phase_key == 'binning':
      if cfg['binned_crpix1'] is not None and cfg['binned_crpix2'] is not None:
        center_pix = (
          float(cfg['binned_crpix1']),
          float(cfg['binned_crpix2']),
        )
    elif cfg['crpix1'] is not None and cfg['crpix2'] is not None:
      center_pix = (
        float(cfg['crpix1']),
        float(cfg['crpix2']),
      )

    self._ensure_corr_workers(ref_img, target_img, center_pix)
    effective_workers = self.n_jobs if self.n_jobs > 1 else 1
    _vprint(self.verbose,2,f"Using {effective_workers} worker{'s' if effective_workers>1 else ''} computing up to {n_neighbors} neighbours each iteration...")

    debug_plots = abs(self.verbose) >= 3
    debug_ctx: DebugPlotContext | None = None
    debug_pdf_path: Path | None = None
    session_started_here = False

    if debug_plots:
      if self._debug_session_depth == 0:
        self._push_debug_session()
        session_started_here = True
      debug_ctx = self._create_debug_context(shift_x, shift_y, phase_key)
      debug_pdf_path = self._shared_debug_pdf_path
      if debug_ctx is not None:
        debug_ctx.reset_scatter_size()
      if debug_pdf_path is not None:
        _vprint(self.verbose,2,f"Saving cross-correlation search progression plots to {debug_pdf_path}")
    all_points, neighbor_offsets = build_shift_structures(shift_x, shift_y)
    _vprint(self.verbose,2,f"Total shift points: {len(all_points)}...")

    cache: Dict[Tuple[int, int], float] = {}
    history: List[Dict[str, float]] = []
    evaluated_count = 0
    total_points = (2 * shift_x + 1) * (2 * shift_y + 1)
    center_trace: List[Tuple[int, int]] = []

    task_queue = self._task_queue
    result_queue = self._result_queue
    has_workers = bool(
      self._corr_workers
      and task_queue
      and result_queue
      and self._shared_payloads is not None
      and self._current_payload_id is not None
    )

    def record_corr(dx: int, dy: int, corr_val: float, sx: float, sy: float) -> None:
      """Cache a correlation value and forward it to the debug plot if needed."""
      key = (dx, dy)
      cache[key] = corr_val
      history.append({
        "dx": float(dx),
        "dy": float(dy),
        "sx": float(sx),
        "sy": float(sy),
        "corr": float(corr_val),
      })
      if debug_ctx is not None:
        debug_ctx.add_point(dx, dy, corr_val)

    def evaluate_points(points: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
      """Evaluate each shift either locally or via the worker pool."""
      results: List[Tuple[int, int, float]] = []
      if not points:
        return results

      worker_points: List[Tuple[int, int]] = []
      for dx, dy in points:
        key = (dx, dy)
        if key in cache:
          results.append((dx, dy, cache[key]))
          continue

        if has_workers and task_queue and result_queue:
          worker_points.append((dx, dy))
          continue

        corr_val = correlation_for_params(
          ref_img,
          target_img,
          float(dx),
          float(dy),
          scale_x_value,
          scale_y_value,
          center=center_pix,
        )
        record_corr(dx, dy, corr_val, scale_x_value, scale_y_value)
        results.append((dx, dy, corr_val))

      if worker_points and has_workers and task_queue and result_queue:
        batch_size = max(1, int(math.ceil(len(worker_points) / max(1, effective_workers))))

        def _chunk_points(seq: List[Tuple[int, int]], size: int) -> Iterable[List[Tuple[int, int]]]:
          """Yield `seq` in chunks of size `size` to smooth job queue pressure."""
          for idx in range(0, len(seq), size):
            yield seq[idx : idx + size]

        pending_jobs: set[int] = set()
        for chunk in _chunk_points(worker_points, batch_size):
          job_id = self._next_job_id
          self._next_job_id += 1
          dxs = [float(pt[0]) for pt in chunk]
          dys = [float(pt[1]) for pt in chunk]
          sxs = [scale_x_value] * len(chunk)
          sys = [scale_y_value] * len(chunk)
          payload_ids = [self._current_payload_id] * len(chunk)
          task_queue.put((job_id, payload_ids, dxs, dys, sxs, sys))
          pending_jobs.add(job_id)

        while pending_jobs:
          job_id, batch = result_queue.get()
          pending_jobs.discard(job_id)
          for dx_val, dy_val, corr_val in batch:
            dx_i = int(dx_val)
            dy_i = int(dy_val)
            record_corr(dx_i, dy_i, corr_val, scale_x_value, scale_y_value)
            results.append((dx_i, dy_i, corr_val))

      return results

    try:
      current_center = clamp_point((int(round(dr0[0])), int(round(dr0[1]))), shift_x, shift_y)
      if current_center not in cache:
        evaluate_points([current_center])
      center_trace.append(current_center)
      best_point, best_corr = max(cache.items(), key=lambda item: item[1])
      global_best_corr = best_corr
      search_phase = 2 if force_plateau else 1
      plateau_counter = 0
      last_best_increase = best_corr
      iteration_number : int = 0
      stall_counts: Dict[Tuple[int, int], int] = defaultdict(int)

      while True:
        if len(cache) >= total_points:
          _vprint(self.verbose,2,"All shift positions evaluated; stopping.")
          break
        iteration_number += 1
        _vprint(
          self.verbose,
          2,
          f"Iteration {iteration_number}: current center {current_center}, cache size {len(cache)}/{total_points}, phase {search_phase}...",
        )
        neighbor_limit = min(n_neighbors, total_points - len(cache))
        new_points = gather_neighbors(
          current_center,
          neighbor_offsets,
          cache,
          neighbor_limit,
          shift_x,
          shift_y,
        )
        evaluated = evaluate_points(new_points)
        evaluated_count = len(cache)

        local_points = [(current_center[0], current_center[1], cache[current_center])]
        local_points.extend(evaluated)

        if debug_ctx is not None:
          debug_ctx.render_iteration(current_center, search_phase)

        best_point, best_corr = max(cache.items(), key=lambda item: item[1])
        if best_corr > global_best_corr:
          global_best_corr = best_corr
        _vprint(self.verbose,2,f"Best local corr: {cache[current_center]:.5f}, best overall corr: {global_best_corr:.5f} at shift {best_point}.")
        search_phase, plateau_counter, last_best_increase, plateau_met, phase_switched = update_phase_state(
          search_phase,
          best_corr,
          last_best_increase,
          plateau_counter,
          max_corr_goal,
          corr_atol,
          corr_rtol,
          plateau_iters,
        )
        if phase_switched and not force_plateau:
          _vprint(self.verbose,2,f"Switching to plateau phase at corr={best_corr:.5f}")
        if plateau_met:
          _vprint(self.verbose,2,"Plateau condition met; stopping.")
          break
        
        _vprint(self.verbose,2,f"Computing next step ...")
        step_vec = compute_gradient_step(local_points, current_center, best_point)
        candidate_center = clamp_point(
          (current_center[0] + int(step_vec[0]), current_center[1] + int(step_vec[1])),
          shift_x,
          shift_y,
        )
        _vprint(self.verbose,2,f"Finding next candidate center from gradient step ...")
        while candidate_center in cache and candidate_center != current_center:
          stall_counts[candidate_center] += 1
          if stall_counts[candidate_center] >= 2:
            fallback_local = first_unvisited(candidate_center, all_points, cache)
            _vprint(self.verbose,2,f"Stalling at {candidate_center} with corr {cache[candidate_center]:.5f}; trying fallback local {fallback_local}...")
            if fallback_local is not None:
              candidate_center = fallback_local
              _vprint(self.verbose,2,f"Falling back to {candidate_center} for next candidate center.")
              break
            _vprint(self.verbose,2,f"No fallback found for stalled candidate {candidate_center}; breaking out of stall loop.")
            break
          next_candidate = clamp_point(
            (candidate_center[0] + int(step_vec[0]), candidate_center[1] + int(step_vec[1])),
            shift_x,
            shift_y,
          )
          if next_candidate == candidate_center:
            break
          candidate_center = next_candidate
          if candidate_center == current_center or not point_in_bounds(candidate_center, shift_x, shift_y):
            break

        if candidate_center == current_center or not point_in_bounds(candidate_center, shift_x, shift_y) or candidate_center in cache:
          fallback = first_unvisited(current_center, all_points, cache)
          if fallback is None:
            _vprint(self.verbose,2,"No remaining unvisited positions; stopping.")
            break
          candidate_center = fallback

        current_center = candidate_center
        center_trace.append(current_center)
        if current_center not in cache:
          evaluate_points([current_center])

      best_summary: Dict[str, float] = {
        "dx": float(best_point[0]),
        "dy": float(best_point[1]),
        "sx": float(scale_x_value),
        "sy": float(scale_y_value),
        "corr": float(best_corr),
      }

      result_payload: Dict[str, Any] = {
        "best": best_summary,
        "history": history,
        "centers": center_trace,
        "evaluated": evaluated_count,
        "total": total_points,
        "phase": "plateau" if search_phase == 2 else "search",
      }
      ref_map_obj = cfg.get('reprojected_reference_on_extended_target_map')
      target_map_obj = cfg.get('extended_target_map')
      if debug_ctx is not None and cache:
        # Render history plot (adaptive based on whether scale varies)
        history_array = np.array([[i, h['dx'], h['dy'], h.get('sx', 1.0), h.get('sy', 1.0), h['corr']] for i, h in enumerate(history)], dtype=float)
        debug_ctx.render_history_plot(history_array)
        
        # Build corrected map for visualization
        corrected_for_viz = None
        if isinstance(target_map_obj, GenericMap):
          try:
            dx_val = float(best_point[0])
            dy_val = float(best_point[1])
            temp_params = {"dx": dx_val, "dy": dy_val, "squeeze_x": 1.0, "squeeze_y": 1.0}
            corrected_for_viz = make_corrected_wcs_map(target_map_obj, temp_params, verbose=0)
          except Exception:
            pass
        
        debug_ctx.render_alignment_overlay(
          ref_img,
          target_img,
          best_point,
          ref_map=ref_map_obj if isinstance(ref_map_obj, GenericMap) else None,
          target_map=target_map_obj if isinstance(target_map_obj, GenericMap) else None,
          corrected_map=corrected_for_viz,
        )
      self._release_payload_data()
      _vprint(self.verbose,1,f"[{label}] Best correlation found: {best_corr:.6f} at shift ({best_point[0]}, {best_point[1]}).")
      return (result_payload, best_point)
    finally:
      if debug_ctx is not None:
        debug_ctx.close()
        if debug_pdf_path is not None:
          _vprint(self.verbose, 2, f"Saved xcorr debug trace to {debug_pdf_path}")
      if session_started_here:
        self._pop_debug_session()

       