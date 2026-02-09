"""Synthetic raster generation and optimization process."""

import time
from pathlib import Path
from typing import Tuple

import astropy.units as u
import numpy as np
from sunpy.map import GenericMap

from .base_node import CoalignmentNode
from .process_result import ProcessResult
from .utils import (
    arcsec_to_pixels,
    apply_shift_and_scale_to_map,
    pixels_to_arcsec,
)
from .optimization import optimize_shift_and_scale
from help_funcs import _vprint, build_synthetic_raster_from_maps
from .debug_plot import create_debug_context


class SyntheticRasterProcess(CoalignmentNode):
    """
    Synthetic raster generation + shift + scale optimization.
    
    Builds a synthetic raster from a temporal sequence of FSI images,
    then performs shift + scale optimization.
    
    Parameters
    ----------
    max_shift : u.Quantity
        Maximum shift to search (must have angle units)
    scale_step : float
        Step size for scale search (e.g., 0.005 = 0.5% steps)
    scale_range : tuple
        Scale search bounds (min, max)
    n_neighbors : int
        Number of neighbors to evaluate per iteration
    max_corr : float
        Target correlation to reach (if < 0, run to plateau)
    plateau_iters : int
        Iterations without improvement before stopping
    """
    
    def __init__(
        self,
        max_shift: u.Quantity = 15.0 * u.arcsec,
        scale_step: float = 0.01,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        n_neighbors: int = 120,
        max_corr: float = -1.0,
        plateau_iters: int = 6,
    ):
        super().__init__()
        
        # Validate units
        if not isinstance(max_shift, u.Quantity):
            raise TypeError("max_shift must be astropy Quantity with angle units")
        if not max_shift.unit.is_equivalent(u.arcsec):
            raise ValueError(f"max_shift must have angle units")
        
        if scale_step < 0:
            raise ValueError("scale_step must be >= 0 (use 0 for shift-only)")
        
        # Store configuration
        self.max_shift = max_shift.to(u.arcsec)
        self.scale_step = scale_step
        self.scale_range = scale_range if scale_step > 0 else None
        self.n_neighbors = n_neighbors
        self.max_corr = max_corr
        self.plateau_iters = plateau_iters
        
        # Set node identity based on whether scale optimization is enabled
        if scale_step > 0:
            self.node_id = "synthetic_raster_shift_scale"
            self.node_name = "Synthetic Raster Process (Shift+Scale)"
        else:
            self.node_id = "synthetic_raster_shift_only"
            self.node_name = "Synthetic Raster Process (Shift-Only)"
    
    def _execute_own_process(self, working_map: GenericMap) -> None:
        """Execute synthetic raster generation and shift+scale optimization."""
        verbose = self.get_verbose_level()
        start_time = time.time()
        
        _vprint(verbose, 1, f"\n{'='*60}")
        _vprint(verbose, 1, f"Starting {self.node_name}")
        _vprint(verbose, 1, f"{'='*60}")
        
        # Get resources
        # base_target = self.get_base_target_map()
        reference_sequence = self.get_reference_sequence()
        workers = self.get_worker_pool()
        
        _vprint(verbose, 2, f"Configuration:")
        _vprint(verbose, 2, f"  Max shift: {self.max_shift}")
        _vprint(verbose, 2, f"  Scale step: {self.scale_step}")
        _vprint(verbose, 2, f"  Scale range: {self.scale_range}")
        _vprint(verbose, 2, f"  N neighbors: {self.n_neighbors}")
        _vprint(verbose, 2, f"  Max corr: {self.max_corr}")
        _vprint(verbose, 2, f"  Plateau iters: {self.plateau_iters}")
        _vprint(verbose, 2, f"  Reference sequence: {len(reference_sequence)} FSI images")
        
        # Build synthetic raster
        _vprint(verbose, 1, f"Building synthetic raster from {len(reference_sequence)} FSI images...")
        synthetic_raster = build_synthetic_raster_from_maps(
            # spice_map=base_target,
            spice_map=working_map,
            fsi_maps=reference_sequence,
            verbose=verbose
        )
        _vprint(verbose, 2, f"  Synthetic raster shape: {synthetic_raster.data.shape}")
        
        # Convert units to pixels (separate X and Y)
        max_shift_pixels_x, max_shift_pixels_y = arcsec_to_pixels(self.max_shift, working_map)
        _vprint(verbose, 2, f"  Max shift (pixels): X={max_shift_pixels_x}, Y={max_shift_pixels_y}")
        
        # Run optimization
        if self.scale_step > 0:
            _vprint(verbose, 1, "Running shift+scale optimization...")
        else:
            _vprint(verbose, 1, "Running shift-only optimization...")
        
        # Get center pixel from working map CRPIX
        center_pix = (
            float(working_map.meta.get('CRPIX1', working_map.data.shape[1] / 2.0)),
            float(working_map.meta.get('CRPIX2', working_map.data.shape[0] / 2.0))
        )
        _vprint(verbose, 2, f"  Center pixel (CRPIX): ({center_pix[0]:.2f}, {center_pix[1]:.2f})")
        
        # Create debug context BEFORE optimization (like old code)
        debug_ctx = None
        debug_writer = self.get_debug_writer()
        output_dir = self.get_output_directory()
        
        if verbose >= 3 and debug_writer is not None and output_dir is not None:
            # Calculate scatter point radius based on search range (like old code)
            max_span = max(max_shift_pixels_x, max_shift_pixels_y)
            computed_radius = max_span * 0.015
            base_radius = min(0.9, max(0.35, computed_radius))
            base_dpi = min(320, 140 + int(max_span * 1.8))
            
            debug_ctx = create_debug_context(
                shift_x=max_shift_pixels_x,
                shift_y=max_shift_pixels_y,
                debug_dir=output_dir,
                pdf_writer=debug_writer,
                pdf_path=output_dir / f"debug_{self.node_id}.pdf",
                close_writer_on_exit=False,
                point_radius_data=base_radius,
                dpi=base_dpi,
                process_label=f"{self.node_name} [{self.node_id}]",
            )
            # Reset scatter size before optimization (like old code)
            if debug_ctx is not None:
                debug_ctx.reset_scatter_size()
        
        best_params, iterations, history = optimize_shift_and_scale(
            target_data=np.asarray(working_map.data, dtype=np.float64),
            reference_data=np.asarray(synthetic_raster.data, dtype=np.float64),
            max_shift=(max_shift_pixels_x, max_shift_pixels_y),
            scale_step=self.scale_step,  # SHIFT+SCALE (4D search)
            scale_range=self.scale_range,
            workers=workers,
            center_pix=center_pix,
            n_neighbors=self.n_neighbors,
            max_corr=self.max_corr,
            plateau_iters=self.plateau_iters,
            verbose=verbose,
            debug_ctx=debug_ctx,
            task_queue=self.get_task_queue(),
            result_queue=self.get_result_queue(),
            shared_payloads=self.get_shared_payloads(),
        )
        
        # Convert shift to arcsec (separate X and Y)
        shift_x_arcsec, shift_y_arcsec = pixels_to_arcsec((best_params['dx'], best_params['dy']), working_map)
        
        # Extract scale factors (default to 1.0 for shift-only)
        scale_x = best_params.get('sx', 1.0)
        scale_y = best_params.get('sy', 1.0)
        
        _vprint(verbose, 1, f"\nOptimization complete:")
        _vprint(verbose, 1, f"  Iterations: {iterations}")
        _vprint(verbose, 1, f"  Shift: ({shift_x_arcsec:.2f}, {shift_y_arcsec:.2f})")
        if self.scale_step > 0:
            _vprint(verbose, 1, f"  Scale: ({scale_x:.4f}, {scale_y:.4f})")
        _vprint(verbose, 1, f"  Correlation: {best_params['corr']:.4f}")
        
        # Apply transformation
        _vprint(verbose, 2, "Applying shift and scale to map...")
        corrected_map = apply_shift_and_scale_to_map(
            working_map,
            shift_x_arcsec,
            shift_y_arcsec,
            scale_x,
            scale_y
        )
        
        # Generate debug output
        debug_pdf_path = None
        animation_path = None
        
        if verbose >= 3 and debug_ctx is not None and output_dir is not None:
            _vprint(verbose, 3, "Generating debug visualizations...")
            
            # Render history plot (exactly like old code)
            history_array = np.array([[h['iteration'], h['dx'], h['dy'], h.get('sx', 1.0), h.get('sy', 1.0), h['corr']] for h in history], dtype=float)
            debug_ctx.render_history_plot(history_array)
            
            # Render alignment overlay (exactly like old code)
            if self.scale_step > 0:
                title = (f"{self.node_name} [{self.node_id}]\n"
                        f"Scale=({scale_x:.4f}, {scale_y:.4f}), "
                        f"Corr={best_params['corr']:.4f}")
            else:
                title = (f"{self.node_name} [{self.node_id}]\n"
                        f"Shift=({best_params['dx']:.1f}, {best_params['dy']:.1f}), "
                        f"Corr={best_params['corr']:.4f}")
            debug_ctx.ax.set_title(title)
            debug_ctx.render_alignment_overlay(
                synthetic_raster.data.astype(np.float64),
                working_map.data.astype(np.float64),
                (int(best_params['dx']), int(best_params['dy'])),
                ref_map=synthetic_raster,
                target_map=working_map,
                corrected_map=corrected_map,
            )
            
            # Create blink animation
            animation_path = output_dir / f"blink_{self.node_id}.gif"
            _vprint(verbose, 3, f"  Creating blink animation: {animation_path}")
            debug_ctx.render_comparison_animation(
                ref_map=synthetic_raster,
                target_map=working_map,
                corrected_map=corrected_map,
                phase_name=self.node_name,
                interval=800,
                n_cycles=3
            )
            
            debug_pdf_path = str(output_dir / f"debug_{self.node_id}.pdf")
        
        # Store result
        execution_time = time.time() - start_time
        self.result = ProcessResult(
            process_id=self.node_id,
            process_name=self.node_name,
            input_map=working_map,
            output_map=corrected_map,
            shift_arcsec=(shift_x_arcsec, shift_y_arcsec),
            shift_pixels=(best_params['dx'], best_params['dy']),
            scale_factors=(scale_x, scale_y),
            correlation_peak=best_params['corr'],
            search_space_explored=0,
            iteration_count=iterations,
            execution_time=execution_time,
            debug_pdf_path=debug_pdf_path,
            animation_path=animation_path,
            reference_reprojected=synthetic_raster,  # Store synthetic raster as "reference_reprojected" for potential debugging
        )
        
        self.is_executed = True
        
        _vprint(verbose, 1, f"Process complete in {execution_time:.1f}s")
        _vprint(verbose, 1, f"{'='*60}\n")
