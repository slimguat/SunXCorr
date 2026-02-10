"""Single-map cross-correlation process (shift-only optimization)."""

from __future__ import annotations

import time
from typing import cast

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from sunpy.map import GenericMap, Map

from .base_node import CoalignmentNode

# from slimfunc_correlation_effort import reproject_map_to_reference
from .debug_plot import create_debug_context
from .optimization import optimize_shift_and_scale
from .process_result import ProcessResult
from .utils import (  # pixels_to_arcsec,
    _vprint,
    apply_shift_to_map,
    arcsec_to_pixels,
    bin_map,
    enlarge_map_by_padding,
    get_pixel_scale_quantity,
    reproject_map_to_reference,
    unbin_map_lattice,
)


class SingleMapProcess(CoalignmentNode):
    """
    Single-map cross-correlation process with optional binning.

    Performs shift-only optimization (no scale adjustment).
    Can operate at binned or full resolution depending on bin_kernel.

    Parameters
    ----------
    max_shift : u.Quantity
        Maximum shift to search (must have angle units)
    bin_kernel : u.Quantity
        Binning kernel size in arcseconds (0 = no binning)
    n_neighbors : int
        Number of neighbors to evaluate per iteration
    max_corr : float
        Target correlation to reach (if < 0, run to plateau)
    plateau_iters : int
        Iterations without improvement before stopping
    """

    def __init__(
        self,
        max_shift: u.Quantity = 10.0 * u.arcsec,
        bin_kernel: u.Quantity = 0.0 * u.arcsec,
        n_neighbors: int = 50,
        max_corr: float = -1.0,
        plateau_iters: int = 3,
    ):
        super().__init__()

        # Validate units
        if not isinstance(max_shift, u.Quantity):
            raise TypeError("max_shift must be astropy Quantity with angle units")
        if not max_shift.unit.is_equivalent(u.arcsec):
            raise ValueError(f"max_shift must have angle units, got {max_shift.unit}")

        if not isinstance(bin_kernel, u.Quantity):
            raise TypeError("bin_kernel must be astropy Quantity with angle units")
        if not bin_kernel.unit.is_equivalent(u.arcsec):
            raise ValueError(f"bin_kernel must have angle units, got {bin_kernel.unit}")

        # Store configuration
        self.max_shift = max_shift.to(u.arcsec)
        self.bin_kernel = bin_kernel.to(u.arcsec)
        self.n_neighbors = n_neighbors
        self.max_corr = max_corr
        self.plateau_iters = plateau_iters

        # Set node identity
        bin_label = (
            f"{self.bin_kernel.value:.1f}as"
            if self.bin_kernel.value > 1
            else "full-res"
        )
        self.node_id = f"single_map_{bin_label}"
        self.node_name = f"Single Map Process ({bin_label})"

    def _execute_own_process(self, working_map: GenericMap) -> None:
        """Execute shift-only optimization.

        This method builds a reprojected reference image on the working map's
        grid, optionally bins both images, and runs a shift-only search using
        `optimize_shift_and_scale` with `scale_step=0`. Results are stored in
        `self.result` as a `ProcessResult` instance.

        Parameters
        ----------
        working_map : GenericMap
            The map to be corrected (target); may be binned internally for
            faster searches depending on `self.bin_kernel`.

        Returns
        -------
        None
            Results are written to `self.result` and `self.is_executed` is set
            to True on completion.
        """
        verbose = self.get_verbose_level()
        start_time = time.time()

        _vprint(verbose, 1, f"\n{'='*60}")
        _vprint(verbose, 1, f"Starting {self.node_name}")
        _vprint(verbose, 1, f"{'='*60}")

        # Get resources
        reference_map = self.get_reference_map()
        workers = self.get_worker_pool()

        _vprint(verbose, 2, "Configuration:")
        _vprint(verbose, 2, f"  Max shift: {self.max_shift}")
        _vprint(verbose, 2, f"  Bin kernel: {self.bin_kernel}")
        _vprint(verbose, 2, f"  N neighbors: {self.n_neighbors}")
        _vprint(verbose, 2, f"  Max corr: {self.max_corr}")
        _vprint(verbose, 2, f"  Plateau iters: {self.plateau_iters}")

        # Convert max_shift to pixels using original map (separate X and Y)
        max_shift_pixels_x, max_shift_pixels_y = arcsec_to_pixels(
            self.max_shift, working_map
        )

        # Reproject reference onto working map's grid
        extended_working_map = enlarge_map_by_padding(
            working_map, pad_x=max_shift_pixels_x, pad_y=max_shift_pixels_y
        )
        _vprint(verbose, 2, "  Reprojecting reference to working map grid...")
        reference_reprojected = reproject_map_to_reference(
            extended_working_map, reference_map
        )
        _vprint(
            verbose,
            2,
            f"  Reprojected reference shape: {cast(NDArray, reference_reprojected.data).shape}",
        )

        _vprint(
            verbose,
            2,
            f"  Max shift (pixels, unbinned): X={max_shift_pixels_x}, Y={max_shift_pixels_y}",
        )

        # Apply binning if requested
        if self.bin_kernel.value >= 1:
            bin_factor_x, bin_factor_y = arcsec_to_pixels(self.bin_kernel, working_map)
            bin_factor_x = max(1, bin_factor_x)
            bin_factor_y = max(1, bin_factor_y)
            _vprint(
                verbose,
                1,
                f"Applying binning: {bin_factor_x}x (X) × {bin_factor_y}x (Y)...",
            )
            working_binned = bin_map(extended_working_map, (bin_factor_x, bin_factor_y))
            reference_binned = bin_map(
                reference_reprojected, (bin_factor_x, bin_factor_y)
            )
            _vprint(
                verbose,
                1,
                f"  Working binned shape: {cast(NDArray, working_binned.data).shape}",
            )
            _vprint(
                verbose,
                1,
                f"  Reference binned shape: {cast(NDArray, reference_binned.data).shape}",
            )

            # Scale max_shift for binned pixels (separate X and Y)
            max_shift_binned_pixels_x = max(1, max_shift_pixels_x // bin_factor_x)
            max_shift_binned_pixels_y = max(1, max_shift_pixels_y // bin_factor_y)
            # Use tuple for asymmetric search (different X/Y pixel scales)
            max_shift_binned_pixels = (
                max_shift_binned_pixels_x,
                max_shift_binned_pixels_y,
            )
            _vprint(
                verbose,
                2,
                f"  Max shift (pixels, binned): X={max_shift_binned_pixels_x}, Y={max_shift_binned_pixels_y}",
            )
        else:
            _vprint(verbose, 1, "Running at full resolution (no binning)")
            working_binned = extended_working_map
            reference_binned = reference_reprojected
            # Use tuple for asymmetric search (different X/Y pixel scales)
            max_shift_binned_pixels = (max_shift_pixels_x, max_shift_pixels_y)
        _vprint(
            verbose,
            2,
            f"  Max shift (pixels, full-res): X={max_shift_binned_pixels[0]}, Y={max_shift_binned_pixels[1]}",
        )
        _vprint(
            verbose,
            2,
            f"  Working map shape: {cast(NDArray, working_binned.data).shape}",
        )
        _vprint(
            verbose,
            2,
            f"  Reference map shape: {cast(NDArray, reference_binned.data).shape}",
        )

        # Run optimization (scale_step=0 for shift-only)
        _vprint(verbose, 1, "Running shift optimization...")

        # Get center pixel from binned map CRPIX
        center_pix = (
            float(
                working_binned.meta.get("CRPIX1", working_binned.data.shape[1] / 2.0)
            ),
            float(
                working_binned.meta.get("CRPIX2", working_binned.data.shape[0] / 2.0)
            ),
        )
        _vprint(
            verbose,
            2,
            f"  Center pixel (CRPIX): ({center_pix[0]:.2f}, {center_pix[1]:.2f})",
        )

        # Create debug context BEFORE optimization (like old code)
        debug_ctx = None
        debug_writer = self.get_debug_writer()
        output_dir = self.get_output_directory()

        if np.abs(verbose) >= 3 and debug_writer is not None and output_dir is not None:
            # Calculate scatter point radius based on search range (like old code)
            # Extract X and Y from tuple
            if isinstance(max_shift_binned_pixels, tuple):
                shift_x_debug, shift_y_debug = max_shift_binned_pixels
            else:
                shift_x_debug = shift_y_debug = max_shift_binned_pixels

            max_span = max(shift_x_debug, shift_y_debug)
            computed_radius = max_span * 0.015
            base_radius = min(0.9, max(0.35, computed_radius))
            base_dpi = 300
            if self.bin_kernel.value == 0 or max_span > 20:
                base_dpi = min(320, 140 + int(max_span * 1.8))

            debug_ctx = create_debug_context(
                shift_x=shift_x_debug,
                shift_y=shift_y_debug,
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
            target_data=np.asarray(working_binned.data, dtype=np.float64),
            reference_data=np.asarray(reference_binned.data, dtype=np.float64),
            max_shift=max_shift_binned_pixels,
            scale_step=0.0,  # SHIFT-ONLY (2D search)
            scale_range=None,
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

        # Convert shift back to arcsec (using binned pixel scale - separate X and Y)
        binned_pixel_scale_x, binned_pixel_scale_y = get_pixel_scale_quantity(
            working_binned
        )
        shift_x_arcsec = best_params["dx"] * binned_pixel_scale_x
        shift_y_arcsec = best_params["dy"] * binned_pixel_scale_y

        _vprint(verbose, 1, "\nOptimization complete:")
        _vprint(verbose, 1, f"  Iterations: {iterations}")
        _vprint(verbose, 1, f"  Shift: ({shift_x_arcsec:.2f}, {shift_y_arcsec:.2f})")
        _vprint(verbose, 1, f"  Correlation: {best_params['corr']:.4f}")

        # Apply shift to original (unbinned) map
        _vprint(verbose, 2, "Applying shift to map...")
        extended_corrected_map = apply_shift_to_map(
            extended_working_map, shift_x_arcsec, shift_y_arcsec
        )
        if self.bin_kernel.value >= 1:
            corrected_map = enlarge_map_by_padding(
                # unbinned_corrected_map,
                extended_corrected_map,
                pad_x=-max_shift_pixels_x,
                pad_y=-max_shift_pixels_y,
            )

            corrected_map = Map(
                working_map.data,
                corrected_map.meta,
                plot_settings=working_map.plot_settings,
            )
        else:
            corrected_map = Map(
                working_map.data,
                extended_corrected_map.meta,
                plot_settings=working_map.plot_settings,
            )

        # Generate debug output
        debug_pdf_path = None
        animation_path = None

        if np.abs(verbose) >= 3 and debug_ctx is not None and output_dir is not None:
            _vprint(verbose, 3, "Generating debug visualizations...")

            # Render history plot (exactly like old code)
            history_array = np.array(
                [
                    [
                        h["iteration"],
                        h["dx"],
                        h["dy"],
                        h.get("sx", 1.0),
                        h.get("sy", 1.0),
                        h["corr"],
                    ]
                    for h in history
                ],
                dtype=float,
            )
            debug_ctx.render_history_plot(history_array)

            # Render alignment overlay using BINNED maps (exactly like old code)
            debug_ctx.ax.set_title(
                f"{self.node_name} [{self.node_id}]\nCorr={best_params['corr']:.4f}"
            )
            debug_ctx.render_alignment_overlay(
                reference_binned.data.astype(np.float64),
                working_binned.data.astype(np.float64),
                (int(best_params["dx"]), int(best_params["dy"])),
                ref_map=reference_binned,
                target_map=working_binned,
                corrected_map=corrected_map,  # Don't show corrected in binned space
            )

            # Create blink animation
            animation_path = output_dir / f"blink_{self.node_id}.gif"
            _vprint(verbose, 3, f"  Creating blink animation: {animation_path}")
            debug_ctx.render_comparison_animation(
                ref_map=reference_binned,
                target_map=working_binned,
                corrected_map=extended_corrected_map,  # Show corrected in binned space for animation
                phase_name=self.node_name + "Binned maps",
                interval=800,
                n_cycles=3,
            )
            debug_ctx.render_comparison_animation(
                ref_map=reference_reprojected,
                target_map=working_map,
                corrected_map=corrected_map,
                phase_name=self.node_name,
                interval=800,
                n_cycles=3,
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
            shift_pixels=(best_params["dx"], best_params["dy"]),
            scale_factors=(1.0, 1.0),
            correlation_peak=best_params["corr"],
            search_space_explored=0,
            iteration_count=iterations,
            execution_time=execution_time,
            debug_pdf_path=debug_pdf_path,
            animation_path=animation_path,
            reference_reprojected=reference_reprojected,  # Store reprojected reference for potential debugging
        )

        self.is_executed = True

        _vprint(verbose, 1, f"Process complete in {execution_time:.1f}s")
        _vprint(verbose, 1, f"{'='*60}\n")
