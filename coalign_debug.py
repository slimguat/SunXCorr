"""Debug plotting context used by the coalignment search loop.

The helpers in this module encapsulate Matplotlib-related logic so that the
main optimizer can request PDF snapshots without pulling plotting code into the
core algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sunpy.map import GenericMap

import datetime

from coalign_helpers import phase_label
from help_funcs import get_coord_mat, normit


def _build_scatter_colormap() -> LinearSegmentedColormap:
  """Grey-to-vibrant ramp emphasizing detail above 0.5 correlation."""
  color_stops = [
    (0.0, "#1a1a1a"),
    (0.2, "#595959"),
    (0.4, "#9e9e9e"),
    (0.6, "#e0e0e0"),
    (0.63, "#00f5ff"),
    (0.66, "#00ff7f"),
    (0.69, "#a2ff00"),
    (0.72, "#ffe600"),
    (0.75, "#ff9600"),
    (0.78, "#ff005d"),
    (0.81, "#d000ff"),
    (0.84, "#5b4bff"),
    (0.88, "#00c2ff"),
    (0.92, "#5b4bff"),
    (0.96, "#d000ff"),
    (1.0, "#00f5ff"),
  ]
  return LinearSegmentedColormap.from_list("xcorr_scatter", color_stops)


SCATTER_COLORMAP = _build_scatter_colormap()

@dataclass
class DebugPlotContext:
  """Collects scatter points and writes them plus overlays to a PDF file."""

  pdf_writer: PdfPages
  fig: plt.Figure
  ax: plt.Axes
  color_mappable: ScalarMappable
  debug_points: List[Tuple[int, int, float]]
  plotted_points: Set[Tuple[int, int]]
  pdf_path: Path
  owns_writer: bool = True
  point_radius_x_data: float = 0.5
  point_radius_y_data: float = 0.5
  marker_patches: List[Rectangle] = field(default_factory=list)

  def add_point(self, dx: int, dy: int, corr_val: float) -> None:
    """Record a correlation sample if it has not been plotted yet."""
    key = (dx, dy)
    if key in self.plotted_points:
      return
    self.plotted_points.add(key)
    self.debug_points.append((dx, dy, corr_val))

  def render_iteration(self, center: Tuple[int, int], phase: int) -> None:
    """Render the scatter cloud plus current center indicator for one step."""
    if not self.debug_points:
      return
    self._remove_non_marker_patches()
    coords = np.array([[pt[0], pt[1]] for pt in self.debug_points], dtype=float)
    corrs = np.array([pt[2] for pt in self.debug_points], dtype=float)
    self._sync_marker_rectangles(coords, corrs)
    rect = Rectangle(
      (center[0] - 0.5, center[1] - 0.5),
      1.0,
      1.0,
      facecolor="none",
      edgecolor="red",
      linewidth=1.0,
    )
    self.ax.add_patch(rect)
    self.ax.set_title(f"Cross-correlation progression · {phase_label(phase)}")
    self.pdf_writer.savefig(self.fig)

  def render_blink_comparison(
      self,
      ref_map: GenericMap,
      target_map: GenericMap,
      corrected_map: GenericMap,
      phase_name: str = "",
      interval: int = 800,
      n_cycles: int = 5,
  ) -> None:
    """
    Render side-by-side blink animations showing before and after correction.
    Saves as GIF or MP4 file in the same directory as the debug PDF.
    
    Left: Reference map blinking with original uncorrected target
    Right: Reference map blinking with corrected target
    
    Parameters
    ----------
    ref_map : GenericMap
        Reference map (FSI or synthetic raster depending on phase)
    target_map : GenericMap
        Original uncorrected target map
    corrected_map : GenericMap
        Corrected target map
    phase_name : str
        Name of the phase (binning, one-map, synthetic) for filename
    interval : int
        Time between frames in milliseconds
    n_cycles : int
        Number of blink cycles
    """
    from slimfunc_correlation_effort import blink_maps
    
    # Create figure with two subplots side by side
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get coordinate matrices for all three maps
    lon_ref, lat_ref = get_coord_mat(ref_map)
    lon_tgt, lat_tgt = get_coord_mat(target_map)
    lon_corr, lat_corr = get_coord_mat(corrected_map)
    
    # Determine x/y limits from reference map with padding
    extx = 0.1
    exty = 0.1
    xlim = (
        lon_ref.min() - (lon_ref.max() - lon_ref.min()) * extx,
        lon_ref.max() + (lon_ref.max() - lon_ref.min()) * extx,
    )
    ylim = (
        lat_ref.min() - (lat_ref.max() - lat_ref.min()) * exty,
        lat_ref.max() + (lat_ref.max() - lat_ref.min()) * exty,
    )
    
    # --- BEFORE correction (left panel) ---
    im1_ref = ax_before.pcolormesh(
        lon_ref, lat_ref, ref_map.data,
        cmap="gray", norm=normit(ref_map.data), rasterized=True
    )
    im1_tgt = ax_before.pcolormesh(
        lon_tgt, lat_tgt, target_map.data,
        cmap=target_map.plot_settings.get("cmap", "sdoaia304"),
        norm=normit(target_map.data), rasterized=True
    )
    im1_tgt.set_visible(False)
    
    # Add contours for target on reference
    data_tgt = target_map.data[~np.isnan(target_map.data)]
    if data_tgt.size > 0:
        levels_tgt = np.percentile(data_tgt, [5, 10, 80, 90, 95, 99])
        cmap_tgt = plt.get_cmap('sdoaia304').reversed()
        im1_contour = ax_before.contour(
            lon_tgt, lat_tgt, target_map.data,
            levels=levels_tgt, cmap=cmap_tgt, linewidths=0.3, alpha=0.5
        )
    
    ax_before.set_xlim(xlim)
    ax_before.set_ylim(ylim)
    ax_before.set_xlabel("Tx (arcsec)")
    ax_before.set_ylabel("Ty (arcsec)")
    phase_label_str = f" ({phase_name})" if phase_name else ""
    ax_before.set_title(f"Before Correction{phase_label_str}")
    
    # --- AFTER correction (right panel) ---
    im2_ref = ax_after.pcolormesh(
        lon_ref, lat_ref, ref_map.data,
        cmap="gray", norm=normit(ref_map.data), rasterized=True
    )
    im2_corr = ax_after.pcolormesh(
        lon_corr, lat_corr, corrected_map.data,
        cmap=corrected_map.plot_settings.get("cmap", "sdoaia304"),
        norm=normit(corrected_map.data), rasterized=True
    )
    im2_corr.set_visible(False)
    
    # Add contours for corrected on reference
    data_corr = corrected_map.data[~np.isnan(corrected_map.data)]
    if data_corr.size > 0:
        levels_corr = np.percentile(data_corr, [5, 10, 80, 90, 95, 99])
        cmap_corr = plt.get_cmap('sdoaia304').reversed()
        im2_contour = ax_after.contour(
            lon_corr, lat_corr, corrected_map.data,
            levels=levels_corr, cmap=cmap_corr, linewidths=0.3, alpha=0.5
        )
    
    ax_after.set_xlim(xlim)
    ax_after.set_ylim(ylim)
    ax_after.set_xlabel("Tx (arcsec)")
    ax_after.set_ylabel("Ty (arcsec)")
    ax_after.set_title(f"After Correction{phase_label_str}")
    
    # Create animation frames
    n_frames = 2 * n_cycles
    
    def update(frame):
        if frame % 2 == 0:
            # Show reference in both panels
            im1_ref.set_visible(True)
            im1_tgt.set_visible(False)
            im2_ref.set_visible(True)
            im2_corr.set_visible(False)
        else:
            # Show target (before) and corrected (after)
            im1_ref.set_visible(False)
            im1_tgt.set_visible(True)
            im2_ref.set_visible(False)
            im2_corr.set_visible(True)
        return im1_ref, im1_tgt, im2_ref, im2_corr
    
    from matplotlib.animation import FuncAnimation, PillowWriter
    ani = FuncAnimation(
        fig, update, frames=list(range(n_frames)),
        interval=interval, blit=False, repeat=True
    )
    
    # Create descriptive title with phase information
    main_title = f"Blink Comparison: Before vs After Correction"
    if phase_name:
        main_title += f" - {phase_name.title()}"
    fig.suptitle(main_title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Generate filename based on phase
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    phase_suffix = f"_{phase_name}" if phase_name else ""
    gif_path = self.pdf_path.parent / f"blink_comparison{phase_suffix}_{timestamp}.gif"
    
    # Save animation as GIF
    try:
        writer = PillowWriter(fps=1000//interval)
        ani.save(str(gif_path), writer=writer, dpi=100)
        print(f"✅ Blink animation saved: {gif_path}")
    except Exception as e:
        print(f"⚠️ Could not save GIF animation: {e}")
    
    plt.close(fig)

  def render_alignment_overlay(
      self,
      ref_img: np.ndarray,
      target_img: np.ndarray,
      best_shift: Tuple[int, int],
      *,
      ref_map: GenericMap | None = None,
      target_map: GenericMap | None = None,
      corrected_map: GenericMap | None = None,
  ) -> None:
    """Append diagnostic overlays (pixel + WCS frames) to the PDF trace."""

    def _finite_range(data: np.ndarray) -> Tuple[float, float]:
      finite_vals = data[np.isfinite(data)]
      if finite_vals.size == 0:
        return (0.0, 1.0)
      return (float(finite_vals.min()), float(finite_vals.max()))

    if ref_map is not None and target_map is not None and corrected_map is not None:
      ref_data = np.asarray(ref_map.data, dtype=float)
      tgt_data = np.asarray(target_map.data, dtype=float)
      corr_data = np.asarray(corrected_map.data, dtype=float)

      ref_vmin, ref_vmax = _finite_range(ref_data)
      tgt_vmin, tgt_vmax = _finite_range(tgt_data)
      if np.isclose(tgt_vmin, tgt_vmax):
        tgt_vmax = tgt_vmin + 1.0
      contour_levels = np.linspace(tgt_vmin, tgt_vmax, 6)

      ny, nx = ref_data.shape
      yy, xx = np.mgrid[0:ny, 0:nx]
      shift_dx = int(best_shift[0])
      shift_dy = int(best_shift[1])

      lon_ref, lat_ref = get_coord_mat(ref_map)
      lon_tgt, lat_tgt = get_coord_mat(target_map)
      lon_corr, lat_corr = get_coord_mat(corrected_map)

      overlay_fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False)
      axes = np.asarray(axes)

      def _plot_panel(
        ax: plt.Axes,
        x_base: np.ndarray,
        y_base: np.ndarray,
        base_data: np.ndarray,
        x_contour: np.ndarray,
        y_contour: np.ndarray,
        contour_data: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
      ) -> None:
        ax.pcolormesh(
          x_base,
          y_base,
          base_data,
          cmap="sdoaia171",
          rasterized=True,
          norm=normit(base_data),
        )
        ax.contour(
          x_contour,
          y_contour,
          contour_data,
          levels=contour_levels,
          cmap="magma_r",
          linewidths=0.5,
          norm=normit(contour_data),
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

      _plot_panel(
        axes[0, 0],
        xx,
        yy,
        ref_data,
        xx,
        yy,
        tgt_data,
        "Pixels · before correction",
        "x (px)",
        "y (px)",
      )
      _plot_panel(
        axes[0, 1],
        xx,
        yy,
        ref_data,
        xx + shift_dx,
        yy + shift_dy,
        tgt_data,
        "Pixels · after correction",
        "x (px)",
        "y (px)",
      )
      _plot_panel(
        axes[1, 0],
        lon_ref,
        lat_ref,
        ref_data,
        lon_tgt,
        lat_tgt,
        tgt_data,
        "Helioprojective · before correction",
        "lon (arcsec)",
        "lat (arcsec)",
      )
      _plot_panel(
        axes[1, 1],
        lon_ref,
        lat_ref,
        ref_data,
        lon_corr,
        lat_corr,
        corr_data,
        "Helioprojective · after correction",
        "lon (arcsec)",
        "lat (arcsec)",
      )
      overlay_fig.tight_layout()
      self.pdf_writer.savefig(overlay_fig)
      plt.close(overlay_fig)
      return

    shift_dx = int(best_shift[0])
    shift_dy = int(best_shift[1])
    ny, nx = ref_img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ref_vmin, ref_vmax = _finite_range(ref_img)
    tgt_vmin, tgt_vmax = _finite_range(target_img)

    overlay_fig, overlay_ax = plt.subplots(figsize=(6, 6))
    overlay_ax.pcolormesh(
      xx,
      yy,
      ref_img,
      cmap="sdoaia171",
      rasterized=True,
      norm=normit(ref_img),
    )
    overlay_ax.contour(
      xx + shift_dx,
      yy + shift_dy,
      target_img,
      cmap="magma_r",
      norm=normit(target_img),
    )
    overlay_ax.set_title(f"Reference vs shifted target (dx={shift_dx}, dy={shift_dy})")
    overlay_ax.set_xlabel("x (pixels)")
    overlay_ax.set_ylabel("y (pixels)")
    self.pdf_writer.savefig(overlay_fig)
    plt.close(overlay_fig)

  def render_history_plot(
      self,
      history: np.ndarray,
  ) -> None:
    """
    Render an adaptive diagnostic plot based on whether sx and sy vary.
    
    If sx and sy are constant (no scale optimization), creates a 2-axis figure:
      - dx vs dy scatter colored by corr
      - corr vs iteration
    
    If sx and sy vary (scale optimization), creates a 3-axis figure:
      - dx vs dy scatter colored by corr
      - sx vs sy scatter colored by corr  
      - corr vs iteration
    
    Parameters
    ----------
    history : np.ndarray
        Array of shape (N, 6) with columns [iteration, dx, dy, sx, sy, corr]
    """
    if history.size == 0:
      return
    
    # Extract columns
    iterations = history[:, 0]
    dx = history[:, 1]
    dy = history[:, 2]
    sx = history[:, 3]
    sy = history[:, 4]
    corr = history[:, 5]
    
    # Detect if sx and sy are constant (no scale optimization)
    sx_std = np.std(sx)
    sy_std = np.std(sy)
    scale_varies = (sx_std > 1e-6) or (sy_std > 1e-6)
    
    if scale_varies:
      # Three axes: dx vs dy, sx vs sy, corr vs iteration
      fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
      
      # dx vs dy colored by corr
      sc1 = axes[0].scatter(dx, dy, c=corr, cmap="viridis", s=22, alpha=0.8)
      axes[0].set_xlabel("dx (pixels)")
      axes[0].set_ylabel("dy (pixels)")
      axes[0].set_title("dx vs dy (color=corr)")
      axes[0].grid(True, alpha=0.3)
      plt.colorbar(sc1, ax=axes[0], label="corr")
      
      # sx vs sy colored by corr
      sc2 = axes[1].scatter(sx, sy, c=corr, cmap="viridis", s=22, alpha=0.8)
      axes[1].set_xlabel("sx (scale)")
      axes[1].set_ylabel("sy (scale)")
      axes[1].set_title("sx vs sy (color=corr)")
      axes[1].grid(True, alpha=0.3)
      plt.colorbar(sc2, ax=axes[1], label="corr")
      
      # corr vs iteration
      axes[2].plot(iterations, corr, color="tab:blue", lw=1.5, marker="o", markersize=3)
      axes[2].set_xlabel("iteration")
      axes[2].set_ylabel("corr")
      axes[2].set_title("corr vs iteration")
      axes[2].grid(True, alpha=0.3)
    else:
      # Two axes: dx vs dy, corr vs iteration (no scale optimization)
      fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
      
      # dx vs dy colored by corr
      sc1 = axes[0].scatter(dx, dy, c=corr, cmap="viridis", s=22, alpha=0.8)
      axes[0].set_xlabel("dx (pixels)")
      axes[0].set_ylabel("dy (pixels)")
      axes[0].set_title("dx vs dy (color=corr)")
      axes[0].grid(True, alpha=0.3)
      plt.colorbar(sc1, ax=axes[0], label="corr")
      
      # corr vs iteration
      axes[1].plot(iterations, corr, color="tab:blue", lw=1.5, marker="o", markersize=3)
      axes[1].set_xlabel("iteration")
      axes[1].set_ylabel("corr")
      axes[1].set_title("corr vs iteration")
      axes[1].grid(True, alpha=0.3)
    
    self.pdf_writer.savefig(fig)
    plt.close(fig)

  def reset_scatter_size(self, data_radius: float | None = None) -> None:
    """Resize scatter markers to match the current axis scale for each data point."""
    if data_radius is not None:
      self.point_radius_x_data = data_radius
      self.point_radius_y_data = data_radius
    if not self.debug_points:
      return
    coords = np.array([[pt[0], pt[1]] for pt in self.debug_points], dtype=float)
    corrs = np.array([pt[2] for pt in self.debug_points], dtype=float)
    self._sync_marker_rectangles(coords, corrs)

  def _sync_marker_rectangles(self, coords: np.ndarray, corrs: np.ndarray) -> None:
    self._clear_marker_patches()
    cmap = self.color_mappable.cmap
    norm = self.color_mappable.norm
    patches: List[Rectangle] = []
    for (x, y), corr in zip(coords, corrs, strict=False):
      lower_left = (
        float(x) - self.point_radius_x_data,
        float(y) - self.point_radius_y_data,
      )
      rect = Rectangle(
        lower_left,
        2.0 * self.point_radius_x_data,
        2.0 * self.point_radius_y_data,
        linewidth=0.0,
      )
      rect.set_facecolor(cmap(norm(corr)))
      rect.set_edgecolor("none")
      rect.set_alpha(0.9)
      rect.set_rasterized(True)
      self.ax.add_patch(rect)
      patches.append(rect)
    self.marker_patches = patches
    self.color_mappable.set_array(corrs)

  def _clear_marker_patches(self) -> None:
    if not self.marker_patches:
      return
    for patch in self.marker_patches:
      try:
        patch.remove()
      except ValueError:
        pass
    self.marker_patches.clear()

  def _remove_non_marker_patches(self) -> None:
    for patch in list(self.ax.patches):
      if patch in self.marker_patches:
        continue
      try:
        patch.remove()
      except ValueError:
        pass

  def close(self) -> None:
    """Close both the Matplotlib figure and the PDF writer."""
    if self.owns_writer:
      self.pdf_writer.close()
    plt.close(self.fig)


def create_debug_context(
  shift_x: int,
  shift_y: int,
  debug_dir: Path,
  *,
  pdf_writer: PdfPages | None = None,
  pdf_path: Path | None = None,
  close_writer_on_exit: bool = True,
  point_radius_data: float = 0.5,
  dpi: int | None = None,
) -> DebugPlotContext:
  """Return a configured `DebugPlotContext` that writes into `debug_dir`."""
  if pdf_writer is None:
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    debug_pdf_path = (debug_dir / f"xcorr_debug_{timestamp}.pdf").resolve()
    pdf_writer = PdfPages(debug_pdf_path)
  else:
    if pdf_path is None:
      raise ValueError("pdf_path must be provided when reusing a PdfPages writer.")
    debug_pdf_path = pdf_path
  fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
  ax.set_xlim(-shift_x - 1, shift_x + 1)
  ax.set_ylim(-shift_y - 1, shift_y + 1)
  color_mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=SCATTER_COLORMAP)
  color_mappable.set_array([])
  cbar = fig.colorbar(color_mappable, ax=ax, shrink=0.85)
  if cbar.solids is not None: cbar.solids.set_rasterized(True)
  cbar.set_label("Correlation")
  ax.set_xlabel("dx (pixels)")
  ax.set_ylabel("dy (pixels)")
  ax.set_title("Cross-correlation progression")
  ax.grid(True, alpha=0.3)
  ctx = DebugPlotContext(
    pdf_writer=pdf_writer,
    fig=fig,
    ax=ax,
    color_mappable=color_mappable,
    debug_points=[],
    plotted_points=set(),
    pdf_path=debug_pdf_path,
    owns_writer=close_writer_on_exit,
    point_radius_x_data=point_radius_data,
    point_radius_y_data=point_radius_data,
  )
  ctx.reset_scatter_size()
  return ctx
