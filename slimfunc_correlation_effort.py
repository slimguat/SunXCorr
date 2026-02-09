"""Cross-correlation helpers for SPICE/FSI alignment workflows.

Usage outline:

- Reproject maps onto a common grid with `reproject_map_to_reference`.
- Estimate shifts and scale factors with `optimize_alignment_local` or
    `optimize_alignment_local_grad_disc_persworkers`.
- Inspect results using `plot_alignment_before_after`,
    `correlation_with_iteration`, and `plot_history_scatter`.
- Apply the correction via `make_corrected_wcs_map` and optionally recover the
    original pointing with `find_original_correction`.
- Create quick before/after animations with `blink_maps`.
"""

from __future__ import annotations
from numba import jit
from reproject import reproject_interp
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import sunpy.map
from help_funcs import get_coord_mat
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from typing import Tuple, Optional, List, Dict, Any, Literal, Union, cast
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sunpy.map import Map,GenericMap
from multiprocessing import Process, Queue
from dataclasses import dataclass
from typing import Optional, Tuple
from numba import jit


__all__: list[str] = [
    "normalized_corr_nan_safe",
    "squeeze_to_ref_grid",
    "shift_image",
    "correlation_for_params",
    "optimize_alignment_local",
    "optimize_alignment_local_grad_disc_persworkers",
    "reproject_map_to_reference",
    "plot_history_scatter",
    "plot_alignment_before_after",
    "correlation_with_iteration",
    "blink_maps",
    "make_corrected_wcs_map",
    "find_original_correction",
]


def normalized_corr_nan_safe(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Normalized correlation between two same-shape images,
    ignoring NaNs (they do not contribute to sums).
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape.")

    valid = np.isfinite(img1) & np.isfinite(img2)
    if not np.any(valid):
        return 0.0

    a = img1[valid]
    b = img2[valid]

    a_c = a - a.mean()
    b_c = b - b.mean()

    num = np.dot(a_c, b_c)
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))

    if denom == 0.0:
        return 0.0

    return num / denom


# ==============================================================
# Helper: squeeze/stretch image
# ==============================================================
def squeeze_to_ref_grid(
    target_img: np.ndarray,
    ref_shape: Tuple[int, int],
    squeeze_x: float = 1.0,
    squeeze_y: float = 1.0,
    center: Optional[Tuple[float, float]] = None,
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    """
    Rescale (stretch/squeeze) target_img onto reference grid using affine_transform.
    """
    target_img = np.asarray(target_img, dtype=np.float64)
    ny, nx = target_img.shape

    if center is None:
        cy = (ny - 1) / 2.0
        cx = (nx - 1) / 2.0
    else:
        cy, cx = center

    # affine: out_coord -> in_coord = A*out_coord + offset
    A = np.array([[1.0 / squeeze_y, 0.0],
                  [0.0, 1.0 / squeeze_x]], dtype=np.float64)

    center_vec = np.array([cy, cx], dtype=np.float64)
    offset = center_vec - A @ center_vec

    squeezed = affine_transform(
        target_img,
        A,
        offset=offset,
        output_shape=ref_shape,
        order=order,
        cval=cval
    )
    return squeezed


# ==============================================================
# Helper: shift image (no wrap)
# ==============================================================
def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Shift image by (dx, dy) with zero padding (no wrap).
    Positive dx -> right, positive dy -> down.
    """
    ny, nx = img.shape
    out = np.full_like(img,np.nan)

    if dx >= 0:
        xs, xe = 0, nx - dx
        xd, xe2 = dx, nx
    else:
        xs, xe = -dx, nx
        xd, xe2 = 0, nx + dx

    if dy >= 0:
        ys, ye = 0, ny - dy
        yd, ye2 = dy, ny
    else:
        ys, ye = -dy, ny
        yd, ye2 = 0, ny + dy

    # no overlap -> return all-NaN
    if ys >= ye or xs >= xe or yd >= ye2 or xd >= xe2:
        return np.full_like(img, np.nan)


    out[yd:ye2, xd:xe2] = img[ys:ye, xs:xe]
    return out


def correlation_for_params(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    dx: float,
    dy: float,
    squeeze_x: float,
    squeeze_y: float,
    center: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Objective function: given dx, dy, squeeze_x, squeeze_y,
    1) squeeze target onto ref grid,
    2) shift it,
    3) compute normalized correlation with ref_img.
    """
    # 1) squeeze onto reference grid
    if squeeze_x == 1.0 and squeeze_y == 1.0:
        squeezed = target_img
    else:
        squeezed = squeeze_to_ref_grid(
            target_img,
            ref_shape=ref_img.shape,
            squeeze_x=squeeze_x,
            squeeze_y=squeeze_y,
            order=1,
            cval=np.nan,
            center=center,
        )

    # 2) shift
    shifted = shift_image(squeezed, int(round(dx)), int(round(dy)))

    # 3) correlation at zero relative shift (already shifted)
    corr = normalized_corr_nan_safe(ref_img, shifted)
    
    # 4) penalty for number of valid pixels overlapping
    original_valid = np.isfinite(ref_img) & np.isfinite(target_img)
    warped_valid = np.isfinite(ref_img) & np.isfinite(shifted)
    ratio = np.sum(warped_valid) / np.sum(original_valid) if np.sum(original_valid) > 0 else 0.0
    # print(ratio,corr,dx,dy,squeeze_x,squeeze_y)
    if ratio>1: # should not happen, but just in case
        ratio = 1.0
    if ratio>0.5: # if half of pixels are still valid, no penalty
        ratio = 1.0
    if ratio>0.0:
        ratio = ratio  # double the penalty effect if less than half pixels are lost
    corr *= ratio
    return corr


# ==============================================================
# JIT Context (precompute masks once per ref/target pair)
# ==============================================================

@dataclass
class CorrContextJIT:
    ref: np.ndarray
    tgt: np.ndarray
    valid_ref: np.ndarray
    original_valid_count: int


def build_corr_context_jit(ref_img: np.ndarray, target_img: np.ndarray) -> CorrContextJIT:
    ref = np.asarray(ref_img, dtype=np.float64)
    tgt = np.asarray(target_img, dtype=np.float64)
    valid_ref = np.isfinite(ref)
    original_valid_count = int((valid_ref & np.isfinite(tgt)).sum())
    return CorrContextJIT(ref=ref, tgt=tgt, valid_ref=valid_ref, original_valid_count=original_valid_count)


# ==============================================================
# JIT helpers
# ==============================================================

@jit(cache=True, fastmath=True)
def _bilinear_sample_ignore_nan(img, y, x):
    ny, nx = img.shape
    if y < 0.0 or x < 0.0 or y > (ny - 1) or x > (nx - 1):
        return np.nan

    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1

    if y1 >= ny:
        y1 = ny - 1
    if x1 >= nx:
        x1 = nx - 1

    wy = y - y0
    wx = x - x0

    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx

    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]

    acc = 0.0
    wsum = 0.0

    if np.isfinite(v00):
        acc += w00 * v00
        wsum += w00
    if np.isfinite(v01):
        acc += w01 * v01
        wsum += w01
    if np.isfinite(v10):
        acc += w10 * v10
        wsum += w10
    if np.isfinite(v11):
        acc += w11 * v11
        wsum += w11

    if wsum == 0.0:
        return np.nan
    return acc / wsum


@jit(cache=True, fastmath=False)
def _corr_fused_scale_shift(ref, tgt, valid_ref,
                           dx_i, dy_i,
                           inv_sx, inv_sy,
                           off_x, off_y,
                           original_valid_count):
    sum_a = 0.0
    sum_b = 0.0
    sum_a2 = 0.0
    sum_b2 = 0.0
    sum_ab = 0.0
    n = 0

    ny, nx = ref.shape

    for y in range(ny):
        for x in range(nx):
            if not valid_ref[y, x]:
                continue

            # undo integer shift on the scaled grid
            y0 = y - dy_i
            x0 = x - dx_i

            # map scaled-grid coord -> original tgt coord
            yin = inv_sy * y0 + off_y
            xin = inv_sx * x0 + off_x

            b = _bilinear_sample_ignore_nan(tgt, yin, xin)
            if not np.isfinite(b):
                continue

            a = ref[y, x]  # finite by valid_ref
            sum_a += a
            sum_b += b
            sum_a2 += a * a
            sum_b2 += b * b
            sum_ab += a * b
            n += 1

    if n == 0:
        return 0.0

    inv_n = 1.0 / n
    cov = sum_ab - (sum_a * sum_b) * inv_n
    var_a = sum_a2 - (sum_a * sum_a) * inv_n
    var_b = sum_b2 - (sum_b * sum_b) * inv_n

    if var_a <= 0.0 or var_b <= 0.0:
        corr = 0.0
    else:
        denom = np.sqrt(var_a * var_b)
        if denom <= 0.0 or (not np.isfinite(denom)):
            corr = 0.0
        else:
            corr = cov / denom

    # penalty for overlap (same spirit as your original)
    if original_valid_count > 0:
        ratio = n / original_valid_count
        if ratio > 1.0:
            ratio = 1.0
        if ratio > 0.5:
            ratio = 1.0
        corr *= ratio

    return corr


# ==============================================================
# Public JIT objective (same signature as correlation_for_params)
# ==============================================================

def correlation_for_params_jit(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    dx: float,
    dy: float,
    squeeze_x: float,
    squeeze_y: float,
    center: Optional[Tuple[float, float]] = None,
    ctx: Optional[CorrContextJIT] = None,
) -> float:
    """
    Drop-in replacement for correlation_for_params.

    - Same params.
    - You can optionally pass a prebuilt ctx to avoid rebuilding masks each call.
    """

    # Build / reuse context
    if ctx is None:
        ctx = build_corr_context_jit(ref_img, target_img)

    # param sanity
    if (not np.isfinite(squeeze_x)) or (not np.isfinite(squeeze_y)):
        return 0.0
    if squeeze_x <= 0.0 or squeeze_y <= 0.0:
        return 0.0

    dx_i = int(np.round(dx))
    dy_i = int(np.round(dy))

    # center convention matches your squeeze_to_ref_grid default
    if center is None:
        ny, nx = ctx.tgt.shape
        cy = (ny - 1) / 2.0
        cx = (nx - 1) / 2.0
    else:
        cy, cx = float(center[0]), float(center[1])

    inv_sy = 1.0 / float(squeeze_y)
    inv_sx = 1.0 / float(squeeze_x)

    # affine_transform-style offset for diagonal scaling about center
    off_y = cy - inv_sy * cy
    off_x = cx - inv_sx * cx

    return _corr_fused_scale_shift(
        ctx.ref, ctx.tgt, ctx.valid_ref,
        dx_i, dy_i,
        inv_sx, inv_sy,
        off_x, off_y,
        ctx.original_valid_count
    )

















def _corr_for_candidate(
    args: Tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
        Optional[Tuple[float, float]],
    ],
) -> float:
    """
    Helper for multiprocessing: evaluate correlation_for_params for one candidate.
    args = (ref_img, target_img, dx, dy, sx, sy)
    """
    ref_img, target_img, dx, dy, sx, sy,center = args
    return correlation_for_params(ref_img, target_img, dx, dy, sx, sy,center)


def optimize_alignment_local(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    dx0: float = 0.0,
    dy0: float = 0.0,
    squeeze_x0: float = 1.0,
    squeeze_y0: float = 1.0,
    vary_shift: bool = True,
    vary_scale_x: bool = False,
    vary_scale_y: bool = False,
    step_dx0: float = 3.0,
    step_dy0: float = 3.0,
    step_sx0: float = 0.02,
    step_sy0: float = 0.02,
    n_neighbors: int = 20,
    max_iter: int = 30,
    step_shrink: float = 0.5,
    min_step_shift: float = 0.25,
    min_step_scale: float = 1e-3,
    center: Optional[Tuple[float, float]] = None,
    squeeze_x_bounds: Tuple[float, float] = (0.9, 1.1),
    squeeze_y_bounds: Tuple[float, float] = (0.9, 1.1),
    shift_range: Optional[Tuple[int, int]] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
        Local, batched search for best (dx, dy, squeeze_x, squeeze_y),
        using N-neighborhood hill-climbing.

        Parameters
        ----------
        ref_img, target_img : 2D arrays
            Reference and uncorrected images.
        dx0, dy0 : float
            Initial shift guess (pixels).
        squeeze_x0, squeeze_y0 : float
            Initial scale guess.
        vary_shift, vary_scale_x, vary_scale_y : bool
            Whether to optimize each parameter.
        step_dx0, step_dy0, step_sx0, step_sy0 : float
            Initial step sizes.
        n_neighbors : int
            Number of neighbors sampled per iteration (plus the current point).
        max_iter : int
            Maximum iterations.
        step_shrink : float
            Factor to shrink step sizes when no improvement (0 < step_shrink < 1).
        min_step_shift : float
            Stop when max(|step_dx|, |step_dy|) < this (if varying shift).
        min_step_scale : float
            Stop when max(|step_sx|, |step_sy|) < this (if varying scale).
        center : (cy, cx) or None
            Center point for scaling transformations. If None, use image center.
        squeeze_x_bounds, squeeze_y_bounds : (float, float)
            Hard bounds for scale parameters.
        shift_range : (int, int) or None
            Bounds for dx and dy (min_shift, max_shift). If None, no bounds.
        n_jobs : int
            Number of CPU processes to use for neighbor evaluation.
            1 = no parallelism (simple loop). >1 uses multiprocessing.Pool.
        verbose : bool
            Print progress.

        Returns
        -------
        best_params : dict
            {
            "dx": ...,
            "dy": ...,
            "squeeze_x": ...,
            "squeeze_y": ...,
            "corr": ...,
            }
        history : np.ndarray
            Array of shape (N_samples, 6) with columns:
            [iteration, dx, dy, squeeze_x, squeeze_y, corr]
    """

    ref_img = np.asarray(ref_img, dtype=np.float64)
    target_img = np.asarray(target_img, dtype=np.float64)

    # current param vector [dx, dy, sx, sy]
    p = np.array([dx0, dy0, squeeze_x0, squeeze_y0], dtype=float)
    prev_p = p.copy()

    # step sizes (0 if fixed)
    step = np.array([
        step_dx0 if vary_shift else 0.0,
        step_dy0 if vary_shift else 0.0,
        step_sx0 if vary_scale_x else 0.0,
        step_sy0 if vary_scale_y else 0.0,
    ], dtype=float)

    # helper to clamp shift if shift_range is given
    def clamp_shift(dx, dy):
        if shift_range is None:
            return dx, dy
        smin, smax = shift_range
        return (
            float(np.clip(dx, smin, smax)),
            float(np.clip(dy, smin, smax)),
        )

    # ---- history storage: [iter, dx, dy, sx, sy, corr] ----
    history_rows = []

    # initial correlation
    p[0], p[1] = clamp_shift(p[0], p[1])
    cur_corr = correlation_for_params(
        ref_img, target_img, p[0], p[1], p[2], p[3],center=center
    )
    # iteration index 0 = initial point
    history_rows.append([0, p[0], p[1], p[2], p[3], cur_corr])

    if verbose:
        print(f"Initial: dx={p[0]:.3f}, dy={p[1]:.3f}, "
              f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")

    for it in range(max_iter):
        # stopping criteria based on step sizes
        step_shift_max = max(abs(step[0]), abs(step[1]))
        step_scale_max = max(abs(step[2]), abs(step[3]))

        if (not vary_shift or step_shift_max < min_step_shift) and \
           (not (vary_scale_x or vary_scale_y) or step_scale_max < min_step_scale):
            if verbose:
                print("Stopping: step sizes below thresholds.")
            break

        direction = p - prev_p
        neighbors = [p.copy()]  # include current point

        # ---- generate neighbors around current point ----
        for _ in range(n_neighbors):
            # random perturbation in [-1, 1] * step
            delta = (2.0 * np.random.rand(4) - 1.0) * step

            # small bias along previous move (if any)
            if np.linalg.norm(direction) > 0:
                direction_unit = direction / np.linalg.norm(direction)
                delta += 0.3 * step * direction_unit

            cand = p + delta

            # clamp scale
            cand[2] = np.clip(cand[2], squeeze_x_bounds[0], squeeze_x_bounds[1])
            cand[3] = np.clip(cand[3], squeeze_y_bounds[0], squeeze_y_bounds[1])

            # clamp shifts (if requested)
            cand[0], cand[1] = clamp_shift(cand[0], cand[1])

            neighbors.append(cand)

        # ---- evaluate all neighbors (optionally in parallel) ----
        best_local_corr = cur_corr
        best_local_p = p.copy()

        if n_jobs == 1:
            # serial evaluation
            for cand in neighbors:
                dx_c, dy_c, sx_c, sy_c = cand
                corr_c = correlation_for_params(ref_img, target_img, dx_c, dy_c, sx_c, sy_c,center=center)
                # log into history
                history_rows.append([it + 1, dx_c, dy_c, sx_c, sy_c, corr_c])

                if corr_c > best_local_corr:
                    best_local_corr = corr_c
                    best_local_p = cand.copy()
        else:
            # parallel evaluation using multiprocessing
            # NOTE: works best in a script guarded with if __name__ == "__main__":
            tasks = [
                (ref_img, target_img, cand[0], cand[1], cand[2], cand[3], center)
                for cand in neighbors
            ]
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_corr_for_candidate, tasks)

            for cand, corr_c in zip(neighbors, results):
                dx_c, dy_c, sx_c, sy_c = cand
                history_rows.append([it + 1, dx_c, dy_c, sx_c, sy_c, corr_c])

                if corr_c > best_local_corr:
                    best_local_corr = corr_c
                    best_local_p = cand.copy()

        # ---- update / shrink steps ----
        if best_local_corr > cur_corr:
            # improvement: move to best neighbor
            prev_p = p.copy()
            p = best_local_p
            cur_corr = best_local_corr
            if verbose:
                print(f"[iter {it+1}] improved: dx={p[0]:.3f}, dy={p[1]:.3f}, "
                      f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")
        else:
            # no improvement → shrink step sizes
            step *= step_shrink
            if verbose:
                print(f"[iter {it+1}] no improvement, shrinking steps to {step}")

    best_params = {
        "dx": p[0],
        "dy": p[1],
        "squeeze_x": p[2],
        "squeeze_y": p[3],
        "corr": cur_corr,
    }

    if verbose:
        print("\nFinal best:",
              f"dx={p[0]:.3f}, dy={p[1]:.3f}, sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")

    history = np.array(history_rows, dtype=float)
    return best_params, history


# ==============================================================
# Helper: reproject map onto reference map grid
# ==============================================================
def reproject_map_to_reference(
    ref_map: GenericMap,
    input_map: GenericMap,
    order: Any = "bilinear",
) -> GenericMap:
    """
    Reproject input_map onto the WCS & pixel grid of ref_map.

    Parameters
    ----------
    ref_map : sunpy.map.Map
        Reference map (defines output WCS and shape).
    input_map : sunpy.map.Map
        Map to be reprojected.
    order : str or int
        Interpolation order. For some reproject versions we may need
        order=1 instead of "bilinear".

    Returns
    -------
    new_map : sunpy.map.Map
        input_map data reprojected onto ref_map's grid.
    """
    target_wcs = ref_map.wcs
    target_shape = ref_map.data.shape

    # NOTE: keyword is shape_out, not target_shape
    reprojected_data, footprint = reproject_interp(
        input_map,           # or (input_map.data, input_map.wcs)
        target_wcs,
        shape_out=target_shape,
        order=order,
    )

    # Mask outside-footprint pixels as NaN
    reprojected_data = np.where(footprint > 0, reprojected_data, np.nan)

    new_meta = ref_map.meta.copy()
    new_map = sunpy.map.Map(reprojected_data, new_meta, plot_settings=input_map.plot_settings)

    return cast(GenericMap, new_map)

# ==============================================================
# Visualization: scatter plot of optimization history
# ==============================================================
def plot_history_scatter(
    history: np.ndarray,
    x_param: str,
    y_param: str,
    plot_kwargs: Dict[str, Any] = {"cmap": "magma_r", "s": 3},
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Scatter plot of optimization history, with user-chosen axes.

    Parameters
    ----------
    history : ndarray, shape (N,6)
        Returned by optimize_alignment_local.
        Columns: [iter, dx, dy, sx, sy, corr]

    x_param, y_param : str
        Any of: "dx", "dy", "sx", "sy", "iter"
        These choose what goes on the x/y axes.

    cmap : str
        Matplotlib colormap name.

    s : float
        Marker size.

    Returns
    -------
    fig, ax : Figure and Axes of the scatter plot.
    """

    # Map names to history columns
    param_to_col = {
        "iter": 0,
        "dx": 1,
        "dy": 2,
        "sx": 3,
        "sy": 4,
        "sx": 3,
        "sy": 4,
        "corr": 5
    }

    if x_param not in param_to_col:
        raise ValueError(f"Invalid x_param '{x_param}'. Use dx, dy, sx, sy, iter.")
    if y_param not in param_to_col:
        raise ValueError(f"Invalid y_param '{y_param}'. Use dx, dy, sx, sy, iter.")

    x = history[:, param_to_col[x_param]]
    y = history[:, param_to_col[y_param]]
    corr = history[:, 5]   # correlation always column 5

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    else:
        fig = ax.figure
    sc = ax.scatter(x, y, c=corr,**plot_kwargs,norm = plt.Normalize(vmin=np.min(corr), vmax=np.max(corr)))

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{x_param} vs {y_param} (colored by correlation)")
    ax.grid(True)
    cbar = fig.colorbar(sc, ax=ax, label="Correlation")
    return fig, ax

# =============================================================
# Visualization: alignment before/after applying best_params
# ==============================================================
def plot_alignment_before_after(
    ref_map: GenericMap,
    target_map: GenericMap,
    best_params: Dict[str, float],
    levels: Optional[np.ndarray] = None,
    center: Optional[Tuple[float, float]] = None,
    axes: Optional[Tuple[Axes, Axes]] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Visual diagnostic of alignment before/after applying best_params.

    Left panel : ref_img with contours of *uncorrected* target_img
    Right panel: ref_img with contours of *corrected* target_img
                 (squeezed + shifted using best_params)

    Parameters
    ----------
    ref_img : 2D array or sunpy.map.Map
        Reference image (e.g. FSI/EUI) on the grid we optimised in.
    target_img : 2D array or sunpy.map.Map
        Target image before correction (e.g. SPICE projected to ref grid or
        original grid used in the optimizer).
    best_params : dict
        Output from optimize_alignment_local, with keys:
        "dx", "dy", "squeeze_x", "squeeze_y".
    levels : array-like or None
        Contour levels. If None, they are computed automatically from target_img.
    cmap_ref : str
        Colormap for the reference image.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    fig, axes : Figure, (ax_before, ax_after)
    """
    ref_img = ref_map
    target_img = target_map
    
    # Allow passing SunPy Maps directly
    if hasattr(ref_img, "data"):
        ref_data = np.asarray(ref_img.data, dtype=float)
    else:
        ref_data = np.asarray(ref_img, dtype=float)

    if hasattr(target_img, "data"):
        tgt_data = np.asarray(target_img.data, dtype=float)
    else:
        tgt_data = np.asarray(target_img, dtype=float)

    # Pull parameters
    dx = float(best_params.get("dx", 0.0))
    dy = float(best_params.get("dy", 0.0))
    sx = float(best_params.get("squeeze_x", 1.0))
    sy = float(best_params.get("squeeze_y", 1.0))

    # Build corrected target: squeeze to ref grid, then shift
    squeezed = squeeze_to_ref_grid(
        tgt_data,
        ref_shape=ref_data.shape,
        squeeze_x=sx,
        squeeze_y=sy,
        order=1,
        center=center,
        cval=np.nan,
    )
    corrected = shift_image(squeezed, int(round(dx)), int(round(dy)))

    # Contour levels (same for before/after)
    if levels is None:
        # use a robust range from percentiles of the ORIGINAL target
        finite_vals = tgt_data[np.isfinite(tgt_data)]
        if finite_vals.size == 0:
            raise ValueError("No finite values in target_img to derive contour levels.")
        vmin = np.percentile(finite_vals, 0)
        vmax = np.percentile(finite_vals, 95)
        levels = np.linspace(vmin, vmax, 5)

    if axes is None:
        figsize = (12, 5)
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    else:
        fig = axes[0].figure
    ax1, ax2 = axes

    # ---------- BEFORE ----------
    im1 = ax1.imshow(ref_data, origin="lower", cmap="gray",norm=ref_map.plot_settings['norm'])
    ax1.contour(tgt_data, levels=levels, cmap="magma_r", linewidths=0.25,alpha=0.5)
    ax1.set_title("Before correction\n(ref + contours of uncorrected target)")
    # plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ---------- AFTER ----------
    im2 = ax2.imshow(ref_data, origin="lower", cmap="gray",norm=ref_map.plot_settings['norm'])
    ax2.contour(corrected, levels=levels, cmap="magma_r", linewidths=0.25,alpha=0.5)
    ax2.set_title("After correction\n(ref + contours of corrected target)")
    # plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")

    # plt.tight_layout()
    return fig, axes

# =============================================================
# Visualization: correlation vs iteration
# =============================================================
def correlation_with_iteration(history: np.ndarray, ax: Optional[Axes] = None) -> None:
    """
    Plot correlation vs iteration from optimization history.
    Parameters
    ----------
    history : np.ndarray
        Array of shape (N_samples, 6) with columns:
        [iteration, dx, dy, squeeze_x, squeeze_y, corr]
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig,ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure
        
    ax.plot(history[:, 0], history[:, 5], ".")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation vs Iteration")


# =============================================================
# Helper: get lon/lat blinking animation
# =============================================================
def blink_maps(
    fsi_map: GenericMap,
    spice_map: GenericMap,
    interval: int = 800,
    n_cycles: int = 10,
    fsi_label: str = "FSI",
    spice_label: str = "SPICE (corrected)",
    use_widgets: bool = False,
    save_path: Optional[str] = None,
    save_kwargs: Optional[Dict[str, Any]] = None,
    xylims: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[Figure, FuncAnimation, Optional[Any]]:
    """
    Create a blinking animation between two maps on the same axes.

    Parameters
    ----------
    fsi_map : sunpy.map.Map
        First map to show (e.g. FSI_map). Its lon/lat limits are used and kept fixed.
    spice_map : sunpy.map.Map
        Second map to show (e.g. corrected SPICE map).
    interval : int, optional
        Time between frames in milliseconds. Default is 800 ms.
    n_cycles : int, optional
        Number of FSI/SPICE cycles. Each cycle has 2 frames (FSI, SPICE).
        Default is 10 (so 20 frames total).
    fsi_label : str, optional
        Title when FSI map is shown.
    spice_label : str, optional
        Title when SPICE map is shown.
    use_widgets : bool, optional
        If True (and ipywidgets is available), returns Jupyter play/slider
        controls to start/stop/scrub the animation frames manually.
    save_path : str or None, optional
        If given (e.g. 'blink.mp4' or 'blink.gif'), the animation is saved
        to this path using FuncAnimation.save().
    save_kwargs : dict or None, optional
        Extra keyword arguments passed to ani.save(), e.g.
        {'fps': 2, 'dpi': 150, 'writer': 'ffmpeg'}.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ani : matplotlib.animation.FuncAnimation
    controls : ipywidgets.HBox or None
        Jupyter controls (Play + Slider) if use_widgets=True and ipywidgets is available,
        otherwise None.
    """

    fig = plt.figure()
    ax = plt.subplot(111)

    # --- FSI map ---
    from matplotlib.animation import FuncAnimation

    lonFSI, latFSI = get_coord_mat(fsi_map)
    im1 = ax.pcolormesh(
        lonFSI,
        latFSI,
        fsi_map.data,
        norm=fsi_map.plot_settings["norm"],
        # cmap=fsi_map.plot_settings["cmap"],
        cmap="gray",
    )
    ax.set_title(fsi_label)

    # --- SPICE map ---
    lonSPICE, latSPICE = get_coord_mat(spice_map)
    im2 = ax.pcolormesh(
        lonSPICE,
        latSPICE,
        spice_map.data,
        norm=spice_map.plot_settings["norm"],
        cmap=spice_map.plot_settings["cmap"],
    )
    im2.set_visible(False)  # start with SPICE hidden
    # add contour of SPICE in  the FSI iamge
    
    data = spice_map.data[~np.isnan(spice_map.data)]
    p99 = np.percentile(data, 99)
    p95 = np.percentile(data, 95)
    p90 = np.percentile(data, 90)
    p80 = np.percentile(data, 80)
    p10 = np.percentile(data, 10)
    p5  = np.percentile(data, 5) 
    levels = [p5, p10, p80, p90, p95, p99]
    cmap = plt.get_cmap('sdoaia304')
    cmap = cmap.reversed()
    im3 = ax.contour(
        lonSPICE,
        latSPICE,
        spice_map.data,
        levels=levels,
        cmap=cmap,
        linewidths=0.3,
        alpha=0.5,
    )
    # --- Lock x/y limits from FSI view ---
    extx = 0.2
    exty = 0.2
    if xylims is not None:
        xlim = xylims[0]
        ylim = xylims[1]
    else:
        xlim = (
            lonSPICE.min() - (lonSPICE.max() - lonSPICE.min()) * extx,
            lonSPICE.max() + (lonSPICE.max() - lonSPICE.min()) * extx,
        )
        ylim = (
            latSPICE.min() - (latSPICE.max() - latSPICE.min()) * exty,
            latSPICE.max() + (latSPICE.max() - latSPICE.min()) * exty,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # total frames = 2 * n_cycles  (FSI, SPICE, FSI, SPICE, …)
    n_frames = 2 * n_cycles
    frames = list(range(n_frames))

    def update(frame):
        if frame % 2 == 0:
            # show FSI
            im1.set_visible(True)
            im2.set_visible(False)
            im3.set_visible(True)
            ax.set_title(fsi_label)
        else:
            # show SPICE
            im1.set_visible(False)
            im2.set_visible(True)
            im3.set_visible(True)
            ax.set_title(spice_label)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return im1, im2, ax

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=False,
        repeat=False,   # no infinite loop
    )

    # --- Optional: save to video / GIF ---
    if save_path is not None:
        if save_kwargs is None:
            save_kwargs = {}
        ani.save(save_path, **save_kwargs)

    controls = None

    # --- Optional: Jupyter start/stop controls with ipywidgets ---
    if use_widgets:
        try:
            import ipywidgets as widgets
            from IPython.display import display

            play = widgets.Play(
                value=0,
                min=0,
                max=n_frames - 1,
                step=1,
                interval=interval,
                description="Press play",
                disabled=False,
            )
            slider = widgets.IntSlider(
                value=0,
                min=0,
                max=n_frames - 1,
                step=1,
                description="Frame",
                continuous_update=False,
            )

            # link play and slider
            widgets.jslink((play, "value"), (slider, "value"))

            # when the slider value changes, update the frame
            def on_value_change(change):
                frame = change["new"]
                update(frame)
                fig.canvas.draw_idle()

            slider.observe(on_value_change, names="value")

            controls = widgets.HBox([play, slider])
            display(controls)

        except ImportError:
            print("ipywidgets not available; widgets controls disabled.")
            controls = None

    return fig, ani, controls

WCSLinearMode = Literal["pc2_unit", "cd_basis_unit", "cdelt_invariant", "cd"]


def build_corrected_wcs_meta_scale_shift(
    map_in: GenericMap,
    dx_opt: float = 0.0,
    dy_opt: float = 0.0,
    sx_opt: float = 1.0,
    sy_opt: float = 1.0,
    verbose: int = 0,
    *,
    linear_mode: WCSLinearMode = "cd_basis_unit",
) -> Dict[str, Any]:
    """
        Construct corrected WCS metadata encoding a pixel-domain affine registration
        (anisotropic scaling about CRPIX followed by a translation), without resampling.

        Pixel-domain transform (registration model)
        ------------------------------------------
            P' = P0 + ΔS (P - P0) + Δ

        with:
        - P  = (pix1, pix2)^T
        - P0 = (CRPIX1, CRPIX2)^T
        - ΔS = diag(s_x, s_y)  (forward scale factors in WCS pixel axis order)
        - Δ  = (Δx, Δy)^T      (forward translation in pixels; +x right, +y down)

        Linear WCS model (pre-projection)
        ---------------------------------
            X = X0 + M (P - P0)

        with:
        - X  = (world1, world2)^T  (intermediate world coords before projection)
        - X0 = (CRVAL1, CRVAL2)^T
        - M  = CD (the 2x2 linear Jacobian at CRPIX)
        - P  = (pix1, pix2)^T
        - P0 = (CRPIX1, CRPIX2)^T

        Enforcing invariance of world coordinates
        -----------------------------------------
            WCS_old(P) = WCS_new(P')   for all P

        Choosing P0' = P0 (CRPIX unchanged) yields:
            M'  = M (ΔS)^{-1}
            X0' = X0 - M' Δ

        Optimizer convention
        --------------------
        The optimizer is assumed to return the inverse correction:
        - s_x = 1 / sx_opt,   s_y = 1 / sy_opt
        - Δx  = -dx_opt,      Δy  = -dy_opt

        FITS linear WCS encodings
        -------------------------
        The linear mapping may be represented either as:
        - CD matrix directly:
                CD_ij  (preferred for unambiguous storage)
            or
        - PC + CDELT factorization (FITS row scaling):
                CD = diag(CDELT1, CDELT2) @ PC

            i.e. CDELT scales ROWS, not columns:
                CD1_j = CDELT1 * PC1_j
                CD2_j = CDELT2 * PC2_j

        linear_mode (normalization / storage convention)
        ------------------------------------------------
        After computing the corrected CD matrix M' (=CD'), this function can store it as:

        - "cd" :
            Store CD' directly in CD1_1..CD2_2 and remove PC/CDELT.

        - "cdelt_invariant" :
            Keep existing CDELT1/2 unchanged and solve PC row-wise:
                PC = diag(1/CDELT) @ CD'
            This preserves the mapping exactly but does not enforce any PC normalization.

        - "pc2_unit" (default) :
            Enforce global (Frobenius) normalization of PC:
                ||PC||_F^2 = sum_{i,j} PC_{i,j}^2 = 1
            while preserving CD' exactly by a single global rescaling:
                PC'    = PC / k
                CDELT' = CDELT * k        (same k applied to both axes)
            so that diag(CDELT') @ PC' == CD'.

        - "cd_basis_unit" :
            Enforce row-wise unit-norm of PC (basis vectors in FITS row-scaled sense):
                ||PC[0,:]||_2 = 1   and   ||PC[1,:]||_2 = 1
            while preserving CD' exactly by per-row rescaling:
                PC'[0,:] = PC[0,:] / r1      CDELT1' = CDELT1 * r1
                PC'[1,:] = PC[1,:] / r2      CDELT2' = CDELT2 * r2
            so that diag(CDELT') @ PC' = CD'.

        Returns
        -------
        dict
            Header-like metadata dict with:
            - CRPIX unchanged
            - updated CRVAL1/2
            - corrected linear mapping stored per `linear_mode`
    """
    meta = copy.deepcopy(map_in.wcs.to_header())

    crval1 = float(meta["CRVAL1"])
    crval2 = float(meta["CRVAL2"])

    # ----- read current linear matrix as CD -----
    def _get_cd_from_header(h: Dict[str, Any]) -> np.ndarray:
        if all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
            return np.array(
                [[float(h["CD1_1"]), float(h["CD1_2"])],
                 [float(h["CD2_1"]), float(h["CD2_2"])]],
                dtype=float,
            )

        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        pc11 = float(h.get("PC1_1", 1.0))
        pc12 = float(h.get("PC1_2", 0.0))
        pc21 = float(h.get("PC2_1", 0.0))
        pc22 = float(h.get("PC2_2", 1.0))

        return np.array(
            [[cdelt1 * pc11, cdelt1 * pc12],
             [cdelt2 * pc21, cdelt2 * pc22]],
            dtype=float,
        )

    CD = _get_cd_from_header(meta)

    # ----- optimizer -> forward affine params -----
    if sx_opt == 0.0 or sy_opt == 0.0:
        raise ValueError("sx_opt and sy_opt must be non-zero.")

    s_x = 1.0 / sx_opt
    s_y = 1.0 / sy_opt
    dx = -dx_opt
    dy = -dy_opt
    
    # ----- corrected linear matrix: CD' = CD @ inv(diag(sx,sy)) (rescales columns) -----
    inv_DS = np.array([[1.0 / s_x, 0.0],
                       [0.0, 1.0 / s_y]], dtype=float)
    CDp = CD @ inv_DS

    # ----- update CRVAL with CRPIX fixed: CRVAL' = CRVAL - CD' @ [dx,dy] -----
    meta["CRVAL1"] = crval1 - (CDp[0, 0] * dx + CDp[0, 1] * dy)
    meta["CRVAL2"] = crval2 - (CDp[1, 0] * dx + CDp[1, 1] * dy)

    # ----- writers -----
    def _write_cd(h: Dict[str, Any], CDm: np.ndarray) -> None:
        h["CD1_1"], h["CD1_2"] = float(CDm[0, 0]), float(CDm[0, 1])
        h["CD2_1"], h["CD2_2"] = float(CDm[1, 0]), float(CDm[1, 1])
        for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
            h.pop(k, None)

    def _write_pc_from_cd_and_cdelt(h: Dict[str, Any], CDm: np.ndarray, cdelt1: float, cdelt2: float) -> None:
        if cdelt1 == 0.0 or cdelt2 == 0.0:
            raise ValueError("Degenerate CDELT: cannot form PC.")
        h["CDELT1"] = float(cdelt1)
        h["CDELT2"] = float(cdelt2)

        h["PC1_1"] = float(CDm[0, 0] / cdelt1)
        h["PC1_2"] = float(CDm[0, 1] / cdelt1)
        h["PC2_1"] = float(CDm[1, 0] / cdelt2)
        h["PC2_2"] = float(CDm[1, 1] / cdelt2)

        for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
            h.pop(k, None)

    def _ensure_cdelt_exists(h: Dict[str, Any]) -> None:
        if "CDELT1" not in h or "CDELT2" not in h:
            raise ValueError("CDELT1/2 not present in header; cannot use PC+CDELT modes.")

    def _write_pc_cdelt_invariant(h: Dict[str, Any], CDm: np.ndarray) -> None:
        _ensure_cdelt_exists(h)
        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        _write_pc_from_cd_and_cdelt(h, CDm, cdelt1, cdelt2)

    def _write_pc2_unit_frobenius(h: Dict[str, Any], CDm: np.ndarray) -> None:
        """
        pc2_unit: enforce sum(PC^2)=1 (Frobenius norm 1), preserving CD.
        Achieved by global scaling of PC by k and compensating both CDELT by k.
        """
        _ensure_cdelt_exists(h)
        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        if cdelt1 == 0.0 or cdelt2 == 0.0:
            raise ValueError("Degenerate CDELT: cannot form PC.")

        PC = np.array([[CDm[0, 0] / cdelt1, CDm[0, 1] / cdelt1],
                       [CDm[1, 0] / cdelt2, CDm[1, 1] / cdelt2]], dtype=float)

        frob = float(np.sqrt(np.sum(PC * PC)))
        if frob == 0.0:
            raise ValueError("Degenerate PC: Frobenius norm zero.")

        # Scale PC down, scale CDELT up -> CD unchanged
        PCp = PC / frob
        cdelt1p = cdelt1 * frob
        cdelt2p = cdelt2 * frob

        if verbose:
            print(f"pc2_unit: Frobenius(PC)={frob:.6g} -> enforcing ||PC||_F=1")

        _write_pc_from_cd_and_cdelt(h, CDm, cdelt1p, cdelt2p)
        # overwrite PC with the normalized one explicitly (avoids tiny float drift)
        h["PC1_1"], h["PC1_2"] = float(PCp[0, 0]), float(PCp[0, 1])
        h["PC2_1"], h["PC2_2"] = float(PCp[1, 0]), float(PCp[1, 1])

    def _write_cd_basis_unit_rows(h: Dict[str, Any], CDm: np.ndarray) -> None:
        """
        cd_basis_unit: enforce ||PC row1||=1 and ||PC row2||=1, preserving CD.
        Achieved by scaling each PC row i by ki and compensating CDELTi by ki.
        """
        _ensure_cdelt_exists(h)
        cdelt1 = float(h["CDELT1"])
        cdelt2 = float(h["CDELT2"])
        if cdelt1 == 0.0 or cdelt2 == 0.0:
            raise ValueError("Degenerate CDELT: cannot form PC.")

        PC = np.array([[CDm[0, 0] / cdelt1, CDm[0, 1] / cdelt1],
                       [CDm[1, 0] / cdelt2, CDm[1, 1] / cdelt2]], dtype=float)

        r1 = float(np.linalg.norm(PC[0, :]))
        r2 = float(np.linalg.norm(PC[1, :]))
        if r1 == 0.0 or r2 == 0.0:
            raise ValueError("Degenerate PC: row norm zero.")

        # Normalize rows, scale CDELT per row -> CD unchanged
        PCp = PC.copy()
        PCp[0, :] /= r1
        PCp[1, :] /= r2
        cdelt1p = cdelt1 * r1
        cdelt2p = cdelt2 * r2

        if verbose:
            print(f"cd_basis_unit: row norms (r1,r2)=({r1:.6g},{r2:.6g}) -> enforcing both = 1")

        _write_pc_from_cd_and_cdelt(h, CDm, cdelt1p, cdelt2p)
        h["PC1_1"], h["PC1_2"] = float(PCp[0, 0]), float(PCp[0, 1])
        h["PC2_1"], h["PC2_2"] = float(PCp[1, 0]), float(PCp[1, 1])

    # ----- dispatch -----
    if linear_mode == "cd":
        _write_cd(meta, CDp)

    elif linear_mode == "cdelt_invariant":
        _write_pc_cdelt_invariant(meta, CDp)

    elif linear_mode == "pc2_unit":
        _write_pc2_unit_frobenius(meta, CDp)

    elif linear_mode == "cd_basis_unit":
        _write_cd_basis_unit_rows(meta, CDp)

    else:
        raise ValueError(
            f"Unknown linear_mode='{linear_mode}'. "
            "Choose from 'pc2_unit', 'cd_basis_unit', 'cdelt_invariant', 'cd'."
        )

    return meta


def make_corrected_wcs_map(
    map_in: GenericMap,
    best_params: Dict[str, float],
    verbose: int = 0,
    linear_mode: WCSLinearMode = "cd_basis_unit",
) -> GenericMap:
    dx = float(best_params.get("dx", 0.0))
    dy = float(best_params.get("dy", 0.0))
    sx = float(best_params.get("squeeze_x", 1.0))
    sy = float(best_params.get("squeeze_y", 1.0))

    new_meta = build_corrected_wcs_meta_scale_shift(
        map_in,
        dx_opt=dx,
        dy_opt=dy,
        sx_opt=sx,
        sy_opt=sy,
        verbose=verbose,
        linear_mode=linear_mode,
    )

    return sunpy.map.Map(map_in.data, new_meta, plot_settings=map_in.plot_settings)

def find_original_correction(
    corrected_map: GenericMap,
    uncorrected_map: GenericMap,
    target_map: GenericMap,
) -> GenericMap:
  """
  Given 2 maps where one was corrected and the other is not in the frame of the correct alignement. Find what would be the correction equivalent to in the frame of the uncorrected original frame which is target_map.
  param corrected_map: SunPy Map that has been corrected
  param uncorected_map: SunPy Map that has not been corrected
  param target_map: SunPy Map that is the target frame to find the correction in.
  """
  

  
  wrong_CRVAL_SPICE_HLP = SkyCoord(
    Quantity(target_map.meta["CRVAL1"],target_map.meta["CUNIT1"]),
    Quantity(target_map.meta["CRVAL2"],target_map.meta["CUNIT2"]),
    frame = target_map.coordinate_frame
  ) 
  wrong_CRVAL_FSI_HLP = wrong_CRVAL_SPICE_HLP.transform_to(uncorrected_map.coordinate_frame)
  wrong_CRVAL_FSI_PXL = uncorrected_map.world_to_pixel(wrong_CRVAL_FSI_HLP)
  corct_CRVAL_FSI_HLP = corrected_map.pixel_to_world(wrong_CRVAL_FSI_PXL[0], wrong_CRVAL_FSI_PXL[1])
  corct_CRVAL_SPICE_HLP = corct_CRVAL_FSI_HLP.transform_to(corrected_map.coordinate_frame)
  
  cdelt1_corrected = Quantity(corrected_map.meta['CDELT1'],corrected_map.meta['CUNIT1'])
  cdelt2_corrected = Quantity(corrected_map.meta['CDELT2'],corrected_map.meta['CUNIT2'])
  cdelt1_uncorrected = Quantity(uncorrected_map.meta['CDELT1'],uncorrected_map.meta['CUNIT1'])
  cdelt2_uncorrected = Quantity(uncorrected_map.meta['CDELT2'],uncorrected_map.meta['CUNIT2'])
  cdelt1_target      = Quantity(target_map.meta['CDELT1'],target_map.meta['CUNIT1'])
  cdelt2_target      = Quantity(target_map.meta['CDELT2'],target_map.meta['CUNIT2'])
  
  scale_x = cdelt1_corrected / cdelt1_uncorrected
  scale_y = cdelt2_corrected / cdelt2_uncorrected
  cdelt1_spice_corrected = cdelt1_target * scale_x
  cdelt2_spice_corrected = cdelt2_target * scale_y
  new_meta = target_map.meta.copy()
  new_meta['CDELT1'] = cdelt1_spice_corrected.to(new_meta['CUNIT1']).value
  new_meta['CDELT2'] = cdelt2_spice_corrected.to(new_meta['CUNIT2']).value
  new_meta['CRVAL1'] = corct_CRVAL_SPICE_HLP.spherical.lon.to(new_meta['CUNIT1']).value
  new_meta['CRVAL2'] = corct_CRVAL_SPICE_HLP.spherical.lat.to(new_meta['CUNIT2']).value
  new_map = Map(target_map.data, new_meta,plot_settings=target_map.plot_settings)
  
  return new_map
  # transform th coordinates to the uncorreted FSI frames 
  
  # uncorrected_CRVAL_coordinates_reprojected = uncorrected_map.()
  

def optimize_alignment_local_grad(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    dx0: float = 0.0,
    dy0: float = 0.0,
    squeeze_x0: float = 1.0,
    squeeze_y0: float = 1.0,
    # per-parameter initial steps
    step_dx0: float = 3.0,
    step_dy0: float = 3.0,
    step_sx0: float = 0.02,
    step_sy0: float = 0.02,
    # per-parameter minimum steps
    min_step_dx: float = 0.25,
    min_step_dy: float = 0.25,
    min_step_sx: float = 1e-3,
    min_step_sy: float = 1e-3,
    n_neighbors: int = 20,
    max_iter: int = 30,
    center: Optional[Tuple[float, float]] = None,
    squeeze_x_bounds: Tuple[float, float] = (0.9, 1.1),
    squeeze_y_bounds: Tuple[float, float] = (0.9, 1.1),
    shift_range: Optional[Tuple[int, int]] = None,
    n_jobs: int = 1,
    verbose: bool = True,
    # plateau detection in correlation space
    corr_atol: float = 1e-4,
    corr_rtol: float = 1e-3,
    plateau_iters: int = 3,
    # per-parameter shrink factors on *plateau* (division)
    # if a factor is None -> jump directly to the min for that parameter on plateau
    shrink_factor_dx: Optional[float] = 2.0,
    shrink_factor_dy: Optional[float] = 2.0,
    shrink_factor_sx: Optional[float] = 2.0,
    shrink_factor_sy: Optional[float] = 2.0,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Stochastic optimizer for (dx, dy, squeeze_x, squeeze_y).

    Behaviour summary
    -----------------
    - Parameters are a 4D vector p = (dx, dy, sx, sy).
    - Each iteration:
        * sample a population of neighbours around p using the current step[]
        * evaluate correlation for all
        * compute a correlation-weighted mean of parameters (mean_curr)
        * define a 4D "gradient" = mean_curr - mean_prev and move p along it
          (all parameters together)
    - Step sizes:
        * per-parameter initial steps: step_dx0, step_dy0, step_sx0, step_sy0
        * per-parameter minimum steps: min_step_dx, min_step_dy, min_step_sx, min_step_sy
        * if wanted a parameter frozen, set its initial step to 0 (and min to 0).

    Crucial plateau logic
    ---------------------
    - As long as correlation does NOT plateau, steps are NOT changed.
      No shrinking on rejected steps, etc.
    - Plateau detection:
        * use the last `plateau_iters` best correlations
        * compute spread = max - min over that window
        * if spread <= corr_atol + corr_rtol*|mean| -> plateau detected
    - On each plateau detection:
        * shrink steps per parameter by division / jump-to-min:
            - if shrink_factor_param is None:
                  step_param -> min_step_param immediately
            - else:
                  step_param -> max(step_param / shrink_factor_param, min_step_param)
        * plateau_count += 1 (we do NOT reset it because of shrinking)
    - plateau_count is only reset when correlation stops being flat
      (spread > tolerance).
    - Stopping condition:
        - plateau_count >= plateau_iters  AND
        - all active steps are at (or below) their min.
    """

    # --- safety checks for plateau shrink factors ---
    for name, fac in [
        ("shrink_factor_dx", shrink_factor_dx),
        ("shrink_factor_dy", shrink_factor_dy),
        ("shrink_factor_sx", shrink_factor_sx),
        ("shrink_factor_sy", shrink_factor_sy),
    ]:
        if fac is not None:
            assert fac > 1.0, f"{name} must be > 1.0 when provided."

    ref_img = np.asarray(ref_img, dtype=np.float64)
    target_img = np.asarray(target_img, dtype=np.float64)

    # parameter vector p = [dx, dy, sx, sy]
    p = np.array([dx0, dy0, squeeze_x0, squeeze_y0], dtype=float)

    # step sizes per component
    step = np.array([step_dx0, step_dy0, step_sx0, step_sy0], dtype=float)

    # per-parameter min steps
    min_step = np.array([min_step_dx, min_step_dy, min_step_sx, min_step_sy],
                        dtype=float)

    # plateau shrink factors as array (object to allow None)
    plateau_factors = np.array(
        [shrink_factor_dx, shrink_factor_dy, shrink_factor_sx, shrink_factor_sy],
        dtype=object,
    )

    def clamp_shift(dx, dy):
        if shift_range is None:
            return dx, dy
        smin, smax = shift_range
        return (
            float(np.clip(dx, smin, smax)),
            float(np.clip(dy, smin, smax)),
        )

    # shrink steps on plateau (division / jump-to-min)
    def shrink_steps_plateau(step_vec):
        new_step = step_vec.copy()
        for i in range(4):
            if new_step[i] <= 0.0:
                new_step[i] = 0.0
                continue
            fac = plateau_factors[i]
            if fac is None:
                new_step[i] = min_step[i]
            else:
                new_step[i] = max(new_step[i] / fac, min_step[i])
        return new_step

    # history: [iter, dx, dy, sx, sy, corr]
    history_rows = []
    corr_history = []

    # initial evaluation
    p[0], p[1] = clamp_shift(p[0], p[1])
    cur_corr = correlation_for_params(
        ref_img, target_img, p[0], p[1], p[2], p[3], center=center
    )
    history_rows.append([0, p[0], p[1], p[2], p[3], cur_corr])
    corr_history.append(cur_corr)

    if verbose:
        print(f"Initial: dx={p[0]:.3f}, dy={p[1]:.3f}, "
              f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")

    mean_prev = p.copy()
    global_best_p = p.copy()
    global_best_corr = cur_corr

    plateau_count = 0

    for it in range(1, max_iter + 1):

        # -------------------------------
        # 1) Sample population around p
        # -------------------------------
        params_list = [p.copy()]
        corrs_list = []

        c_center = correlation_for_params(
            ref_img, target_img,
            p[0], p[1], p[2], p[3],
            center=center
        )
        corrs_list.append(c_center)
        history_rows.append([it, p[0], p[1], p[2], p[3], c_center])

        # neighbours
        for _ in range(n_neighbors):
            delta = (2.0 * np.random.rand(4) - 1.0) * step
            cand = p + delta

            cand[2] = np.clip(cand[2], squeeze_x_bounds[0], squeeze_x_bounds[1])
            cand[3] = np.clip(cand[3], squeeze_y_bounds[0], squeeze_y_bounds[1])
            cand[0], cand[1] = clamp_shift(cand[0], cand[1])

            params_list.append(cand)

        # evaluate neighbours
        if n_jobs == 1:
            for cand in params_list[1:]:
                dx_c, dy_c, sx_c, sy_c = cand
                corr_c = correlation_for_params(
                    ref_img, target_img, dx_c, dy_c, sx_c, sy_c, center=center
                )
                corrs_list.append(corr_c)
                history_rows.append([it, dx_c, dy_c, sx_c, sy_c, corr_c])
        else:
            tasks = [
                (ref_img, target_img, cand[0], cand[1], cand[2], cand[3], center)
                for cand in params_list[1:]
            ]
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_corr_for_candidate, tasks)
            for cand, corr_c in zip(params_list[1:], results):
                dx_c, dy_c, sx_c, sy_c = cand
                corrs_list.append(corr_c)
                history_rows.append([it, dx_c, dy_c, sx_c, sy_c, corr_c])

        corrs = np.array(corrs_list, dtype=float)
        params = np.vstack(params_list)

        # -------------------------------
        # 2) Local & global best
        # -------------------------------
        idx_best_local = np.argmax(corrs)
        best_local_corr = corrs[idx_best_local]
        best_local_p = params[idx_best_local].copy()

        if best_local_corr > global_best_corr:
            global_best_corr = best_local_corr
            global_best_p = best_local_p.copy()

        if verbose:
            print(f"[iter {it}] best local corr = {best_local_corr:.6f}, "
                  f"global best = {global_best_corr:.6f}")

        corr_history.append(best_local_corr)

        # -------------------------------
        # 3) Population mean & gradient
        # -------------------------------
        c_min = np.min(corrs)
        weights = corrs - c_min + 1e-6
        wsum = np.sum(weights)
        if wsum <= 0:
            mean_curr = p.copy()
        else:
            mean_curr = np.sum(params * weights[:, None], axis=0) / wsum

        grad = mean_curr - mean_prev
        mean_prev = mean_curr.copy()

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0.0:
            direction = grad / grad_norm
        else:
            direction = np.zeros_like(grad)

        # -------------------------------
        # 4) Proposed new center
        # -------------------------------
        base_step_len = np.linalg.norm(step)
        if grad_norm == 0.0 or base_step_len == 0.0:
            p_proposed = best_local_p.copy()
            corr_proposed = best_local_corr
        else:
            # scale based on correlation level
            if best_local_corr < 0.6:
                factor = 2.0
            elif best_local_corr < 0.85:
                factor = 1.0
            else:
                factor = 0.5

            step_len = base_step_len * factor
            p_proposed = p + step_len * direction

            p_proposed[0], p_proposed[1] = clamp_shift(p_proposed[0], p_proposed[1])
            p_proposed[2] = np.clip(p_proposed[2], squeeze_x_bounds[0],
                                    squeeze_x_bounds[1])
            p_proposed[3] = np.clip(p_proposed[3], squeeze_y_bounds[0],
                                    squeeze_y_bounds[1])

            corr_proposed = correlation_for_params(
                ref_img, target_img,
                p_proposed[0], p_proposed[1],
                p_proposed[2], p_proposed[3],
                center=center
            )
            history_rows.append([it, p_proposed[0], p_proposed[1],
                                 p_proposed[2], p_proposed[3], corr_proposed])

        # accept if not worse than best_local, else fallback — BUT no shrinking here
        if corr_proposed >= best_local_corr:
            p = p_proposed
            cur_corr = corr_proposed
            if verbose:
                print(f"[iter {it}] gradient step accepted: "
                      f"dx={p[0]:.3f}, dy={p[1]:.3f}, "
                      f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")
        else:
            p = best_local_p.copy()
            cur_corr = best_local_corr
            if verbose:
                print(f"[iter {it}] fallback to best local (no step shrink).")

        # -------------------------------
        # 5) Plateau detection & step shrink
        # -------------------------------
        if len(corr_history) >= plateau_iters:
            recent = np.array(corr_history[-plateau_iters:], dtype=float)
            c_max = recent.max()
            c_min_recent = recent.min()
            c_mean = recent.mean()

            spread = c_max - c_min_recent
            tol = corr_atol + corr_rtol * abs(c_mean)

            if spread <= tol:
                plateau_count += 1
                if verbose:
                    print(f"[iter {it}] plateau: spread={spread:.3e}, "
                          f"tol={tol:.3e}, plateau_count={plateau_count}")
                # NOW shrink steps (only here)
                step = shrink_steps_plateau(step)

                # stop if plateau persisted AND all active steps at min
                all_at_min = True
                for i in range(4):
                    if step[i] > 0.0 and step[i] > min_step[i] + 1e-12:
                        all_at_min = False
                        break
                if plateau_count >= plateau_iters and all_at_min:
                    if verbose:
                        print("Plateau persisted and all steps at min; stopping.")
                    break
            else:
                # correlation moved again -> reset plateau count
                plateau_count = 0

    best_params = {
        "dx": p[0],
        "dy": p[1],
        "squeeze_x": p[2],
        "squeeze_y": p[3],
        "corr": cur_corr,
    }

    if verbose:
        print("\nFinal center: "
              f"dx={p[0]:.3f}, dy={p[1]:.3f}, "
              f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")
        print("Global best corr found:", global_best_corr)

    history = np.array(history_rows, dtype=float)
    return best_params, history

from numba import jit
import numpy as np
from multiprocessing import Pool

@jit(nopython=True)
def fst_neighbors_4d(
    extent: float,
    size0: float = 1.0,
    size1: float = 1.0,
    size2: float = 1.0,
    size3: float = 1.0,
    verbose: int = 0,
) -> List[List[float]]:
    ext_int = int(extent)
    sizes = np.array((size0, size1, size2, size3))
    min_size = sizes.min()

    # normalized weights so the smallest axis has weight 1
    r0 = min_size / size0
    r1 = min_size / size1
    r2 = min_size / size2
    r3 = min_size / size3

    if verbose >= 1:
        print("sizes:", size0, size1, size2, size3)
        print("r:", r0, r1, r2, r3)

    km_list = []
    extent_2 = extent * extent

    for k0 in range(-ext_int, ext_int + 1):
        for k1 in range(-ext_int, ext_int + 1):
            for k2 in range(-ext_int, ext_int + 1):
                for k3 in range(-ext_int, ext_int + 1):
                    # skip the origin
                    if k0 == 0 and k1 == 0 and k2 == 0 and k3 == 0:
                        continue

                    s = (
                        (r0 * k0) * (r0 * k0) +
                        (r1 * k1) * (r1 * k1) +
                        (r2 * k2) * (r2 * k2) +
                        (r3 * k3) * (r3 * k3)
                    )

                    if s <= extent_2:
                        km_list.append([k0, k1, k2, k3, s])

    return km_list

import numpy as np

def collapse_normalized_offsets_for_frozen_dims(
    normalized_offsets: np.ndarray,
    step: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    Given:
      - normalized_offsets: array of shape (N, 4) with integer offsets (k0, k1, k2, k3)
      - step: array of shape (4,) with per-parameter step sizes

    If step[i] == 0 for some i (frozen dimension), then any offsets that differ
    only in that dimension are equivalent in parameter space.

    This function:
      1) zeroes out the frozen dimensions in normalized_offsets
      2) removes duplicate rows while preserving order

    Returns
    -------
    collapsed : np.ndarray of shape (M, 4)
        Filtered offsets, still ordered from closest to farthest.
    """
    normalized_offsets = np.asarray(normalized_offsets, dtype=np.int64)
    step = np.asarray(step, dtype=float)

    # frozen dims: step == 0  (these will never change)
    frozen = (step == 0.0)
    if not np.any(frozen):
        # nothing to do
        return normalized_offsets

    # copy & zero-out frozen dimensions
    offsets = normalized_offsets.copy()
    offsets[:, frozen] = 0

    # remove duplicate rows while preserving order
    unique_rows = []
    seen = set()
    for row in offsets:
        key = tuple(int(v) for v in row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    collapsed = np.array(unique_rows, dtype=np.int64)
    if verbose:
        print(
            "collapse_normalized_offsets_for_frozen_dims: "
            f"{normalized_offsets.shape[0]} -> {collapsed.shape[0]} offsets "
            f"(frozen dims = {np.where(frozen)[0].tolist()})"
        )
    return collapsed


def optimize_alignment_local_grad_disc(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    dx0: float = 0.0,
    dy0: float = 0.0,
    squeeze_x0: float = 1.0,
    squeeze_y0: float = 1.0,
    # per-parameter initial steps
    step_dx0: float = 3.0,
    step_dy0: float = 3.0,
    step_sx0: float = 0.02,
    step_sy0: float = 0.02,
    # per-parameter minimum steps
    min_step_dx: float = 0.25,
    min_step_dy: float = 0.25,
    min_step_sx: float = 1e-3,
    min_step_sy: float = 1e-3,
    n_neighbors: int = 20,
    max_iter: int = 30,
    center: Optional[Tuple[float, float]] = None,
    squeeze_x_bounds: Tuple[float, float] = (0.9, 1.1),
    squeeze_y_bounds: Tuple[float, float] = (0.9, 1.1),
    shift_range: Optional[Tuple[int, int]] = None,
    n_jobs: int = 1,
    verbose: bool = True,
    # plateau detection in correlation space
    corr_atol: float = 1e-4,
    corr_rtol: float = 1e-3,
    plateau_iters: int = 3,
    # per-parameter shrink factors on *plateau* (division)
    # if a factor is None -> jump directly to the min for that parameter on plateau
    shrink_factor_dx: Optional[float] = 2.0,
    shrink_factor_dy: Optional[float] = 2.0,
    shrink_factor_sx: Optional[float] = 2.0,
    shrink_factor_sy: Optional[float] = 2.0,
    # how far to go in the integer-lattice neighbour generator
    neighbor_extent: int = 11,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Discrete stochastic optimizer for (dx, dy, squeeze_x, squeeze_y).

    Neighbours are drawn from a precomputed 4D integer lattice,
    ordered from closest to farthest in an anisotropic metric,
    then scaled by the current step = [step_dx, step_dy, step_sx, step_sy].
    """

    # --- safety checks for plateau shrink factors ---
    for name, fac in [
        ("shrink_factor_dx", shrink_factor_dx),
        ("shrink_factor_dy", shrink_factor_dy),
        ("shrink_factor_sx", shrink_factor_sx),
        ("shrink_factor_sy", shrink_factor_sy),
    ]:
        if fac is not None:
            assert fac > 1.0, f"{name} must be > 1.0 when provided."

    ref_img = np.asarray(ref_img, dtype=np.float64)
    target_img = np.asarray(target_img, dtype=np.float64)

    # ---- precompute normalized integer offsets, ordered by distance ----
    VALs = fst_neighbors_4d(
        neighbor_extent,
        size0=1.0,
        size1=1.0,
        size2=1.0,
        size3=1.0,
        verbose=0,
    )
    # sort by s (distance^2)
    VALs = sorted(VALs, key=lambda x: x[4])
    # keep only (k0, k1, k2, k3)
    normalized_offsets = np.array(VALs, dtype=np.int64)[:, 0:4]

    # simple cache: param_key -> correlation
    cache = {}

    # parameter vector p = [dx, dy, sx, sy]
    p = np.array([dx0, dy0, squeeze_x0, squeeze_y0], dtype=float)

    # step sizes per component
    step = np.array([step_dx0, step_dy0, step_sx0, step_sy0], dtype=float)

    # per-parameter min steps
    min_step = np.array(
        [min_step_dx, min_step_dy, min_step_sx, min_step_sy],
        dtype=float
    )

    # plateau shrink factors as array (object to allow None)
    plateau_factors = np.array(
        [shrink_factor_dx, shrink_factor_dy, shrink_factor_sx, shrink_factor_sy],
        dtype=object,
    )

    def clamp_shift(dx, dy):
        if shift_range is None:
            return dx, dy
        smin, smax = shift_range
        return (
            float(np.clip(dx, smin, smax)),
            float(np.clip(dy, smin, smax)),
        )

    # shrink steps on plateau (division / jump-to-min)
    def shrink_steps_plateau(step_vec):
        new_step = step_vec.copy()
        for i in range(4):
            if new_step[i] <= 0.0:
                new_step[i] = 0.0
                continue
            fac = plateau_factors[i]
            if fac is None:
                new_step[i] = min_step[i]
            else:
                new_step[i] = max(new_step[i] / fac, min_step[i])
        return new_step

    # history: [iter, dx, dy, sx, sy, corr]
    history_rows = []
    corr_history = []

    # ---------- initial evaluation ----------
    p[0], p[1] = clamp_shift(p[0], p[1])
    key0 = tuple(np.round(p, 6))
    if key0 in cache:
        cur_corr = cache[key0]
    else:
        cur_corr = correlation_for_params(
            ref_img, target_img, p[0], p[1], p[2], p[3], center=center
        )
        cache[key0] = cur_corr

    history_rows.append([0, p[0], p[1], p[2], p[3], cur_corr])
    corr_history.append(cur_corr)

    if verbose:
        print(f"Initial: dx={p[0]:.3f}, dy={p[1]:.3f}, "
              f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")

    mean_prev = p.copy()
    global_best_p = p.copy()
    global_best_corr = cur_corr

    plateau_count = 0

    for it in range(1, max_iter + 1):

        # -------------------------------
        # 1) Sample population around p
        # -------------------------------
        params_list = []
        corrs_list = []

        # center point
        key_center = tuple(np.round(p, 6))
        if key_center in cache:
            c_center = cache[key_center]
        else:
            c_center = correlation_for_params(
                ref_img, target_img,
                p[0], p[1], p[2], p[3],
                center=center
            )
            cache[key_center] = c_center

        params_list.append(p.copy())
        corrs_list.append(c_center)
        history_rows.append([it, p[0], p[1], p[2], p[3], c_center])

        # neighbours: use closest integer offsets, scaled by current step
        neighbors = []
        seen_keys = set()
        for k in normalized_offsets:
            if len(neighbors) >= n_neighbors:
                break

            delta = np.array(k, dtype=float) * step  # element-wise
            cand = p + delta

            # clamp scales & shifts
            cand[2] = np.clip(cand[2], squeeze_x_bounds[0], squeeze_x_bounds[1])
            cand[3] = np.clip(cand[3], squeeze_y_bounds[0], squeeze_y_bounds[1])
            cand[0], cand[1] = clamp_shift(cand[0], cand[1])

            key = tuple(np.round(cand, 6))

            # avoid duplicates in THIS iteration
            if key in seen_keys or key == key_center:
                continue
            seen_keys.add(key)

            # mark whether we need to evaluate or can reuse cache
            needs_eval = key not in cache
            neighbors.append((cand, key, needs_eval))

        # evaluate neighbours (some from cache, some new)
        if n_jobs == 1:
            for cand, key, needs_eval in neighbors:
                if needs_eval:
                    dx_c, dy_c, sx_c, sy_c = cand
                    corr_c = correlation_for_params(
                        ref_img, target_img,
                        dx_c, dy_c, sx_c, sy_c,
                        center=center
                    )
                    cache[key] = corr_c
                else:
                    corr_c = cache[key]

                params_list.append(cand)
                corrs_list.append(corr_c)
                history_rows.append(
                    [it, cand[0], cand[1], cand[2], cand[3], corr_c]
                )
        else:
            # parallel only for those that need evaluation
            tasks = []
            idx_eval = []
            for idx, (cand, key, needs_eval) in enumerate(neighbors):
                if needs_eval:
                    tasks.append(
                        (ref_img, target_img, cand[0], cand[1], cand[2], cand[3], center)
                    )
                    idx_eval.append(idx)

            if tasks:
                with Pool(processes=n_jobs) as pool:
                    results = pool.map(_corr_for_candidate, tasks)
            else:
                results = []

            # assign results back
            eval_iter = iter(results)
            for cand, key, needs_eval in neighbors:
                if needs_eval:
                    corr_c = next(eval_iter)
                    cache[key] = corr_c
                else:
                    corr_c = cache[key]

                params_list.append(cand)
                corrs_list.append(corr_c)
                history_rows.append(
                    [it, cand[0], cand[1], cand[2], cand[3], corr_c]
                )

        corrs = np.array(corrs_list, dtype=float)
        params = np.vstack(params_list)

        # -------------------------------
        # 2) Local & global best
        # -------------------------------
        idx_best_local = np.argmax(corrs)
        best_local_corr = corrs[idx_best_local]
        best_local_p = params[idx_best_local].copy()

        if best_local_corr > global_best_corr:
            global_best_corr = best_local_corr
            global_best_p = best_local_p.copy()

        if verbose:
            print(f"[iter {it}] best local corr = {best_local_corr:.6f}, "
                  f"global best = {global_best_corr:.6f}")

        corr_history.append(best_local_corr)

        # -------------------------------
        # 3) Population mean & gradient
        # -------------------------------
        c_min = np.min(corrs)
        weights = corrs - c_min + 1e-6
        wsum = np.sum(weights)
        if wsum <= 0:
            mean_curr = p.copy()
        else:
            mean_curr = np.sum(params * weights[:, None], axis=0) / wsum

        grad = mean_curr - mean_prev
        mean_prev = mean_curr.copy()

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0.0:
            direction = grad / grad_norm
        else:
            direction = np.zeros_like(grad)

        # -------------------------------
        # 4) Proposed new center
        # -------------------------------
        base_step_len = np.linalg.norm(step)
        if grad_norm == 0.0 or base_step_len == 0.0:
            p_proposed = best_local_p.copy()
            corr_proposed = best_local_corr
        else:
            # scale based on correlation level
            if best_local_corr < 0.6:
                factor = 2.0
            elif best_local_corr < 0.85:
                factor = 1.0
            else:
                factor = 0.5

            step_len = base_step_len * factor
            p_proposed = p + step_len * direction

            p_proposed[0], p_proposed[1] = clamp_shift(p_proposed[0], p_proposed[1])
            p_proposed[2] = np.clip(p_proposed[2], squeeze_x_bounds[0],
                                    squeeze_x_bounds[1])
            p_proposed[3] = np.clip(p_proposed[3], squeeze_y_bounds[0],
                                    squeeze_y_bounds[1])

            key_prop = tuple(np.round(p_proposed, 6))
            if key_prop in cache:
                corr_proposed = cache[key_prop]
            else:
                corr_proposed = correlation_for_params(
                    ref_img, target_img,
                    p_proposed[0], p_proposed[1],
                    p_proposed[2], p_proposed[3],
                    center=center
                )
                cache[key_prop] = corr_proposed

            history_rows.append([it, p_proposed[0], p_proposed[1],
                                 p_proposed[2], p_proposed[3], corr_proposed])

        # accept if not worse than best_local, else fallback (no shrink here)
        if corr_proposed >= best_local_corr:
            p = p_proposed
            cur_corr = corr_proposed
            if verbose:
                print(f"[iter {it}] gradient step accepted: "
                      f"dx={p[0]:.3f}, dy={p[1]:.3f}, "
                      f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")
        else:
            p = best_local_p.copy()
            cur_corr = best_local_corr
            if verbose:
                print(f"[iter {it}] fallback to best local (no step shrink).")

        # -------------------------------
        # 5) Plateau detection & step shrink
        # -------------------------------
        if len(corr_history) >= plateau_iters:
            recent = np.array(corr_history[-plateau_iters:], dtype=float)
            c_max = recent.max()
            c_min_recent = recent.min()
            c_mean = recent.mean()

            spread = c_max - c_min_recent
            tol = corr_atol + corr_rtol * abs(c_mean)

            if spread <= tol:
                plateau_count += 1
                if verbose:
                    print(f"[iter {it}] plateau: spread={spread:.3e}, "
                          f"tol={tol:.3e}, plateau_count={plateau_count}")
                # shrink steps only here
                step = shrink_steps_plateau(step)

                # stop if plateau persisted AND all active steps at min
                all_at_min = True
                for i in range(4):
                    if step[i] > 0.0 and step[i] > min_step[i] + 1e-12:
                        all_at_min = False
                        break
                if plateau_count >= plateau_iters and all_at_min:
                    if verbose:
                        print("Plateau persisted and all steps at min; stopping.")
                    break
            else:
                plateau_count = 0

    best_params = {
        "dx": p[0],
        "dy": p[1],
        "squeeze_x": p[2],
        "squeeze_y": p[3],
        "corr": cur_corr,
    }

    if verbose:
        print("\nFinal center: "
              f"dx={p[0]:.3f}, dy={p[1]:.3f}, "
              f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")
        print("Global best corr found:", global_best_corr)

    history = np.array(history_rows, dtype=float)
    return best_params, history




from help_funcs import _vprint

def _corr_worker_loop(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    center: Optional[Tuple[float, float]],
    task_queue: Queue,
    result_queue: Queue,
    verbose: int = 0,
) -> None:
    """
    Persistent worker:
    - ref_img, target_img, center are captured once at process start.
    - Then we loop forever, waiting for tasks on task_queue.
    - Each task is either:
        * (job_id, dx, dy, sx, sy)
        * or None (sentinel to stop).
    - For each job, compute corr and send (job_id, corr) back.
    """
    while True:
        _vprint(verbose, 2, "Worker waiting for task...")
        task = task_queue.get()
        if task is None:
            # sentinel: time to stop this worker
            break

        job_id, dx, dy, sx, sy = task
        corr = correlation_for_params(
            ref_img, target_img,
            dx, dy, sx, sy,
            center=center
        )
        result_queue.put((job_id, corr))


from multiprocessing import Process, Queue
import numpy as np

def optimize_alignment_local_grad_disc_persworkers(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    dx0: float = 0.0,
    dy0: float = 0.0,
    squeeze_x0: float = 1.0,
    squeeze_y0: float = 1.0,
    # per-parameter initial steps
    step_dx0: float = 3.0,
    step_dy0: float = 3.0,
    step_sx0: float = 0.02,
    step_sy0: float = 0.02,
    # per-parameter minimum steps
    min_step_dx: float = 0.25,
    min_step_dy: float = 0.25,
    min_step_sx: float = 1e-3,
    min_step_sy: float = 1e-3,
    n_neighbors: int = 20,
    max_iter: int = 30,
    center: Optional[Tuple[float, float]] = None,
    squeeze_x_bounds: Tuple[float, float] = (0.9, 1.1),
    squeeze_y_bounds: Tuple[float, float] = (0.9, 1.1),
    shift_range: Optional[Tuple[int, int]] = None,
    n_jobs: int = 1,
    verbose: bool = True,
    # plateau detection in correlation space
    corr_atol: float = 1e-4,
    corr_rtol: float = 1e-3,
    plateau_iters: int = 3,
    # per-parameter shrink factors on *plateau* (division)
    # if a factor is None -> jump directly to the min for that parameter on plateau
    shrink_factor_dx: Optional[float] = 2.0,
    shrink_factor_dy: Optional[float] = 2.0,
    shrink_factor_sx: Optional[float] = 2.0,
    shrink_factor_sy: Optional[float] = 2.0,
    # how far to go in the integer-lattice neighbour generator
    neighbor_extent: int = 11,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Discrete stochastic optimizer for (dx, dy, squeeze_x, squeeze_y).

    Neighbours are drawn from a precomputed 4D integer lattice,
    ordered from closest to farthest in an anisotropic metric,
    then scaled by the current step = [step_dx, step_dy, step_sx, step_sy].

    Parallel mode (n_jobs > 1):
    ----------------------------
    - Starts `n_jobs` worker processes.
    - Each worker holds `ref_img`, `target_img`, and `center` in memory.
    - Main loop sends parameter sets (dx, dy, sx, sy) via a task_queue.
    - Workers compute correlation and return via result_queue.
    - Workers persist for the entire optimization and are then shut down.
    """

    # --- safety checks for plateau shrink factors ---
    for name, fac in [
        ("shrink_factor_dx", shrink_factor_dx),
        ("shrink_factor_dy", shrink_factor_dy),
        ("shrink_factor_sx", shrink_factor_sx),
        ("shrink_factor_sy", shrink_factor_sy),
    ]:
        if fac is not None:
            assert fac > 1.0, f"{name} must be > 1.0 when provided."

    ref_img = np.asarray(ref_img, dtype=np.float64)
    target_img = np.asarray(target_img, dtype=np.float64)

    # ---- precompute normalized integer offsets, ordered by distance ----
    VALs = fst_neighbors_4d(
        neighbor_extent,
        size0=1.0,
        size1=1.0,
        size2=1.0,
        size3=1.0,
        verbose=0,
    )
    VALs = sorted(VALs, key=lambda x: x[4])
    normalized_offsets = np.array(VALs, dtype=np.int64)[:, 0:4]

    # simple cache: param_key -> correlation
    cache = {}

    # parameter vector p = [dx, dy, sx, sy]
    p = np.array([dx0, dy0, squeeze_x0, squeeze_y0], dtype=float)

    # step sizes per component
    step = np.array([step_dx0, step_dy0, step_sx0, step_sy0], dtype=float)

    # collapse offsets along frozen dimensions (step == 0)
    normalized_offsets = collapse_normalized_offsets_for_frozen_dims(
        normalized_offsets,
        step,
        verbose=verbose,
    )
    
    # per-parameter min steps
    min_step = np.array(
        [min_step_dx, min_step_dy, min_step_sx, min_step_sy],
        dtype=float
    )

    # plateau shrink factors as array (object to allow None)
    plateau_factors = np.array(
        [shrink_factor_dx, shrink_factor_dy, shrink_factor_sx, shrink_factor_sy],
        dtype=object,
    )

    def clamp_shift(dx, dy):
        if shift_range is None:
            return dx, dy
        smin, smax = shift_range
        return (
            float(np.clip(dx, smin, smax)),
            float(np.clip(dy, smin, smax)),
        )

    # shrink steps on plateau (division / jump-to-min)
    def shrink_steps_plateau(step_vec):
        new_step = step_vec.copy()
        for i in range(4):
            if new_step[i] <= 0.0:
                new_step[i] = 0.0
                continue
            fac = plateau_factors[i]
            if fac is None:
                new_step[i] = min_step[i]
            else:
                new_step[i] = max(new_step[i] / fac, min_step[i])
        return new_step

    # history: [iter, dx, dy, sx, sy, corr]
    history_rows = []
    corr_history = []

    # ---------- parallel infrastructure ----------
    workers = []
    task_queue = None
    result_queue = None
    next_job_id = 0  # monotonically increasing job id

    if n_jobs > 1:
        task_queue = Queue()
        result_queue = Queue()
        for _ in range(n_jobs):
            proc = Process(
                target=_corr_worker_loop,
                args=(ref_img, target_img, center, task_queue, result_queue),
            )
            proc.daemon = True
            proc.start()
            workers.append(proc)

    try:
        # ---------- initial evaluation ----------
        p[0], p[1] = clamp_shift(p[0], p[1])
        key0 = tuple(np.round(p, 6))
        if key0 in cache:
            cur_corr = cache[key0]
        else:
            if n_jobs > 1:
                # Evaluate using worker logic or directly — direct call is simpler here.
                cur_corr = correlation_for_params(
                    ref_img, target_img, p[0], p[1], p[2], p[3], center=center
                )
            else:
                cur_corr = correlation_for_params(
                    ref_img, target_img, p[0], p[1], p[2], p[3], center=center
                )
            cache[key0] = cur_corr

        history_rows.append([0, p[0], p[1], p[2], p[3], cur_corr])
        corr_history.append(cur_corr)

        if verbose:
            print(f"Initial: dx={p[0]:.3f}, dy={p[1]:.3f}, "
                  f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")

        mean_prev = p.copy()
        global_best_p = p.copy()
        global_best_corr = cur_corr

        plateau_count = 0

        # ===================== main optimization loop =====================
        for it in range(1, max_iter + 1):

            # -------------------------------
            # 1) Sample population around p
            # -------------------------------
            params_list = []
            corrs_list = []

            # center point
            key_center = tuple(np.round(p, 6))
            if key_center in cache:
                c_center = cache[key_center]
            else:
                c_center = correlation_for_params(
                    ref_img, target_img,
                    p[0], p[1], p[2], p[3],
                    center=center
                )
                cache[key_center] = c_center

            params_list.append(p.copy())
            corrs_list.append(c_center)
            history_rows.append([it, p[0], p[1], p[2], p[3], c_center])

            # neighbours: use closest integer offsets, scaled by current step
            neighbors = []
            seen_keys = set()
            for k in normalized_offsets:
                if len(neighbors) >= n_neighbors:
                    break

                delta = np.array(k, dtype=float) * step  # element-wise
                cand = p + delta

                # clamp scales & shifts
                cand[2] = np.clip(cand[2], squeeze_x_bounds[0], squeeze_x_bounds[1])
                cand[3] = np.clip(cand[3], squeeze_y_bounds[0], squeeze_y_bounds[1])
                cand[0], cand[1] = clamp_shift(cand[0], cand[1])

                key = tuple(np.round(cand, 6))

                # avoid duplicates in THIS iteration
                if key in seen_keys or key == key_center:
                    continue
                seen_keys.add(key)

                needs_eval = key not in cache
                neighbors.append((cand, key, needs_eval))

            # -------------------------------
            # 1b) Evaluate neighbours
            # -------------------------------
            if n_jobs == 1 or not workers:
                # serial path
                for cand, key, needs_eval in neighbors:
                    if needs_eval:
                        dx_c, dy_c, sx_c, sy_c = cand
                        corr_c = correlation_for_params(
                            ref_img, target_img,
                            dx_c, dy_c, sx_c, sy_c,
                            center=center
                        )
                        cache[key] = corr_c
                    else:
                        corr_c = cache[key]

                    params_list.append(cand)
                    corrs_list.append(corr_c)
                    history_rows.append(
                        [it, cand[0], cand[1], cand[2], cand[3], corr_c]
                    )
            else:
                # parallel path with persistent workers
                # split neighbors into cached and needing evaluation
                jobs_to_send = []
                jobid_to_data = {}
                n_new = 0

                # first: add cached neighbors immediately
                for cand, key, needs_eval in neighbors:
                    if not needs_eval:
                        corr_c = cache[key]
                        params_list.append(cand)
                        corrs_list.append(corr_c)
                        history_rows.append(
                            [it, cand[0], cand[1], cand[2], cand[3], corr_c]
                        )
                    else:
                        # queue later
                        job_id = next_job_id
                        next_job_id += 1
                        jobs_to_send.append((job_id, cand, key))
                        jobid_to_data[job_id] = (cand, key)
                        n_new += 1

                # send tasks to workers
                for job_id, cand, key in jobs_to_send:
                    dx_c, dy_c, sx_c, sy_c = cand
                    task_queue.put((job_id, dx_c, dy_c, sx_c, sy_c))

                # collect results for this iteration
                for _ in range(n_new):
                    job_id, corr_c = result_queue.get()
                    cand, key = jobid_to_data[job_id]
                    cache[key] = corr_c

                    params_list.append(cand)
                    corrs_list.append(corr_c)
                    history_rows.append(
                        [it, cand[0], cand[1], cand[2], cand[3], corr_c]
                    )

            corrs = np.array(corrs_list, dtype=float)
            params = np.vstack(params_list)

            # -------------------------------
            # 2) Local & global best
            # -------------------------------
            idx_best_local = np.argmax(corrs)
            best_local_corr = corrs[idx_best_local]
            best_local_p = params[idx_best_local].copy()

            if best_local_corr > global_best_corr:
                global_best_corr = best_local_corr
                global_best_p = best_local_p.copy()

            if verbose:
                print(f"[iter {it}] best local corr = {best_local_corr:.6f}, "
                      f"global best = {global_best_corr:.6f}")

            corr_history.append(best_local_corr)

            # -------------------------------
            # 3) Population mean & gradient
            # -------------------------------
            c_min = np.min(corrs)
            weights = corrs - c_min + 1e-6
            wsum = np.sum(weights)
            if wsum <= 0:
                mean_curr = p.copy()
            else:
                mean_curr = np.sum(params * weights[:, None], axis=0) / wsum

            grad = mean_curr - mean_prev

            # 🔒 mask out frozen parameters in the gradient
            active = (step > 0.0)
            grad = grad * active

            mean_prev = mean_curr.copy()

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0.0:
                direction = grad / grad_norm
            else:
                direction = np.zeros_like(grad)

            # -------------------------------
            # 4) Proposed new center
            # -------------------------------
            base_step_len = np.linalg.norm(step)
            if grad_norm == 0.0 or base_step_len == 0.0:
                p_proposed = best_local_p.copy()
                corr_proposed = best_local_corr
            else:
                # scale based on correlation level
                if best_local_corr < 0.6:
                    factor = 2.0
                elif best_local_corr < 0.85:
                    factor = 1.0
                else:
                    factor = 0.5

                step_len = base_step_len * factor
                p_proposed = p + step_len * direction

                p_proposed[0], p_proposed[1] = clamp_shift(p_proposed[0], p_proposed[1])
                p_proposed[2] = np.clip(p_proposed[2], squeeze_x_bounds[0],
                                        squeeze_x_bounds[1])
                p_proposed[3] = np.clip(p_proposed[3], squeeze_y_bounds[0],
                                        squeeze_y_bounds[1])
                # 🔒 hard-freeze any dimension with step == 0
                for i in range(4):
                    if step[i] == 0.0:
                        p_proposed[i] = p[i]

                key_prop = tuple(np.round(p_proposed, 6))
                if key_prop in cache:
                    corr_proposed = cache[key_prop]
                else:
                    corr_proposed = correlation_for_params(
                        ref_img, target_img,
                        p_proposed[0], p_proposed[1],
                        p_proposed[2], p_proposed[3],
                        center=center
                    )
                    cache[key_prop] = corr_proposed

                history_rows.append([it, p_proposed[0], p_proposed[1],
                                     p_proposed[2], p_proposed[3], corr_proposed])

            # accept if not worse than best_local, else fallback
            if corr_proposed >= best_local_corr:
                p = p_proposed
                cur_corr = corr_proposed
                if verbose:
                    print(f"[iter {it}] gradient step accepted: "
                          f"dx={p[0]:.3f}, dy={p[1]:.3f}, "
                          f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")
            else:
                p = best_local_p.copy()
                cur_corr = best_local_corr
                if verbose:
                    print(f"[iter {it}] fallback to best local (no step shrink).")
            
            # -------------------------------
            # 5) Plateau detection & step shrink
            # -------------------------------
            if len(corr_history) >= plateau_iters:
                recent = np.array(corr_history[-plateau_iters:], dtype=float)
                c_max = recent.max()
                c_min_recent = recent.min()
                c_mean = recent.mean()

                spread = c_max - c_min_recent
                tol = corr_atol + corr_rtol * abs(c_mean)

                if spread <= tol:
                    plateau_count += 1
                    if verbose:
                        print(f"[iter {it}] plateau: spread={spread:.3e}, "
                              f"tol={tol:.3e}, plateau_count={plateau_count}")
                    if plateau_count >= plateau_iters and all_at_min:
                        if verbose:
                            print("Plateau persisted and all steps at min; stopping.")
                        break
                    # stop if plateau persisted AND all active steps at min
                    all_at_min = True
                    for i in range(4):
                        if step[i] > 0.0 and step[i] > min_step[i] + 1e-12:
                            all_at_min = False
                            break
                    if plateau_count >= plateau_iters:
                        if verbose:
                            print("Plateau detected; shrinking steps.")
                        # shrink steps only here
                        step = shrink_steps_plateau(step)
                        if verbose:
                            print(f"New steps: dx={step[0]:.3f}, dy={step[1]:.3f}, "
                                  f"sx={step[2]:.5f}, sy={step[3]:.5f}")
                        plateau_count = 0
                        
                    
                    
                else:
                    plateau_count = 0

        best_params = {
            "dx": p[0],
            "dy": p[1],
            "squeeze_x": p[2],
            "squeeze_y": p[3],
            "corr": cur_corr,
        }

        if verbose:
            print("\nFinal center: "
                  f"dx={p[0]:.3f}, dy={p[1]:.3f}, "
                  f"sx={p[2]:.5f}, sy={p[3]:.5f}, corr={cur_corr:.6f}")
            print("Global best corr found:", global_best_corr)

        history = np.array(history_rows, dtype=float)
        return best_params, history

    finally:
        # shut down workers cleanly
        if workers:
            # send sentinel to each worker
            for _ in workers:
                task_queue.put(None)
            for proc in workers:
                proc.join()

def update_raster_coordinates(
    raster_set: List['SPICEL3Raster'],
    new_maps: List[GenericMap],
    histories: List[np.ndarray],
    redo: bool = False,
    verbose: int = 0,
) -> None:
    for ind in range(len(raster_set)):
      raster = raster_set[ind]
      date = np.datetime64(raster.lines[0].obs_date,'s')
      if verbose>=2:
        print(f"Raster {ind} with date {date}")
          
  
      #search the date that corresponds to the closest new_map in 
      for ind2 in range(len(new_maps)):
        new_map = new_maps[ind2]
        date_new_map = new_map.meta['date-obs']
        date_new_map = np.datetime64(date_new_map, 's')
        if date == date_new_map:
          if verbose>=2:
            print(f"Found matching map for raster {ind} at index {ind2} with date {date_new_map}")
          # update the raster coordinates
          history = histories[ind2]
          best_row = history[
            np.argmax(history[:, -1])
            ]
          iteration = best_row[0]
          best_params_pixel = best_row[1:-1]# dx dy sx sy
          corr = best_row[-1]
          new_crval1 = new_map.meta['CRVAL1']
          new_crval2 = new_map.meta['CRVAL2']
          new_cdelt1 = new_map.meta['CDELT1']
          new_cdelt2 = new_map.meta['CDELT2']
          new_cunit1 = new_map.meta['CUNIT1']
          new_cunit2 = new_map.meta['CUNIT2']

          old_crval1 = raster.lines[0].headers['rad'].get('CRVAL1_',None)
          old_cdelt1 = raster.lines[0].headers['rad'].get('CDELT1_',None)
          old_crval2 = raster.lines[0].headers['rad'].get('CRVAL2_',None)
          old_cdelt2 = raster.lines[0].headers['rad'].get('CDELT2_',None)
          if any(v is not None for v in [old_crval1, old_crval2, old_cdelt1, old_cdelt2]) and not redo:
            if verbose>=1:
              print(f'\033[91mOld CRVAL or CDELT values not found for raster {ind} with date {date}\033[0m')
            break
          else:
            if verbose>=2:
              print(f"_________________________________________________________________________________")
              print(f"_________________________________________________________________________________")
              print(f"Updating raster {ind} coordinates:")
              print(f"  Old CRVAL1: {old_crval1}, New CRVAL1: {new_crval1}")
              print(f"  Old CRVAL2: {old_crval2}, New CRVAL2: {new_crval2}")
              print(f"  Old CDELT1: {old_cdelt1}, New CDELT1: {new_cdelt1}")
              print(f"  Old CDELT2: {old_cdelt2}, New CDELT2: {new_cdelt2}")
              print(f"_________________________________________________________________________________")
              print(f"_________________________________________________________________________________")
              
            old_crval1 = raster.lines[0].headers['rad']['CRVAL1']
            old_crval2 = raster.lines[0].headers['rad']['CRVAL2']
            old_cdelt1 = raster.lines[0].headers['rad']['CDELT1']
            old_cdelt2 = raster.lines[0].headers['rad']['CDELT2']
          
          # now roll over all the lines and update the headers
          for ind3,line in enumerate(raster.lines):
            for par in line.headers.keys():
              raster_set[ind].lines[ind3].headers[par]['CDELT1_'] = old_cdelt1 if old_cdelt1 is not None else line.headers[par]['CDELT1']
              raster_set[ind].lines[ind3].headers[par]['CDELT2_'] = old_cdelt2 if old_cdelt2 is not None else line.headers[par]['CDELT2']
              raster_set[ind].lines[ind3].headers[par]['CRVAL2_'] = old_crval2 if old_crval2 is not None else line.headers[par]['CRVAL2']
              raster_set[ind].lines[ind3].headers[par]['CRVAL1_'] = old_crval1 if old_crval1 is not None else line.headers[par]['CRVAL1']
              raster_set[ind].lines[ind3].headers[par]['CRVAL1'] = Quantity(new_crval1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
              raster_set[ind].lines[ind3].headers[par]['CRVAL2'] = Quantity(new_crval2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
              raster_set[ind].lines[ind3].headers[par]['CDELT1'] = Quantity(new_cdelt1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
              raster_set[ind].lines[ind3].headers[par]['CDELT2'] = Quantity(new_cdelt2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
              raster_set[ind].lines[ind3].headers[par]['COALIGN_DX_PX'] = best_params_pixel[0]
              raster_set[ind].lines[ind3].headers[par]['COALIGN_DY_PX'] = best_params_pixel[1]
              raster_set[ind].lines[ind3].headers[par]['COALIGN_SX']    = best_params_pixel[2]
              raster_set[ind].lines[ind3].headers[par]['COALIGN_SY']    = best_params_pixel[3]
              raster_set[ind].lines[ind3].headers[par]['COALIGN_CORR']  = corr
              raster_set[ind].lines[ind3].headers[par]['COALIGN_IT']    = iteration
          
          
          raster_set[ind].FIP_header['CDELT1_'] = old_cdelt1 if old_cdelt1 is not None else line.headers[par]['CDELT1']
          raster_set[ind].FIP_header['CDELT2_'] = old_cdelt2 if old_cdelt2 is not None else line.headers[par]['CDELT2']
          raster_set[ind].FIP_header['CRVAL2_'] = old_crval2 if old_crval2 is not None else line.headers[par]['CRVAL2']
          raster_set[ind].FIP_header['CRVAL1_'] = old_crval1 if old_crval1 is not None else line.headers[par]['CRVAL1']
          raster_set[ind].FIP_header['CRVAL1'] = Quantity(new_crval1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
          raster_set[ind].FIP_header['CRVAL2'] = Quantity(new_crval2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
          raster_set[ind].FIP_header['CDELT1'] = Quantity(new_cdelt1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
          raster_set[ind].FIP_header['CDELT2'] = Quantity(new_cdelt2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
          raster_set[ind].FIP_header['COALIGN_DX_PX'] = best_params_pixel[0]
          raster_set[ind].FIP_header['COALIGN_DY_PX'] = best_params_pixel[1]
          raster_set[ind].FIP_header['COALIGN_SX']    = best_params_pixel[2]
          raster_set[ind].FIP_header['COALIGN_SY']    = best_params_pixel[3]
          raster_set[ind].FIP_header['COALIGN_CORR']  = corr
          raster_set[ind].FIP_header['COALIGN_IT']    = iteration
          
          raster_set[ind].FIP_err_header['CDELT1_'] = old_cdelt1 if old_cdelt1 is not None else line.headers[par]['CDELT1']
          raster_set[ind].FIP_err_header['CDELT2_'] = old_cdelt2 if old_cdelt2 is not None else line.headers[par]['CDELT2']
          raster_set[ind].FIP_err_header['CRVAL2_'] = old_crval2 if old_crval2 is not None else line.headers[par]['CRVAL2']
          raster_set[ind].FIP_err_header['CRVAL1_'] = old_crval1 if old_crval1 is not None else line.headers[par]['CRVAL1']
          raster_set[ind].FIP_err_header['CRVAL1'] = Quantity(new_crval1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
          raster_set[ind].FIP_err_header['CRVAL2'] = Quantity(new_crval2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
          raster_set[ind].FIP_err_header['CDELT1'] = Quantity(new_cdelt1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
          raster_set[ind].FIP_err_header['CDELT2'] = Quantity(new_cdelt2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
          raster_set[ind].FIP_err_header['COALIGN_DX_PX'] = best_params_pixel[0]
          raster_set[ind].FIP_err_header['COALIGN_DY_PX'] = best_params_pixel[1]
          raster_set[ind].FIP_err_header['COALIGN_SX']    = best_params_pixel[2]
          raster_set[ind].FIP_err_header['COALIGN_SY']    = best_params_pixel[3]
          raster_set[ind].FIP_err_header['COALIGN_CORR']  = corr
          raster_set[ind].FIP_err_header['COALIGN_IT']    = iteration
          
          raster_set[ind].density_header['CDELT1_'] = old_cdelt1 if old_cdelt1 is not None else line.headers[par]['CDELT1']
          raster_set[ind].density_header['CDELT2_'] = old_cdelt2 if old_cdelt2 is not None else line.headers[par]['CDELT2']
          raster_set[ind].density_header['CRVAL2_'] = old_crval2 if old_crval2 is not None else line.headers[par]['CRVAL2']
          raster_set[ind].density_header['CRVAL1_'] = old_crval1 if old_crval1 is not None else line.headers[par]['CRVAL1']
          raster_set[ind].density_header['CRVAL1'] = Quantity(new_crval1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
          raster_set[ind].density_header['CRVAL2'] = Quantity(new_crval2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
          raster_set[ind].density_header['CDELT1'] = Quantity(new_cdelt1,new_cunit1).to(line.headers['rad']['CUNIT1']).value
          raster_set[ind].density_header['CDELT2'] = Quantity(new_cdelt2,new_cunit2).to(line.headers['rad']['CUNIT2']).value
          raster_set[ind].density_header['COALIGN_DX_PX'] = best_params_pixel[0]
          raster_set[ind].density_header['COALIGN_DY_PX'] = best_params_pixel[1]
          raster_set[ind].density_header['COALIGN_SX']    = best_params_pixel[2]
          raster_set[ind].density_header['COALIGN_SY']    = best_params_pixel[3]
          raster_set[ind].density_header['COALIGN_CORR']  = corr
          raster_set[ind].density_header['COALIGN_IT']    = iteration
          
          break
          
            
          
          
        if ind2 == len(new_maps)-1:
          if verbose>=0:
            # in red
            print(f'\033[91mNo matching map found for raster {ind} with date {date}\033[0m')
    return 










