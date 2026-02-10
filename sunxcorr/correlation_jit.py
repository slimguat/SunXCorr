"""High-performance correlation helpers (Numba/JIT).

This module provides both reference (pure NumPy) and Numba-accelerated
implementations of the normalized correlation objective used by the
optimization routines. The JIT variant supports a prebuilt context to
avoid recomputing masks per call.
"""

from __future__ import annotations

import numpy as np
from numba import jit
from typing import Optional, Tuple, cast
from dataclasses import dataclass
from scipy.ndimage import affine_transform


def normalized_corr_nan_safe(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Normalized correlation between two same-shape images,
    ignoring NaNs (they do not contribute to sums).

    Parameters
    ----------
    img1, img2 : ndarray
        Arrays of identical shape. Values that are NaN in either input
        are ignored in the correlation computation.

    Returns
    -------
    float
        Pearson-like normalized correlation in [-1, 1]. Returns 0.0 when
        no overlapping finite pixels or when denominator is zero.
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
    """Rescale (stretch/squeeze) target_img onto reference grid using affine_transform.

    Parameters
    ----------
    target_img : ndarray
        Image to be resampled.
    ref_shape : tuple[int, int]
        Output shape (ny, nx) of the reference grid.
    squeeze_x, squeeze_y : float
        Scaling factors applied to x and y axes (forward squeeze).
    center : tuple or None
        Center pixel coordinates for scaling; defaults to image centre.
    order : int
        Interpolation order passed to `scipy.ndimage.affine_transform`.
    cval : float
        Fill value for points outside the boundaries (defaults to NaN).

    Returns
    -------
    ndarray
        Resampled image aligned to `ref_shape`.
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
        offset=cast(float,offset),
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




