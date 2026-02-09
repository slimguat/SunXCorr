
"""Validation and timing tests for a faster correlation path.

Run: python test_correlation_fast.py
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform

from slimfunc_correlation_effort import (
    normalized_corr_nan_safe,
    squeeze_to_ref_grid,
    shift_image,
    correlation_for_params,
)
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from numba import njit,jit


@dataclass
class CorrContextFast:
    ref: np.ndarray
    tgt: np.ndarray
    valid_ref: np.ndarray
    original_valid_count: int


def build_corr_context_fast(ref_img: np.ndarray, target_img: np.ndarray) -> CorrContextFast:
    ref = np.asarray(ref_img, dtype=np.float64)
    tgt = np.asarray(target_img, dtype=np.float64)
    valid_ref = np.isfinite(ref)
    original_valid_count = int((valid_ref & np.isfinite(tgt)).sum())
    return CorrContextFast(ref=ref, tgt=tgt, valid_ref=valid_ref, original_valid_count=original_valid_count)


@jit(cache=True, fastmath=False)
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
def _corr_fused_scale_shift(ref, tgt, valid_ref, dx_i, dy_i, inv_sx, inv_sy, off_x, off_y, original_valid_count):
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

    # ratio penalty (your current logic)
    if original_valid_count > 0:
        ratio = n / original_valid_count
        if ratio > 1.0:
            ratio = 1.0
        if ratio > 0.5:
            ratio = 1.0
        corr *= ratio

    return corr


def correlation_for_params_fastest(
    ctx: CorrContextFast,
    dx: float,
    dy: float,
    squeeze_x: float,
    squeeze_y: float,
    center: Optional[Tuple[float, float]] = None,
) -> float:
    if (not np.isfinite(squeeze_x)) or (not np.isfinite(squeeze_y)):
        return 0.0
    if squeeze_x <= 0.0 or squeeze_y <= 0.0:
        return 0.0

    dx_i = int(np.round(dx))
    dy_i = int(np.round(dy))

    if center is None:
        ny, nx = ctx.tgt.shape
        cy = (ny - 1) / 2.0
        cx = (nx - 1) / 2.0
    else:
        cy, cx = float(center[0]), float(center[1])

    inv_sy = 1.0 / float(squeeze_y)
    inv_sx = 1.0 / float(squeeze_x)
    off_y = cy - inv_sy * cy
    off_x = cx - inv_sx * cx

    return _corr_fused_scale_shift(
        ctx.ref, ctx.tgt, ctx.valid_ref,
        dx_i, dy_i,
        inv_sx, inv_sy,
        off_x, off_y,
        ctx.original_valid_count
    )



def test_correlation_fastest_equivalence(tol: float = 1e-3) -> None:
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((128, 96))
    tgt = rng.standard_normal((128, 96))

    ctx = build_corr_context_fast(ref, tgt)

    cases = [
        (0.0, 0.0, 1.0, 1.0),
        (3.4, -2.2, 1.0, 1.0),
        (0.0, 0.0, 0.95, 1.05),
        (5.1, -4.7, 0.9, 1.1),
    ]

    # Warm-up JIT (very important for timing)
    _ = correlation_for_params_fastest(ctx, 0.0, 0.0, 1.0, 1.0)

    for dx, dy, sx, sy in cases:
        print("CASE", dx, dy, sx, sy)
        c_new = correlation_for_params_fastest(ctx, dx, dy, sx, sy)
        c_old = correlation_for_params(ref, tgt, dx, dy, sx, sy)
        if not np.isclose(c_old, c_new, atol=tol):
            raise AssertionError(
                f"FASTEST mismatch (dx,dy,sx,sy)=({dx},{dy},{sx},{sy}): {c_old} vs {c_new}"
            )

import time

def benchmark_fastest(n_iters: int = 1) -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((1048, 192))
    tgt = rng.standard_normal((1048, 192))

    ctx_fast = build_corr_context_fast(ref, tgt)

    params = (5.1, -4.7, 0.9, 1.1)

    # warmup
    _ = correlation_for_params_fastest(ctx_fast, *params)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = correlation_for_params_fastest(ctx_fast, *params)
    t1 = time.perf_counter()
    print(f"FASTEST fused: {(t1 - t0):.6e} s")

def benchmark_old_vs_new(n_iters: int = 200) -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((1048, 192))
    tgt = rng.standard_normal((1048, 192))

    params = (5.1, -4.7, 0.9, 1.1)

    # Build context for new method
    ctx_fast = build_corr_context_fast(ref, tgt)

    # -----------------------
    # Warmups (very important)
    # -----------------------
    # Warm up NEW (Numba compile happens here)
    _ = correlation_for_params_fastest(ctx_fast, *params)

    # Warm up OLD (SciPy will have its own internal warmups/caches)
    _ = correlation_for_params(ref, tgt, *params)

    # -----------------------
    # Benchmark OLD
    # -----------------------
    t0 = time.perf_counter()
    acc_old = 0.0
    for _ in range(n_iters):
        acc_old += correlation_for_params(ref, tgt, *params)
    t1 = time.perf_counter()

    # -----------------------
    # Benchmark NEW
    # -----------------------
    t2 = time.perf_counter()
    acc_new = 0.0
    for _ in range(n_iters):
        acc_new += correlation_for_params_fastest(ctx_fast, *params)
    t3 = time.perf_counter()

    old_total = t1 - t0
    new_total = t3 - t2

    print("=== Benchmark (old vs new) ===")
    print(f"shape: {ref.shape}, n_iters: {n_iters}")
    print(f"OLD total: {old_total:.6e} s  | per iter: {old_total/n_iters:.6e} s")
    print(f"NEW total: {new_total:.6e} s  | per iter: {new_total/n_iters:.6e} s")
    print(f"Speedup:   {old_total / new_total:.2f}×")
    print(f"(ignore) acc_old={acc_old:.6e}, acc_new={acc_new:.6e}")

if __name__ == "__main__":
    test_correlation_fastest_equivalence()
    print("Equivalence test passed.")
    benchmark_fastest()
    benchmark_old_vs_new()