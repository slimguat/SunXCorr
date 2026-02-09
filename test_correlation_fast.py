
"""Validation and timing tests for a faster correlation path.

Run: python test_correlation_fast.py
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit
from scipy.ndimage import affine_transform

from slimfunc_correlation_effort import (
    normalized_corr_nan_safe,
    squeeze_to_ref_grid,
    shift_image,
    correlation_for_params,
)


# =========================
# New (fast) helper versions
# =========================

@dataclass
class CorrContext:
    ref: np.ndarray
    tgt: np.ndarray
    valid_ref: np.ndarray
    original_valid: np.ndarray
    original_valid_count: int
    work_shifted: np.ndarray


def build_corr_context(ref_img: np.ndarray, target_img: np.ndarray) -> CorrContext:
    ref = np.asarray(ref_img, dtype=np.float64)
    tgt = np.asarray(target_img, dtype=np.float64)
    valid_ref = np.isfinite(ref)
    original_valid = valid_ref & np.isfinite(tgt)
    original_valid_count = int(original_valid.sum())
    work_shifted = np.full_like(ref, np.nan)
    return CorrContext(ref, tgt, valid_ref, original_valid, original_valid_count, work_shifted)


def squeeze_to_ref_grid_fast(
    target_img: np.ndarray,
    ref_shape: Tuple[int, int],
    squeeze_x: float = 1.0,
    squeeze_y: float = 1.0,
    center: Optional[Tuple[float, float]] = None,
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    target_img = np.asarray(target_img, dtype=np.float64)
    ny, nx = target_img.shape

    if center is None:
        cy = (ny - 1) / 2.0
        cx = (nx - 1) / 2.0
    else:
        cy, cx = center

    A = np.array(
        [[1.0 / squeeze_y, 0.0], [0.0, 1.0 / squeeze_x]],
        dtype=np.float64,
    )

    center_vec = np.array([cy, cx], dtype=np.float64)
    offset = center_vec - A @ center_vec

    return affine_transform(
        target_img,
        A,
        offset=offset,
        output_shape=ref_shape,
        order=order,
        cval=cval,
    )


@njit
def shift_image_jit(img, dx, dy, out):
    out.fill(np.nan)
    ny, nx = img.shape
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
    if ys >= ye or xs >= xe or yd >= ye2 or xd >= xe2:
        return out
    out[yd:ye2, xd:xe2] = img[ys:ye, xs:xe]
    return out

def shift_image_fast(img: np.ndarray, dx: int, dy: int, out: np.ndarray) -> np.ndarray:
    out.fill(np.nan)
    ny, nx = img.shape
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
    if ys >= ye or xs >= xe or yd >= ye2 or xd >= xe2:
        return out
    out[yd:ye2, xd:xe2] = img[ys:ye, xs:xe]
    return out


@njit
def normalized_corr_masked_jit(ref, shifted, valid_ref):
    valid = (valid_ref & np.isfinite(shifted)).ravel()
    ref_flat = ref.ravel()
    shifted_flat = shifted.ravel()
    if not np.any(valid):
        return 0.0
    a = ref_flat[valid]
    b = shifted_flat[valid]
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    num = np.dot(a_c, b_c)
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    if denom == 0.0:
        return 0.0
    return num / denom
    # Apply mask after squeeze and shift, match original logic
    mask = valid_ref & np.isfinite(shifted)
    if not np.any(mask):
        return 0.0
    a = ref[mask]
    b = shifted[mask]
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    num = np.dot(a_c, b_c)
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    if denom == 0.0:
        return 0.0
    return num / denom

def normalized_corr_masked(ref: np.ndarray, shifted: np.ndarray, valid_ref: np.ndarray) -> float:
    valid = valid_ref & np.isfinite(shifted)
    if not np.any(valid):
        return 0.0
    a = ref[valid]
    b = shifted[valid]
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = np.dot(a_c, b_c)
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    if denom == 0.0:
        return 0.0
    return num / denom


def correlation_for_params_fast(
    ctx: CorrContext,
    dx: float,
    dy: float,
    squeeze_x: float,
    squeeze_y: float,
    center: Optional[Tuple[float, float]] = None,
) -> float:
    if squeeze_x == 1.0 and squeeze_y == 1.0:
        squeezed = ctx.tgt
    else:
        squeezed = squeeze_to_ref_grid_fast(
            ctx.tgt,
            ref_shape=ctx.ref.shape,
            squeeze_x=squeeze_x,
            squeeze_y=squeeze_y,
            order=1,
            cval=np.nan,
            center=center,
        )

    shifted = shift_image_fast(squeezed, int(round(dx)), int(round(dy)), ctx.work_shifted)

    corr = normalized_corr_masked(ctx.ref, shifted, ctx.valid_ref)

    if ctx.original_valid_count > 0:
        warped_valid = ctx.valid_ref & np.isfinite(shifted)
        ratio = warped_valid.sum() / ctx.original_valid_count
        if ratio > 1.0:
            ratio = 1.0
        if ratio > 0.5:
            ratio = 1.0
        corr *= ratio

    return corr


# =========================
# Tests
# =========================


def assert_close(a: np.ndarray, b: np.ndarray, tol: float, name: str) -> None:
    if not np.allclose(a, b, atol=tol, equal_nan=True):
        diff = np.nanmax(np.abs(a - b))
        raise AssertionError(f"{name} failed: max diff={diff}")


def test_shift_and_squeeze(tol: float = 1e-10) -> None:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((128, 96))

    # Shift-only
    dx, dy = 7, -9
    out_fast = np.full_like(img, np.nan)
    shifted_fast = shift_image_fast(img, dx, dy, out_fast)
    shifted_old = shift_image(img, dx, dy)
    # JIT warmup
    out_jit = np.full_like(img, np.nan)
    shift_image_jit(img, dx, dy, out_jit)
    shifted_jit = shift_image_jit(img, dx, dy, out_jit)
    assert_close(shifted_fast, shifted_old, tol, "shift_image")
    assert_close(shifted_jit, shifted_old, tol, "shift_image_jit")

    # Squeeze-only
    sx, sy = 0.92, 1.08
    squeezed_fast = squeeze_to_ref_grid_fast(img, img.shape, sx, sy)
    squeezed_old = squeeze_to_ref_grid(img, img.shape, sx, sy)
    assert_close(squeezed_fast, squeezed_old, tol, "squeeze_to_ref_grid")

    # Squeeze + shift
    shifted_fast = shift_image_fast(squeezed_fast, dx, dy, out_fast)
    shifted_old = shift_image(squeezed_old, dx, dy)
    shifted_jit = shift_image_jit(squeezed_fast, dx, dy, out_jit)
    assert_close(shifted_fast, shifted_old, tol, "squeeze+shift")
    assert_close(shifted_jit, shifted_old, tol, "squeeze+shift_jit")


def test_correlation_equivalence(tol: float = 1e-2) -> None:
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((128, 96))
    tgt = rng.standard_normal((128, 96))
    ctx = build_corr_context(ref, tgt)

    cases = [
        (0.0, 0.0, 1.0, 1.0),
        (3.4, -2.2, 1.0, 1.0),
        (0.0, 0.0, 0.95, 1.05),
        (5.1, -4.7, 0.9, 1.1),
    ]

    # JIT warmup
    normalized_corr_masked_jit(ref, tgt, np.isfinite(ref))
    for dx, dy, sx, sy in cases:
        squeezed_old = squeeze_to_ref_grid(tgt, ref.shape, sx, sy)
        squeezed_fast = squeeze_to_ref_grid_fast(tgt, ref.shape, sx, sy)
        squeezed_jit = squeeze_to_ref_grid_fast(tgt, ref.shape, sx, sy)
        shifted_old = shift_image(squeezed_old, int(round(dx)), int(round(dy)))
        shifted_fast = shift_image_fast(squeezed_fast, int(round(dx)), int(round(dy)), np.full_like(ref, np.nan))
        shifted_jit = shift_image_jit(squeezed_jit, int(round(dx)), int(round(dy)), np.full_like(ref, np.nan))
        valid_mask_old = np.isfinite(ref) & np.isfinite(squeezed_old)
        valid_mask_fast = np.isfinite(ref) & np.isfinite(squeezed_fast)
        valid_mask_jit = np.isfinite(ref) & np.isfinite(squeezed_jit)
        a_old = ref[valid_mask_old]
        b_old = shifted_old[valid_mask_old]
        a_fast = ref[valid_mask_fast]
        b_fast = shifted_fast[valid_mask_fast]
        a_jit = ref[valid_mask_jit]
        b_jit = shifted_jit[valid_mask_jit]
        # Use the same mask and correlation function for JIT as the original
        c_old = normalized_corr_nan_safe(a_old, b_old)
        c_fast = normalized_corr_nan_safe(a_fast, b_fast)
        c_jit = normalized_corr_nan_safe(a_jit, b_jit)
        # Print detailed debug info
        print(f"DEBUG: dx={dx}, dy={dy}, sx={sx}, sy={sy}")
        print(f"c_old={c_old}, c_fast={c_fast}, c_jit={c_jit}")
        print(f"a_old[:5]={a_old[:5]}, b_old[:5]={b_old[:5]}")
        print(f"a_fast[:5]={a_fast[:5]}, b_fast[:5]={b_fast[:5]}")
        print(f"a_jit[:5]={a_jit[:5]}, b_jit[:5]={b_jit[:5]}")
        print(f"valid pixels (old): {len(a_old)}, (fast): {len(a_fast)}, (jit): {len(a_jit)}")
        print(f"shifted_old stats: min={np.nanmin(shifted_old)}, max={np.nanmax(shifted_old)}, mean={np.nanmean(shifted_old)}")
        print(f"shifted_fast stats: min={np.nanmin(shifted_fast)}, max={np.nanmax(shifted_fast)}, mean={np.nanmean(shifted_fast)}")
        print(f"shifted_jit stats: min={np.nanmin(shifted_jit)}, max={np.nanmax(shifted_jit)}, mean={np.nanmean(shifted_jit)}")
        print(f"squeezed_old stats: min={np.nanmin(squeezed_old)}, max={np.nanmax(squeezed_old)}, mean={np.nanmean(squeezed_old)}")
        print(f"squeezed_fast stats: min={np.nanmin(squeezed_fast)}, max={np.nanmax(squeezed_fast)}, mean={np.nanmean(squeezed_fast)}")
        print(f"squeezed_jit stats: min={np.nanmin(squeezed_jit)}, max={np.nanmax(squeezed_jit)}, mean={np.nanmean(squeezed_jit)}")
        if not np.isclose(c_old, c_fast, atol=tol):
            raise AssertionError(
                f"correlation mismatch for FAST (dx,dy,sx,sy)=({dx},{dy},{sx},{sy}): {c_old} vs {c_fast}"
            )
        if not np.isclose(c_old, c_jit, atol=tol):
            raise AssertionError(
                f"correlation mismatch for JIT (dx,dy,sx,sy)=({dx},{dy},{sx},{sy}): {c_old} vs {c_jit}"
            )
        c_old = correlation_for_params(ref, tgt, dx, dy, sx, sy)
        c_new = correlation_for_params_fast(ctx, dx, dy, sx, sy)
        if not np.isclose(c_old, c_new, atol=tol):
            print(f"DEBUG: dx={dx}, dy={dy}, sx={sx}, sy={sy}")
            print(f"c_old={c_old}, c_new={c_new}")
            raise AssertionError(
                f"correlation mismatch for (dx,dy,sx,sy)=({dx},{dy},{sx},{sy}): {c_old} vs {c_new}"
            )
        if not np.isclose(c_old, c_jit, atol=tol):
            print(f"DEBUG: dx={dx}, dy={dy}, sx={sx}, sy={sy}")
            print(f"c_old={c_old}, c_jit={c_jit}, atol={tol}")
            print(f"shifted_jit stats: min={np.nanmin(shifted_jit)}, max={np.nanmax(shifted_jit)}, mean={np.nanmean(shifted_jit)}")
            print(f"ref stats: min={np.nanmin(ref)}, max={np.nanmax(ref)}, mean={np.nanmean(ref)}")
            # Print valid pixel counts
            valid_mask = np.isfinite(ref) & np.isfinite(shifted_jit)
            print(f"valid pixels (JIT): {np.sum(valid_mask)}")
            valid_mask_orig = np.isfinite(ref) & np.isfinite(squeeze_to_ref_grid_fast(tgt, ref.shape, sx, sy))
            print(f"valid pixels (orig): {np.sum(valid_mask_orig)}")
            # Print arrays used in correlation
            print(f"JIT a: {ref.ravel()[valid_mask.ravel()][:5]}")
            print(f"JIT b: {shifted_jit.ravel()[valid_mask.ravel()][:5]}")
            squeezed_orig = squeeze_to_ref_grid_fast(tgt, ref.shape, sx, sy)
            print(f"Orig a: {ref[valid_mask_orig][:5]}")
            print(f"Orig b: {squeezed_orig[valid_mask_orig][:5]}")
            raise AssertionError(
                f"correlation mismatch for JIT (dx,dy,sx,sy)=({dx},{dy},{sx},{sy}): {c_old} vs {c_jit}"
            )

    # Edge case: all NaN
    ref_nan = np.full((800, 200), np.nan)
    tgt_nan = np.full((800, 200), np.nan)
    ctx_nan = build_corr_context(ref_nan, tgt_nan)
    c_old = correlation_for_params(ref_nan, tgt_nan, 0, 0, 1.0, 1.0)
    c_new = correlation_for_params_fast(ctx_nan, 0, 0, 1.0, 1.0)
    shifted_jit = shift_image_jit(tgt_nan, 0, 0, np.full_like(ref_nan, np.nan))
    c_jit = normalized_corr_masked_jit(ref_nan, shifted_jit, np.isfinite(ref_nan))
    if not np.isclose(c_old, c_new, atol=tol):
        raise AssertionError("correlation mismatch for NaN case")
    if not np.isclose(c_old, c_jit, atol=tol):
        raise AssertionError("correlation mismatch for NaN case (JIT)")
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((128, 96))
    tgt = rng.standard_normal((128, 96))
    ctx = build_corr_context(ref, tgt)

    cases = [
        (0.0, 0.0, 1.0, 1.0),
        (3.4, -2.2, 1.0, 1.0),
        (0.0, 0.0, 0.95, 1.05),
        (5.1, -4.7, 0.9, 1.1),
    ]

    for dx, dy, sx, sy in cases:
        c_old = correlation_for_params(ref, tgt, dx, dy, sx, sy)
        c_new = correlation_for_params_fast(ctx, dx, dy, sx, sy)
        if not np.isclose(c_old, c_new, atol=tol):
            raise AssertionError(
                f"correlation mismatch for (dx,dy,sx,sy)=({dx},{dy},{sx},{sy}): {c_old} vs {c_new}"
            )

    # Edge case: all NaN
    ref_nan = np.full((800, 200), np.nan)
    tgt_nan = np.full((800, 200), np.nan)
    ctx_nan = build_corr_context(ref_nan, tgt_nan)
    c_old = correlation_for_params(ref_nan, tgt_nan, 0, 0, 1.0, 1.0)
    c_new = correlation_for_params_fast(ctx_nan, 0, 0, 1.0, 1.0)
    if not np.isclose(c_old, c_new, atol=tol):
        raise AssertionError("correlation mismatch for NaN case")


def benchmark(n_iters: int = 200, tol=10e-2) -> None:
    img = np.random.rand(256, 192)
    dx, dy = 5, 7
    out = np.full_like(img, np.nan)
    # Warmup
    shift_image_jit(img, dx, dy, out)
    t0 = time.time()
    for _ in range(10):
        shift_image(img, dx, dy)
    t1 = time.time()
    for _ in range(10):
        shift_image_fast(img, dx, dy, out)
    t2 = time.time()
    for _ in range(10):
        shift_image_jit(img, dx, dy, out)
    t3 = time.time()
    print(f"shift_image (old): {(t1-t0)/10:.6f} s")
    print(f"shift_image_fast: {(t2-t1)/10:.6f} s")
    print(f"shift_image_jit: {(t3-t2)/10:.6f} s")

    ref = np.random.rand(256, 192)
    tgt = np.random.rand(256, 192)
    valid_ref = np.isfinite(ref)
    normalized_corr_masked_jit(ref, tgt, valid_ref)
    t0 = time.time()
    for _ in range(10):
        normalized_corr_masked(ref, tgt, valid_ref)
    t1 = time.time()
    for _ in range(10):
        normalized_corr_masked_jit(ref, tgt, valid_ref)
    t2 = time.time()
    print(f"normalized_corr_masked (old): {(t1-t0)/10:.6f} s")
    print(f"normalized_corr_masked_jit: {(t2-t1)/10:.6f} s")

    print("Running shift/squeeze tests...")
    test_shift_and_squeeze(tol=tol)
    print("Shift/squeeze tests passed.")

    print("Running correlation equivalence tests...")
    test_correlation_equivalence(tol=tol)
    print("Correlation equivalence tests passed.")

    print("Running benchmark...")
    benchmark(n_iters=200)

    print("\nIndividual speed tests:")
    img = np.random.rand(256, 192)
    dx, dy = 5, 7
    out = np.full_like(img, np.nan)
    # Warmup
    shift_image_jit(img, dx, dy, out)
    t0 = time.time()
    for _ in range(10):
        shift_image(img, dx, dy)
    t1 = time.time()
    for _ in range(10):
        shift_image_fast(img, dx, dy, out)
    t2 = time.time()
    for _ in range(10):
        shift_image_jit(img, dx, dy, out)
    t3 = time.time()
    print(f"shift_image (old): {(t1-t0)/10:.6f} s")
    print(f"shift_image_fast: {(t2-t1)/10:.6f} s")
    print(f"shift_image_jit: {(t3-t2)/10:.6f} s")

    ref = np.random.rand(256, 192)
    tgt = np.random.rand(256, 192)
    valid_ref = np.isfinite(ref)
    normalized_corr_masked_jit(ref, tgt, valid_ref)
    t0 = time.time()
    for _ in range(10):
        normalized_corr_masked(ref, tgt, valid_ref)
    t1 = time.time()
    for _ in range(10):
        normalized_corr_masked_jit(ref, tgt, valid_ref)
    t2 = time.time()
    print(f"normalized_corr_masked (old): {(t1-t0)/10:.6f} s")
    print(f"normalized_corr_masked_jit: {(t2-t1)/10:.6f} s")
    # Warmup
    shift_image_jit(img, dx, dy, out)
    t0 = time.time()
    for _ in range(10):
        shift_image(img, dx, dy)
    t1 = time.time()
    for _ in range(10):
        shift_image_fast(img, dx, dy, out)
    t2 = time.time()
    for _ in range(10):
        shift_image_jit(img, dx, dy, out)
    t3 = time.time()
    print(f"shift_image (old): {(t1-t0)/10:.6f} s")
    print(f"shift_image_fast: {(t2-t1)/10:.6f} s")
    print(f"shift_image_jit: {(t3-t2)/10:.6f} s")

    ref = np.random.rand(256, 192)
    tgt = np.random.rand(256, 192)
    valid_ref = np.isfinite(ref)
    normalized_corr_masked_jit(ref, tgt, valid_ref)
    t0 = time.time()
    for _ in range(10):
        normalized_corr_masked(ref, tgt, valid_ref)
    t1 = time.time()
    for _ in range(10):
        normalized_corr_masked_jit(ref, tgt, valid_ref)
    t2 = time.time()
    print(f"normalized_corr_masked (old): {(t1-t0)/10:.6f} s")
    print(f"normalized_corr_masked_jit: {(t2-t1)/10:.6f} s")

    tol = 1e-10
    print("Running shift/squeeze tests...")
    test_shift_and_squeeze(tol=tol)
    print("Shift/squeeze tests passed.")

    print("Running correlation equivalence tests...")
    test_correlation_equivalence(tol=tol)
    print("Correlation equivalence tests passed.")

    print("Running benchmark...")
    benchmark(n_iters=200)

def main():
    tol = 10**-1
    print("Running shift/squeeze tests...")
    test_shift_and_squeeze(tol=tol)
    print("Shift/squeeze tests passed.")

    print("Running correlation equivalence tests...")
    test_correlation_equivalence(tol=tol)
    print("Correlation equivalence tests passed.")

    print("Running benchmark...")
    benchmark(n_iters=200,tol=tol)

if __name__ == "__main__":
    main()
