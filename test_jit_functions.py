import numpy as np
from scipy.ndimage import affine_transform
from numba import njit

# --- Original functions ---
def normalized_corr_nan_safe(img1: np.ndarray, img2: np.ndarray) -> float:
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

def squeeze_to_ref_grid(target_img, ref_shape, squeeze_x=1.0, squeeze_y=1.0, center=None, order=1, cval=np.nan):
    target_img = np.asarray(target_img, dtype=np.float64)
    ny, nx = target_img.shape
    if center is None:
        cy = (ny - 1) / 2.0
        cx = (nx - 1) / 2.0
    else:
        cy, cx = center
    A = np.array([[1.0 / squeeze_y, 0.0], [0.0, 1.0 / squeeze_x]], dtype=np.float64)
    center_vec = np.array([cy, cx], dtype=np.float64)
    offset = center_vec - A @ center_vec
    squeezed = affine_transform(target_img, A, offset=offset, output_shape=ref_shape, order=order, cval=cval)
    return squeezed

def shift_image(img, dx, dy):
    ny, nx = img.shape
    out = np.full_like(img, np.nan)
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
        return np.full_like(img, np.nan)
    out[yd:ye2, xd:xe2] = img[ys:ye, xs:xe]
    return out

# --- Numba JIT versions ---
@njit
def shift_image_jit(img, dx, dy):
    ny, nx = img.shape
    out = np.full(img.shape, np.nan)
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
def normalized_corr_nan_safe_jit(img1, img2):
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    valid = np.isfinite(img1_flat) & np.isfinite(img2_flat)
    if not np.any(valid):
        return 0.0
    a = img1_flat[valid]
    b = img2_flat[valid]
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    num = np.dot(a_c, b_c)
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    if denom == 0.0:
        return 0.0
    return num / denom

# squeeze_to_ref_grid cannot be fully JITed due to affine_transform (scipy), but we can JIT the offset calculation
@njit
def squeeze_offset_jit(ny, nx, squeeze_x, squeeze_y, cy, cx):
    A = np.array([[1.0 / squeeze_y, 0.0], [0.0, 1.0 / squeeze_x]])
    center_vec = np.array([cy, cx])
    offset = center_vec - A @ center_vec
    return offset

# --- Test routines ---
def test_shift():
    img = np.arange(16).reshape(4,4).astype(np.float64)
    dx, dy = 1, 2
    shifted = shift_image(img, dx, dy)
    shifted_jit = shift_image_jit(img, dx, dy)
    assert np.allclose(shifted, shifted_jit, equal_nan=True)
    print("shift_image: PASS")

def test_corr():
    img1 = np.random.rand(10,10)
    img2 = np.random.rand(10,10)
    corr = normalized_corr_nan_safe(img1, img2)
    corr_jit = normalized_corr_nan_safe_jit(img1, img2)
    assert np.isclose(corr, corr_jit)
    print("normalized_corr_nan_safe: PASS")

def test_squeeze():
    img = np.random.rand(10,10)
    ref_shape = (10,10)
    squeeze_x, squeeze_y = 1.2, 0.8
    squeezed = squeeze_to_ref_grid(img, ref_shape, squeeze_x, squeeze_y)
    ny, nx = img.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    offset = squeeze_offset_jit(ny, nx, squeeze_x, squeeze_y, cy, cx)
    # Compare offset calculation
    A = np.array([[1.0 / squeeze_y, 0.0], [0.0, 1.0 / squeeze_x]])
    center_vec = np.array([cy, cx])
    offset_np = center_vec - A @ center_vec
    assert np.allclose(offset, offset_np)
    print("squeeze_to_ref_grid offset: PASS")

if __name__ == "__main__":
    import time
    test_shift()
    test_corr()
    test_squeeze()
    print("All tests passed.")

    # Timing benchmarks
    print("\nTiming benchmarks:")
    img = np.random.rand(1000, 1000)
    dx, dy = 5, 7
    # Warmup for JIT
    shift_image_jit(img, dx, dy)
    t0 = time.time()
    for _ in range(10):
        shift_image(img, dx, dy)
    t1 = time.time()
    for _ in range(10):
        shift_image_jit(img, dx, dy)
    t2 = time.time()
    print(f"shift_image (original): {(t1-t0)/10:.6f} s")
    print(f"shift_image_jit: {(t2-t1)/10:.6f} s")

    img1 = np.random.rand(1000, 1000)
    img2 = np.random.rand(1000, 1000)
    normalized_corr_nan_safe_jit(img1, img2)
    t0 = time.time()
    for _ in range(10):
        normalized_corr_nan_safe(img1, img2)
    t1 = time.time()
    for _ in range(10):
        normalized_corr_nan_safe_jit(img1, img2)
    t2 = time.time()
    print(f"normalized_corr_nan_safe (original): {(t1-t0)/10:.6f} s")
    print(f"normalized_corr_nan_safe_jit: {(t2-t1)/10:.6f} s")

    # Squeeze offset timing
    ny, nx = img.shape
    squeeze_x, squeeze_y = 1.2, 0.8
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    squeeze_offset_jit(ny, nx, squeeze_x, squeeze_y, cy, cx)
    t0 = time.time()
    for _ in range(10000):
        A = np.array([[1.0 / squeeze_y, 0.0], [0.0, 1.0 / squeeze_x]])
        center_vec = np.array([cy, cx])
        offset_np = center_vec - A @ center_vec
    t1 = time.time()
    for _ in range(10000):
        squeeze_offset_jit(ny, nx, squeeze_x, squeeze_y, cy, cx)
    t2 = time.time()
    print(f"squeeze offset (original): {(t1-t0)/10000:.8f} s")
    print(f"squeeze_offset_jit: {(t2-t1)/10000:.8f} s")
