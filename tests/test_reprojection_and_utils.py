import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sunpy.map import Map  # noqa: E402

from sunxcorr.utils import reproject_map_to_reference  # noqa: E402


def _find_two_fits():
    base = Path(__file__).resolve().parents[1] / "fits_files"
    fits = list(base.rglob("*.fits"))
    if not fits:
        pytest.skip("No FITS files available in fits_files/; skip tests")
    if len(fits) == 1:
        return fits[0], fits[0]
    return fits[0], fits[1]


def test_reproject_map_to_reference_basic():
    t, r = _find_two_fits()
    target = Map(t)
    reference = Map(r)
    reprojected = reproject_map_to_reference(target, reference)
    data = np.asarray(reprojected.data)
    finite_frac = np.isfinite(data).sum() / data.size
    # Allow lower coverage for some instrument/WCS combinations; require at least
    # a small fraction of mapped pixels or at least some finite pixels.
    assert (
        finite_frac > 0.03 and np.isfinite(data).sum() > 100
    ), f"Too few reprojected pixels: frac={finite_frac:.4f}, count={np.isfinite(data).sum()}"
    # statistics shouldn't be NaN
    assert np.isfinite(np.nanmean(data))
    assert np.isfinite(np.nanstd(data))
