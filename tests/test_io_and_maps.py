import sys
from pathlib import Path
import pytest

# Ensure package import from workspace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sunpy.map import Map
from sunxcorr.utils import _vprint


def _find_two_fits():
    base = Path(__file__).resolve().parents[1] / "fits_files"
    fits = list(base.rglob("*.fits"))
    if not fits:
        pytest.skip("No FITS files available in fits_files/; skip tests")
    if len(fits) == 1:
        return fits[0], fits[0]
    return fits[0], fits[1]


def test_load_map_and_metadata():
    t, r = _find_two_fits()
    m = Map(t)
    assert hasattr(m, 'data')
    assert m.data.size > 0
    # basic header items exist
    hdr = m.meta
    assert 'date-avg' in hdr or 'date' in hdr or 'DATE' in hdr


def test_two_maps_pixel_scale_and_shapes():
    t, r = _find_two_fits()
    mt = Map(t)
    mr = Map(r)
    # shapes
    assert mt.data.ndim == 2
    assert mr.data.ndim == 2
    # pixel scale retrieval via CRVAL/CRPIX presence
    assert 'crpix1' in mt.meta.keys() or 'CRPIX1' in mt.meta.keys() or 'crpix1' in map(str.lower, mt.meta.keys())
    assert 'crpix1' in mr.meta.keys() or 'CRPIX1' in mr.meta.keys() or 'crpix1' in map(str.lower, mr.meta.keys())
