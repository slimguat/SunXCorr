import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sunpy.map import Map
from sunxcorr import Orchestrator, SingleMapProcess
import astropy.units as u


def _pick_maps():
    base = Path(__file__).resolve().parents[1] / "fits_files"
    fits = list(base.rglob("*.fits"))
    if not fits:
        pytest.skip("No FITS files available in fits_files/; skip tests")
    # choose two maps (or same if only one)
    if len(fits) == 1:
        return Map(fits[0]), Map(fits[0])
    return Map(fits[0]), Map(fits[1])


def test_single_map_minimal_run():
    target_map, reference_map = _pick_maps()
    root = Orchestrator(n_workers=1)
    root.node_id = "test_root"
    root.node_name = "Test Root"
    root.base_target_map = target_map
    root.reference_map = reference_map
    root.verbose = 0
    proc = SingleMapProcess(max_shift=5.0 * u.arcsec, bin_kernel=0.0 * u.arcsec, n_neighbors=4)
    root.add_child(proc)

    # run pipeline
    root.execute()

    assert proc.result is not None
    # correlation value within [-1,1]
    assert -1.0 <= proc.result.correlation_peak <= 1.0
