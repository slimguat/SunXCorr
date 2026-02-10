import os
import tempfile
from pathlib import Path

from sunxcorr.synthetic_raster_process import SyntheticRasterProcess
from sunxcorr.single_map_process import SingleMapProcess


def test_synthetic_raster_process_short_sequence(tmp_path=None):
    """Run a short SyntheticRasterProcess on a small set of FSI files.

    This test locates up to 4 FSI files in `fits_files/FSI` within the project,
    builds a minimal synthetic raster sequence, runs the pipeline in a
    light configuration (shift-only, small neighbors) and asserts that the
    process completes and returns expected keys.
    """
    repo_root = Path(__file__).resolve().parents[1]
    fsi_dir = repo_root / "sunxcorr" / "fits_files" / "FSI"
    if not fsi_dir.exists():
        # Try alternate path at repo root
        fsi_dir = repo_root / "fits_files" / "FSI"

    files = sorted([p for p in fsi_dir.iterdir() if p.suffix.lower() in (".fits", ".fit")])[:4]
    if len(files) < 2:
        # Not enough sample data to run this test in this environment; skip.
        return

    if tmp_path is None:
        td = tempfile.TemporaryDirectory()
        outdir = Path(td.name) / "sr_out"
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path(tmp_path) / "sr_out"
        outdir.mkdir(parents=True, exist_ok=True)

    # Build maps for sequence
    from sunpy.map import Map
    from sunxcorr import Orchestrator

    seq_maps = [Map(p) for p in files]

    # Use first map as base target / working_map
    base_map = seq_maps[0]

    # Create orchestrator/root node to host resources
    root = Orchestrator(n_workers=1)
    root.node_id = "test_root"
    root.node_name = "Test Root"
    root.base_target_map = base_map
    root.reference_sequence = seq_maps
    root.verbose = 0

    import astropy.units as u

    # Configure a minimal synthetic raster process (shift-only)
    sr = SyntheticRasterProcess(max_shift=15.0 * u.arcsec, scale_step=0.0, n_neighbors=8)
    root.add_child(sr)

    # Execute pipeline
    root.execute()

    assert sr.result is not None
    # Ensure shift-like outputs exist in ProcessResult
    assert hasattr(sr.result, 'shift_arcsec') or hasattr(sr.result, 'shift_pixels')
