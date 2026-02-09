#!/usr/bin/env python3
"""Test script for new coalignment framework."""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool

import astropy.units as u
from sunpy.map import Map

from coalignment import Orchestrator, SingleMapProcess, SyntheticRasterProcess
from help_funcs import get_EUI_paths


def test_binning_only(target_path, reference_path, verbose=2):
    """Test binning process only."""
    print("\n" + "="*70)
    print("TEST 1: Binning Process Only")
    print("="*70)
    
    # Load maps
    target_map = Map(target_path)
    reference_map = Map(reference_path)
    
    # Setup debug output if verbose >= 3
    debug_writer = None
    output_dir = None
    if verbose >= 3:
        from matplotlib.backends.backend_pdf import PdfPages
        output_dir = Path("data_storage/output_figures_2")
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_pdf = output_dir / "debug_binning_only.pdf"
        debug_writer = PdfPages(debug_pdf)
        print(f"[Debug] PDF output: {debug_pdf}")
    
    # Create root node (workers setup automatically)
    root = Orchestrator(n_workers=48)
    root.node_id = "root"
    root.node_name = "Root Orchestrator"
    root.base_target_map = target_map
    root.reference_map = reference_map
    root.verbose = verbose
    root.debug_writer = debug_writer
    root.output_directory = output_dir
    
    # Add binning process
    binning = SingleMapProcess(
        max_shift=700.0 * u.arcsec,
        bin_kernel=50.0 * u.arcsec,  # Same as old bin_kernel_arcsec
        n_neighbors=48
    )
    root.add_child(binning)
    
    # Execute
    root.execute()
    
    # Cleanup workers
    root.cleanup()
    
    # Print results
    result = root.get_final_result()
    print(f"\n[Result] {result.process_name}:")
    print(f"  Shift: ({result.shift_arcsec[0]:.2f}, {result.shift_arcsec[1]:.2f})")
    print(f"  Correlation: {result.correlation_peak:.5f}")
    print(f"  Time: {result.execution_time:.1f}s")
    
    return root


def test_full_resolution_only(target_path, reference_path, verbose=2):
    """Test full resolution process only."""
    print("\n" + "="*70)
    print("TEST 2: Full Resolution Process Only")
    print("="*70)
    
    # Load maps
    target_map = Map(target_path)
    reference_map = Map(reference_path)
    
    # Create root node (workers setup automatically)
    root = Orchestrator(n_workers=48)
    root.node_id = "root"
    root.node_name = "Root Orchestrator"
    root.base_target_map = target_map
    root.reference_map = reference_map
    root.verbose = verbose
    
    # Add full resolution process
    full_res = SingleMapProcess(
        max_shift=700.0 * u.arcsec,
        bin_kernel=0.0 * u.arcsec,  # No binning
        n_neighbors=48
    )
    root.add_child(full_res)
    
    # Execute
    root.execute()
    
    # Cleanup workers
    root.cleanup()
    
    # Print results
    result = root.get_final_result()
    print(f"\n[Result] {result.process_name}:")
    print(f"  Shift: ({result.shift_arcsec[0]:.2f}, {result.shift_arcsec[1]:.2f})")
    print(f"  Correlation: {result.correlation_peak:.5f}")
    print(f"  Time: {result.execution_time:.1f}s")
    
    return root


def test_synthetic_raster_only(target_path, reference_path, verbose=2):
    """Test synthetic raster process only."""
    print("\n" + "="*70)
    print("TEST 3: Synthetic Raster Process Only")
    print("="*70)
    
    # Load maps
    target_map = Map(target_path)
    reference_map = Map(reference_path)
    
    # Auto-discover FSI sequence
    import numpy as np
    from help_funcs import _vprint
    
    # date_avg = np.datetime64(target_map.date.isot)
    # half_day = np.timedelta64(12, "h")
    # date_beg = date_avg - half_day
    # date_end = date_avg + half_day
    date_beg = np.datetime64(target_map.meta['date_beg'])+ np.timedelta64(-1, "h")
    date_end = np.datetime64(target_map.meta['date_end'])+ np.timedelta64(-1, "h")
    
    _vprint(verbose, 2, f"Searching for FSI images from {date_beg} to {date_end}")
    
    local_dir = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2")
    if not local_dir.is_dir():
        local_dir = reference_path.parent
    
    fsi_paths = get_EUI_paths(date_beg, date_end, local_dir=local_dir, verbose=verbose)
    
    # Filter for FSI 174
    fsi_174_paths = [p for p in fsi_paths if 'fsi174' in p.name.lower() and 'short' not in p.name.lower()]
    
    _vprint(verbose, 1, f"Found {len(fsi_174_paths)} FSI 174 images")
    
    if not fsi_174_paths:
        print("WARNING: No FSI sequence found, using single reference")
        fsi_sequence = [reference_map]
    else:
        fsi_sequence = fsi_174_paths
    
    # Create root node (workers setup automatically)
    root = Orchestrator(n_workers=48)
    root.node_id = "root"
    root.node_name = "Root Orchestrator"
    root.base_target_map = target_map
    root.reference_sequence = fsi_sequence
    root.verbose = verbose
    
    # Add synthetic raster process
    synthetic = SyntheticRasterProcess(
        max_shift=700.0 * u.arcsec,
        scale_step=0.01,  # 1% steps
        n_neighbors=80
    )
    root.add_child(synthetic)
    
    # Execute
    root.execute()
    
    # Cleanup workers
    root.cleanup()
    
    # Print results
    result = root.get_final_result()
    print(f"\n[Result] {result.process_name}:")
    print(f"  Shift: ({result.shift_arcsec[0]:.2f}, {result.shift_arcsec[1]:.2f})")
    print(f"  Scale: ({result.scale_factors[0]:.4f}, {result.scale_factors[1]:.4f})")
    print(f"  Correlation: {result.correlation_peak:.5f}")
    print(f"  Time: {result.execution_time:.1f}s")
    
    return root


def test_complete_pipeline(target_path, reference_path, verbose=2):
    import numpy as np
    from help_funcs import _vprint,normit
     
    """Test complete 3-stage pipeline."""
    print("\n" + "="*70)
    print("TEST 4: Complete Pipeline (Binning → Full Res → Synthetic)")
    print("="*70)
    # Load maps
    target_map = Map(target_path)
    reference_map = Map(reference_path)
    target_map.plot_settings = {
      'norm': normit(target_map.data),
      'cmap': 'magma',
      'aspect': 'equal',
    }
    # Auto-discover FSI sequence
    
    date_avg = np.datetime64(target_map.date.isot)
    half_day = np.timedelta64(12, "h")
    date_beg = date_avg - half_day
    date_end = date_avg + half_day
    
    local_dir = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2")
    if not local_dir.is_dir():
        local_dir = reference_path.parent
    
    fsi_paths = get_EUI_paths(date_beg, date_end, local_dir=local_dir, verbose=verbose)
    fsi_174_paths = [p for p in fsi_paths if 'fsi174' in p.name.lower() and 'short' not in p.name.lower()]
    
    if not fsi_174_paths:
        fsi_sequence = [reference_map]
    else:
        fsi_sequence = fsi_174_paths
    
    _vprint(verbose, 1, f"Found {len(fsi_sequence)} FSI images for synthetic raster")
    
    # Setup debug output if verbose >= 3
    debug_writer = None
    output_dir = None
    if verbose >= 3:
        from matplotlib.backends.backend_pdf import PdfPages
        output_dir = Path("data_storage/output_figures_2")
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_pdf = output_dir / "debug_complete_pipeline.pdf"
        debug_writer = PdfPages(debug_pdf)
        print(f"[Debug] PDF output: {debug_pdf}")
    
    # Create root node (workers setup automatically)
    root = Orchestrator(n_workers=48)
    root.node_id = "root"
    root.node_name = "Root Orchestrator"
    root.base_target_map = target_map
    root.reference_map = reference_map
    root.reference_sequence = fsi_sequence
    root.verbose = verbose
    root.debug_writer = debug_writer
    root.output_directory = output_dir
    
    # Add all three processes
    root.add_child(SingleMapProcess(
        max_shift=700.0 * u.arcsec,
        bin_kernel=50.0 * u.arcsec,
        n_neighbors=48
    ))
    
    root.add_child(SingleMapProcess(
        max_shift=100.0 * u.arcsec,  # kernel_size * 2
        bin_kernel=0.0 * u.arcsec,
        n_neighbors=48
    ))
    
    root.add_child(SyntheticRasterProcess(
        max_shift=50.0 * u.arcsec,  # kernel_size
        scale_step=0.01,
        n_neighbors=80
    ))
    
    # Execute entire pipeline
    root.execute()
    
    # Cleanup workers
    root.cleanup()
    
    # Close debug writer
    if debug_writer is not None:
        debug_writer.close()
        print(f"\n[Debug] Saved PDF: {debug_pdf}")
    
    # Print all results
    all_results = root.get_all_results()
    
    print("\n" + "="*70)
    print("PIPELINE RESULTS:")
    print("="*70)
    
    for result in all_results:
        print(f"\n[{result.process_name}]")
        print(f"  Shift: ({result.shift_arcsec[0]:.2f}, {result.shift_arcsec[1]:.2f})")
        if result.scale_factors[0] != 1.0:
            print(f"  Scale: ({result.scale_factors[0]:.4f}, {result.scale_factors[1]:.4f})")
        print(f"  Correlation: {result.correlation_peak:.5f}")
        print(f"  Time: {result.execution_time:.1f}s")
        if result.animation_path:
            print(f"  Blink GIF: {result.animation_path.resolve()}")
    
    return root


def main():
    parser = argparse.ArgumentParser(description="Test new coalignment framework")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("fits_files/SPICE_706_2024-10-17T000050.215.fits"),
        help="Path to SPICE raster"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("fits_files/FSI_174_2024-10-17T014055.208.fits"),
        help="Path to FSI reference"
    )
    parser.add_argument(
        "--test",
        choices=["binning", "full-res", "synthetic", "complete", "all"],
        default="complete",
        help="Which test to run"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity level (0-4)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.target.is_file():
        print(f"ERROR: Target file not found: {args.target}")
        return 1
    if not args.reference.is_file():
        print(f"ERROR: Reference file not found: {args.reference}")
        return 1
    
    # Run tests
    if args.test == "binning" or args.test == "all":
        test_binning_only(args.target, args.reference, args.verbose)
    
    if args.test == "full-res" or args.test == "all":
        test_full_resolution_only(args.target, args.reference, args.verbose)
    
    if args.test == "synthetic" or args.test == "all":
        test_synthetic_raster_only(args.target, args.reference, args.verbose)
    
    if args.test == "complete" or args.test == "all":
        test_complete_pipeline(args.target, args.reference, args.verbose)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
