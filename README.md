# Solar Coalignment

A Python library for high-precision coalignment of solar spectroscopic rasters with photometric reference images using cross-correlation optimization.

## Overview

This library provides a robust three-phase pipeline for aligning SPICE spectroscopic rasters with EUI/FSI reference images. The progressive refinement approach balances computational efficiency with sub-pixel accuracy, making it suitable for operational pipelines and scientific analysis.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import sunpy.map
from coaligner import Coaligner

# Load maps
spice_raster = sunpy.map.Map('spice_raster.fits')
fsi_reference = sunpy.map.Map('eui_fsi_174.fits')

# Initialize and run coalignment
aligner = Coaligner(
    map_to_coalign=spice_raster,
    reference_map=fsi_reference,
    n_jobs=8
)

# Execute three-phase alignment
aligner.run_binned_xcorr()
aligner.run_one_map_xcorr()
aligner.run_synthetic_raster_xcorr()

# Access corrected WCS parameters
corrected_wcs = aligner.best_params
corrected_map = aligner.results['synthetic_raster']['corrected_target_map']
```

## Algorithm

The library implements a three-phase progressive refinement strategy:

### Phase 1: Binned Cross-Correlation
- **Purpose**: Fast coarse alignment
- **Method**: Spatially binned maps (~50 arcsec/pixel)
- **Search space**: 2D shift (±20-40 pixels)
- **Typical runtime**: 10-30 seconds

### Phase 2: Full-Resolution Refinement
- **Purpose**: Sub-pixel accuracy
- **Method**: Full-resolution single reference map
- **Search space**: 2D shift (±10 pixels)
- **Typical runtime**: 2-5 minutes

### Phase 3: Synthetic Raster Optimization
- **Purpose**: Temporal matching and scale correction
- **Method**: Synthetic raster from FSI image sequence
- **Search space**: 4D (shift + scale)
- **Typical runtime**: 5-15 minutes

The optimizer uses discrete gradient-based stochastic search with:
- Parallel correlation evaluation via persistent worker processes
- Correlation-weighted population gradients
- Adaptive step sizes with plateau detection
- Graceful convergence criteria

## API Reference

### Main Class

```python
class Coaligner:
    """Three-phase solar image coalignment pipeline."""
    
    def __init__(self, map_to_coalign, reference_map, 
                 list_of_reference_maps=None, verbose=0, n_jobs=cpu_count()-1)
        """Initialize coaligner with maps and runtime settings."""
    
    def run_binned_xcorr(self)
        """Execute Phase 1: coarse alignment with binned maps."""
    
    def run_one_map_xcorr(self, seed_shift=None)
        """Execute Phase 2: full-resolution refinement."""
    
    def run_synthetic_raster_xcorr(self)
        """Execute Phase 3: synthetic raster optimization."""
```

### Key Functions

```python
def correlation_for_params(ref_img, target_img, dx, dy, sx, sy, center=None)
    """NaN-safe normalized correlation with geometric parameters."""

def make_corrected_wcs_map(map_obj, params, verbose=0)
    """Apply geometric correction to map WCS metadata."""

def reproject_map_to_reference(target_map, reference_map)
    """Reproject target onto reference grid."""

def build_synthetic_raster_from_maps(target_map, fsi_maps, ...)
    """Generate synthetic raster from FSI sequence."""
```

## Configuration

### Optimization Parameters

```python
aligner = Coaligner(spice_map, fsi_map, n_jobs=16)

# Phase 1 configuration
aligner.xcorr_binned_kwargs['shift_range'] = (30, 30)
aligner.xcorr_binned_kwargs['plateau_iters'] = 5
aligner.bin_kernel_arcsec = 60 * u.arcsec

# Phase 2 configuration
aligner.xcorr_one_map_kwargs['shift_range'] = (15, 15)
aligner.xcorr_one_map_kwargs['max_corr'] = 0.9

# Phase 3 configuration
aligner.synthetic_kwargs['scale_range'] = (0.95, 1.05)
aligner.synthetic_kwargs['scale_step_x'] = 5e-4
aligner.synthetic_kwargs['n_neighbors'] = 120
aligner.synthetic_kwargs['reference_local_dir'] = '/data/eui/fsi174'
```

### FSI Image Discovery

```python
# Automatic discovery from filesystem
aligner.synthetic_reference_time_window = np.timedelta64(2, 'h')
aligner.synthetic_kwargs['reference_channel_keyword'] = 'fsi174'
aligner.synthetic_kwargs['reference_exclude_tokens'] = ('short', 'test')

# Or provide explicit list
fsi_maps = [sunpy.map.Map(f) for f in fsi_files]
aligner = Coaligner(spice_map, fsi_ref, list_of_reference_maps=fsi_maps)
```

## Performance Optimization

1. **Parallelization**: Set `n_jobs` to number of physical cores
2. **Neighbor sampling**: Balance between exploration (120) and speed (40)
3. **Scale bounds**: Tighten `scale_range` if plate scales are well calibrated
4. **Convergence**: Adjust `plateau_iters` for patience vs speed tradeoff
5. **Early stopping**: Set `max_corr` threshold for sufficient alignment

## Requirements

- Python ≥ 3.11
- NumPy
- SunPy ≥ 5.0
- Astropy
- Matplotlib
- SciPy
- reproject

See `requirements.txt` for complete dependency list.

## Testing

```bash
# Run test suite
python test_coaligner.py
python test_adaptive_plot.py
python test_blink_comparison.py

# With custom data
python test_coaligner.py --spice /path/to/spice.fits --fsi /path/to/fsi.fits
```

## Module Structure

```
coaligner.py              # Main orchestrator class
coalign_helpers.py        # Search algorithms and gradients
coalign_workers.py        # Persistent worker processes
coalign_debug.py          # Visualization context
coalign_preprocess.py     # Data preparation
help_funcs.py             # Map manipulation and I/O
slimfunc_correlation_effort.py  # Core correlation functions
```

## Credits

This work builds upon alignment tools and techniques by:
- [Antoine Dolliou](https://github.com/adolliou/euispice_coreg)
- [Gabriel Pelouze](https://git.ias.u-psud.fr/gpelouze/align_images)
- [Frédéric Auchère](http://git.ias.u-psud.fr/fauchere)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{solar_coalignment,
  author = {Zergua, Salim},
  title = {Solar Coalignment Library},
  year = {2026},
  url = {https://github.com/yourusername/Cross_correlation}
}
```
