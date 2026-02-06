# Blink Animation Feature - Implementation Summary

## Overview
The blink animation feature creates side-by-side GIF animations showing before/after correction comparisons during the coalignment process. These animations are automatically generated at the end of each search phase when debug mode is active (verbosity ≥ 3).

## Implementation Details

### 1. Core Method: `render_blink_comparison()`
**Location:** `coalign_debug.py` (lines 97-164)

**Signature:**
```python
def render_blink_comparison(
    self,
    ref_map: GenericMap,
    target_map: GenericMap,
    corrected_map: GenericMap,
    phase_name: str = "",
    interval: int = 800,
    n_cycles: int = 5,
) -> None
```

**Key Features:**
- Creates a 2-panel figure (14x6 inches)
- Left panel: Reference map ↔ Original target (before correction)
- Right panel: Reference map ↔ Corrected target (after correction)
- Adds percentile contours (5, 10, 80, 90, 95, 99) for visual guidance
- Saves as GIF using PillowWriter
- Filename format: `blink_comparison_{phase_name}_{timestamp}.gif`
- Saved in the same directory as the PDF debug output

**Animation Parameters:**
- `interval`: 800ms between frames (default)
- `n_cycles`: 5 complete blink cycles (default)
- Total frames: 10 (2 frames per cycle × 5 cycles)

### 2. Integration Points in Coaligner

The blink comparison is called at **three locations** in `coaligner.py`:

#### A. Shift+Scale Search (`_run_shift_scale_search`)
**Line:** ~833
```python
if corrected_for_viz is not None and isinstance(ref_map_obj, GenericMap):
    phase_num = 1 if phase_key == 'binning' else (2 if phase_key == 'one_map' else 3)
    debug_ctx.render_blink_comparison(
        ref_map_obj, target_map_obj, corrected_for_viz, 
        phase_name=phase_label(phase_num)
    )
```

#### B. Shift-Only Search (`_run_shift_search`)
**Line:** ~1642
```python
if corrected_for_viz is not None and isinstance(ref_map_obj, GenericMap):
    phase_num = 1 if phase_key == 'binning' else 2
    debug_ctx.render_blink_comparison(
        ref_map_obj, target_map_obj, corrected_for_viz,
        phase_name=phase_label(phase_num)
    )
```

#### C. Phase Finalization (`_finalize_phase_result`)
**Line:** ~1317
```python
if self._shared_debug_writer is not None and corrected_extended is not None:
    temp_ctx = DebugPlotContext(...)
    phase_num = 1 if phase_key == 'binning' else 2
    temp_ctx.render_blink_comparison(
        ref_map, target_map, corrected_extended,
        phase_name=phase_label(phase_num)
    )
```

### 3. Phase Name Mapping

The phase names are automatically determined:
- Phase 1 (binning) → `"search phase"` or `"plateau phase"`
- Phase 2 (one-map) → `"search phase"` or `"plateau phase"`
- Phase 3 (synthetic_raster) → `"search phase"` or `"plateau phase"`

## Usage

### Automatic Generation
When running coalignment with verbosity ≥ 3:
```bash
python test_coaligner.py --jobs 4 --neighbors 4 --shift-x 20 --shift-y 20
```

The GIF files will be automatically created in:
```
Cross_correlation/data_storage/xcorr/blink_comparison_*.gif
```

### Output Files
Example files from a successful run:
```
blink_comparison_search phase_20260206_101138.gif     (1.6 MB)
blink_comparison_plateau phase_20260206_101226.gif    (1.3 MB)
```

## Testing

### Unit Test
**File:** `test_blink_comparison.py`

Run with:
```bash
cd Cross_correlation
python test_blink_comparison.py
```

Expected output:
```
Testing blink comparison rendering...
2026-02-06 11:09:42 - matplotlib.animation - INFO: Animation.save using <class 'matplotlib.animation.PillowWriter'>
✅ Blink animation saved: /tmp/.../blink_comparison_test_20260206_100942.gif
✓ Blink comparison rendered successfully
  GIF: 621.3 KB at blink_comparison_test_20260206_100942.gif
✅ Blink comparison test passed!
```

### Integration Test
**File:** `test_coaligner.py`

The full coalignment test will generate GIF animations for each phase where debug output is active.

## Technical Notes

1. **Dependencies:**
   - matplotlib.animation.FuncAnimation
   - matplotlib.animation.PillowWriter
   - Pillow (PIL) must be installed in the environment

2. **Performance:**
   - Each GIF takes ~2-3 seconds to generate
   - File size: typically 1-2 MB per animation
   - Does not significantly impact overall coalignment runtime

3. **Animation Quality:**
   - Resolution: 100 DPI
   - FPS: ~1.25 (calculated from 800ms interval)
   - Repeats infinitely when viewed in most viewers

4. **Error Handling:**
   - If GIF saving fails, a warning is printed but execution continues
   - PDF debug output is unaffected by GIF generation failures

## Viewing the Animations

GIF files can be viewed with:
- Web browsers (drag and drop)
- Image viewers (Eye of GNOME, Preview, etc.)
- Scientific notebook environments (Jupyter, VS Code)

Example Python viewing code:
```python
from IPython.display import Image, display
display(Image(filename='blink_comparison_search phase_20260206_101138.gif'))
```

## Future Enhancements

Potential improvements:
1. Add MP4 format option (requires ffmpeg)
2. Configurable FPS and cycle count
3. Optional side-by-side difference maps
4. Customizable contour levels
5. Annotation overlay (correlation values, shift parameters)

## See Also

- `CHANGES_adaptive_debug_plots.md` - Adaptive history plotting feature
- `CHANGES_blink_animation.md` - Detailed technical documentation
- `coalign_debug.py` - Debug plotting context implementation
- `coaligner.py` - Main coalignment orchestrator
