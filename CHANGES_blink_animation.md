# Blink Animation Feature for Debug Plots

## Summary
Added blink animation comparison to coalignment debug plots, showing side-by-side before/after correction visualizations.

## New Features

### 1. `DebugPlotContext.render_blink_comparison()` Method
**Location:** [coalign_debug.py](coalign_debug.py)

**Functionality:**
Creates a side-by-side blink animation comparing:
- **Left panel:** Reference map blinking with original uncorrected target
- **Right panel:** Reference map blinking with corrected target

**Parameters:**
- `ref_map`: Reference map (FSI, binned, or synthetic raster depending on phase)
- `target_map`: Original uncorrected target map
- `corrected_map`: WCS-corrected target map
- `interval`: Time between frames (default 800ms)
- `n_cycles`: Number of blink cycles (default 5)

**Output:**
- Multiple PDF frames showing the blink sequence
- Contour overlays using percentile levels (p5, p10, p80, p90, p95, p99)
- Coordinated axis limits across both panels

### 2. Integration Points

#### A. Shift-Only Search (`_run_shift_search`)
- Builds corrected map on-the-fly for visualization
- Calls `render_blink_comparison` after alignment overlay
- Used in binning and one-map phases

#### B. Shift+Scale Search (`_run_shift_scale_search`)
- Builds corrected map with both shift and scale parameters
- Renders blink comparison for synthetic raster phase
- Shows how scale optimization affects alignment

#### C. Phase Finalization (`_finalize_phase_result`)
- Renders blink comparison after corrected maps are saved
- Uses extended target map for proper visualization
- Conditional on debug session being active

## How It Works

1. **Reference Selection:**
   - Binning phase: Uses reprojected reference on binned grid
   - One-map phase: Uses reprojected reference on full resolution grid
   - Synthetic phase: Uses synthetic FSI raster as reference

2. **Animation Frames:**
   - Even frames: Show reference map (grayscale) + target contours
   - Odd frames: Show target/corrected map (color) with SDO AIA 304 colormap
   - Contours remain visible on reference frames for context

3. **Spatial Alignment:**
   - Both panels use same coordinate system (Tx, Ty in arcsec)
   - Axis limits derived from reference map with 10% padding
   - All maps transformed to common helioprojective frame

## Usage

The blink animations are automatically generated when:
1. Debug verbosity ≥ 3 (enables debug plots)
2. All required maps are available (reference, target, corrected)
3. Debug PDF writer is active

No user code changes needed - runs automatically during coalignment.

## Example Output Structure

For each phase with debug enabled:
```
xcorr_debug_TIMESTAMP.pdf:
  ├─ Scatter progression plots
  ├─ History plot (adaptive 2 or 3 axes)
  ├─ Alignment overlay (2×2 grid)
  ├─ Blink comparison frames (10 frames for n_cycles=5)
  └─ [Next phase...]
```

## Testing

Run the test suite to verify:
```bash
python test_blink_comparison.py  # Unit test for rendering logic
python test_coaligner.py          # Full integration test
```

## Benefits

1. **Visual Validation:** Quickly see if correction improved alignment
2. **Before/After Comparison:** Side-by-side makes quality assessment easier
3. **Animation Context:** Blinking helps identify spatial shifts and scaling
4. **Phase-Specific:** Uses appropriate reference for each optimization phase
5. **Automatic:** No manual intervention required

## Technical Notes

- Animations saved as individual frames to PDF (not as video)
- Uses matplotlib's FuncAnimation internally but saves static frames
- Contours computed from data percentiles for robust visualization
- Handles NaN values gracefully in contour and color mapping
- Memory efficient: creates temporary figures, closes after saving
