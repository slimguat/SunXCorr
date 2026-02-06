# Adaptive Debug Plot Feature

## Summary
Added adaptive debug plotting to the coalignment optimizer that automatically adjusts the visualization based on whether scale optimization is active.

## Changes Made

### 1. `coalign_debug.py` - New Method Added
**Location:** `DebugPlotContext` class

**New Method:** `render_history_plot(history: np.ndarray)`

**Functionality:**
- Automatically detects if sx and sy parameters vary during optimization (standard deviation > 1e-6)
- **When sx/sy are constant** (no scale optimization):
  - Creates 2-axis figure:
    - dx vs dy scatter plot (colored by correlation)
    - correlation vs iteration line plot
- **When sx/sy vary** (scale optimization active):
  - Creates 3-axis figure:
    - dx vs dy scatter plot (colored by correlation)
    - sx vs sy scatter plot (colored by correlation)
    - correlation vs iteration line plot

**Parameters:**
- `history`: numpy array of shape (N, 6) with columns [iteration, dx, dy, sx, sy, corr]

### 2. `coaligner.py` - Integration Points
**Locations:** Two optimization methods were updated

#### Location 1: Around line 807-820
Added history plot rendering before alignment overlay in the first optimization path.

#### Location 2: Around line 1575-1588  
Added history plot rendering before alignment overlay in the second optimization path.

**Implementation:**
```python
# Convert history list to numpy array
history_array = np.array([
    [h['iteration'], h['dx'], h['dy'], h['sx'], h['sy'], h['corr']] 
    for h in history
], dtype=float)

# Render adaptive plot
debug_ctx.render_history_plot(history_array)
```

## Usage
This feature is automatically activated when debug context is enabled during coalignment. No user code changes are required - the visualization adapts based on the optimization parameters.

## Benefits
1. **Automatic adaptation**: No need to manually choose between 2 or 3 axes
2. **Cleaner output**: Only shows relevant plots for the optimization type
3. **Better diagnostics**: Easier to understand what parameters were optimized
4. **Consistent with workflow**: Works for both binned and full-resolution searches

## Testing
Run any coalignment workflow with debug enabled:
- **Shift-only** (ll_cash case): Will show 2-axis plot (dx vs dy + corr vs iteration)
- **Shift+scale** (raster synthetic case): Will show 3-axis plot (dx vs dy, sx vs sy, corr vs iteration)
