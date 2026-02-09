# New Coalignment Framework - Implementation Summary

## Files Created

### Package Structure: `coalignment/`

1. **`__init__.py`** - Public API exports
   - Exports: CoalignmentNode, SingleMapProcess, SyntheticRasterProcess, ProcessResult

2. **`process_result.py`** - Data structures
   - ProcessResult dataclass with all metadata

3. **`utils.py`** - Utility functions
   - Unit conversions (arcsec ↔ pixels)
   - Map operations (binning, shift application)
   - Functions: get_pixel_scale_quantity, arcsec_to_pixels, pixels_to_arcsec, bin_map, apply_shift_to_map, apply_shift_and_scale_to_map

4. **`optimization.py`** - Core optimization algorithm
   - optimize_shift_and_scale() - Unified for 2D (shift) and 4D (shift+scale)
   - scale_step=0 → 2D search
   - scale_step>0 → 4D search
   - Helper functions: _generate_neighbors, _evaluate_single_point, _evaluate_points_batch

5. **`base_node.py`** - Abstract base class
   - CoalignmentNode - Composite pattern implementation
   - Tree structure (parent, children)
   - Memory-safe resource getters (get_worker_pool, get_debug_writer, etc.)
   - Data getters (get_base_target_map, get_working_map, etc.)
   - Result collection (get_final_result, get_all_results)

6. **`single_map_process.py`** - Shift-only process
   - SingleMapProcess(max_shift, bin_kernel, ...)
   - Handles both binned and full-resolution
   - Uses optimization.py with scale_step=0

7. **`synthetic_raster_process.py`** - Shift+scale process
   - SyntheticRasterProcess(max_shift, scale_step, ...)
   - Builds synthetic raster from FSI sequence
   - Uses optimization.py with scale_step>0

8. **`../test_coalignment.py`** - Test script
   - 4 test scenarios:
     1. Binning only
     2. Full resolution only
     3. Synthetic raster only
     4. Complete pipeline (all three)

## Key Design Principles

### 1. Composite Pattern
- Single base class for both orchestrator and leaf processes
- Nodes can have children (composite) or logic (leaf), not both
- Recursive execution through tree

### 2. Memory Safety
- Data lives at root only (base_target_map, current_working_map)
- Children access via getters (no duplication)
- Resources inherited up the tree

### 3. Astropy Units
- All angle parameters use u.Quantity
- Validation at constructor
- Automatic conversion to pixels internally

### 4. Unified Optimization
- Single algorithm for shift and shift+scale
- scale_step parameter controls dimensionality
- Reused code from old coaligner.py

### 5. Short Methods
- Process classes delegate to external functions
- optimization.py handles algorithm
- utils.py handles map operations
- Classes remain readable

## Verbosity Levels (Preserved)

- `-1`: Warnings only
- `0`: Important info (phase start/end)
- `1`: Verbose (iteration progress, results)
- `2`: Debug info (configuration, intermediate values)
- `3+`: Debug plots (PDF, blink animations) - TODO

## Usage Example

```python
from coalignment import CoalignmentNode, SingleMapProcess, SyntheticRasterProcess
import astropy.units as u

# Create root
root = CoalignmentNode()
root.base_target_map = spice_map
root.reference_map = fsi_map
root.reference_sequence = fsi_images
root.verbose = 2

# Add processes
root.add_child(SingleMapProcess(
    max_shift=20.0 * u.arcsec,
    bin_kernel=50.0 * u.arcsec
))

root.add_child(SingleMapProcess(
    max_shift=10.0 * u.arcsec,
    bin_kernel=0.0 * u.arcsec
))

root.add_child(SyntheticRasterProcess(
    max_shift=40.0 * u.arcsec,
    scale_step=0.001
))

# Execute entire tree
root.execute()

# Get results
final_result = root.get_final_result()
all_results = root.get_all_results()
```

## Testing

Run test script:
```bash
cd /home/smzergua/workshop/2026/Cross_correlation

# Test individual processes
python test_coalignment.py --test binning --verbose 2
python test_coalignment.py --test full-res --verbose 2
python test_coalignment.py --test synthetic --verbose 2

# Test complete pipeline
python test_coalignment.py --test complete --verbose 2

# Run all tests
python test_coalignment.py --test all --verbose 2
```

## TODO

1. **Debug Visualization** (np.abs(verbose) >= 3)
   - Integrate coalign_debug.py
   - Generate PDF plots
   - Create blink animations

2. **Worker Pool Integration**
   - Currently uses serial execution
   - Need to integrate coalign_workers.py
   - Parallel correlation evaluation

3. **Performance Testing**
   - Compare with old coaligner.py
   - Verify correlation values match
   - Check execution times

4. **Documentation**
   - Add docstring examples
   - Create user guide
   - Migration guide from old API

## Files Unchanged (Legacy)

- `coaligner.py` - Original implementation (preserved for reference)
- `test_coaligner.py` - Original test (preserved for reference)
- All other existing files remain unchanged
