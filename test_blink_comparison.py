"""Quick test to verify blink comparison rendering works."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
from pathlib import Path

# Mock GenericMap class for testing
class MockGenericMap:
    def __init__(self, data, meta=None, plot_settings=None):
        self.data = data
        self.meta = meta or {}
        self.plot_settings = plot_settings or {"cmap": "sdoaia304"}

# Mock get_coord_mat function
def mock_get_coord_mat(map_obj):
    ny, nx = map_obj.data.shape
    lon = np.linspace(-100, 100, nx)
    lat = np.linspace(-100, 100, ny)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon_grid, lat_grid

# Test the blink comparison logic
def test_blink_comparison():
    """Test that blink comparison frames can be generated."""
    
    # Create synthetic maps
    ny, nx = 50, 50
    ref_data = np.random.rand(ny, nx) * 100
    target_data = np.random.rand(ny, nx) * 80 + np.roll(ref_data, (2, 3), axis=(0, 1)) * 0.3
    corrected_data = np.random.rand(ny, nx) * 80 + np.roll(ref_data, (0, 1), axis=(0, 1)) * 0.4
    
    ref_map = MockGenericMap(ref_data)
    target_map = MockGenericMap(target_data)
    corrected_map = MockGenericMap(corrected_data)
    
    # Monkey patch get_coord_mat
    import coalign_debug
    original_get_coord_mat = coalign_debug.get_coord_mat
    coalign_debug.get_coord_mat = mock_get_coord_mat
    
    # Also mock normit to return a simple normalizer
    def mock_normit(data):
        from matplotlib.colors import Normalize
        return Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    
    coalign_debug.normit = mock_normit
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_blink.pdf"
            
            with PdfPages(pdf_path) as pdf:
                from coalign_debug import DebugPlotContext
                
                # Create a temporary debug context
                ctx = DebugPlotContext(
                    pdf_writer=pdf,
                    fig=None,
                    ax=None,
                    color_mappable=None,
                    debug_points=[],
                    plotted_points=set(),
                    pdf_path=pdf_path,
                    owns_writer=False,
                )
                
                # Render blink comparison
                ctx.render_blink_comparison(ref_map, target_map, corrected_map, phase_name="test", n_cycles=2)
            
            # Check GIF was created
            gif_files = list(Path(tmpdir).glob("blink_comparison_test_*.gif"))
            assert len(gif_files) > 0, "GIF file should be created"
            gif_path = gif_files[0]
            assert gif_path.stat().st_size > 1000, "GIF should have content"
            
            print(f"✓ Blink comparison rendered successfully")
            print(f"  GIF: {gif_path.stat().st_size / 1024:.1f} KB at {gif_path.name}")
            
    finally:
        # Restore original functions
        coalign_debug.get_coord_mat = original_get_coord_mat

if __name__ == "__main__":
    print("Testing blink comparison rendering...")
    test_blink_comparison()
    print("✅ Blink comparison test passed!")
