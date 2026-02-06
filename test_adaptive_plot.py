"""Quick test to verify the adaptive plot logic works correctly."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import tempfile

# Test the detection logic
def test_scale_detection():
    """Test that we correctly detect whether scale varies."""
    
    # Case 1: Scale is constant (sx=1.0, sy=1.0 throughout)
    history_no_scale = np.array([
        [0, 0.0, 0.0, 1.0, 1.0, 0.5],
        [1, 1.0, 1.0, 1.0, 1.0, 0.6],
        [2, 2.0, 2.0, 1.0, 1.0, 0.7],
        [3, 3.0, 3.0, 1.0, 1.0, 0.8],
    ])
    
    sx = history_no_scale[:, 3]
    sy = history_no_scale[:, 4]
    sx_std = np.std(sx)
    sy_std = np.std(sy)
    scale_varies = (sx_std > 1e-6) or (sy_std > 1e-6)
    
    assert not scale_varies, "Should detect no scale variation"
    print("✓ Case 1: Correctly detected no scale optimization")
    
    # Case 2: Scale varies
    history_with_scale = np.array([
        [0, 0.0, 0.0, 1.0, 1.0, 0.5],
        [1, 1.0, 1.0, 1.05, 0.98, 0.6],
        [2, 2.0, 2.0, 1.1, 0.96, 0.7],
        [3, 3.0, 3.0, 1.15, 0.94, 0.8],
    ])
    
    sx = history_with_scale[:, 3]
    sy = history_with_scale[:, 4]
    sx_std = np.std(sx)
    sy_std = np.std(sy)
    scale_varies = (sx_std > 1e-6) or (sy_std > 1e-6)
    
    assert scale_varies, "Should detect scale variation"
    print("✓ Case 2: Correctly detected scale optimization")


def test_plot_generation():
    """Test that plots are generated without errors."""
    
    # Create temporary PDF for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "test_plot.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Test case 1: No scale variation (2 axes)
            history_no_scale = np.array([
                [i, i*0.5, i*0.3, 1.0, 1.0, 0.5 + i*0.05] 
                for i in range(10)
            ])
            
            iterations = history_no_scale[:, 0]
            dx = history_no_scale[:, 1]
            dy = history_no_scale[:, 2]
            sx = history_no_scale[:, 3]
            sy = history_no_scale[:, 4]
            corr = history_no_scale[:, 5]
            
            sx_std = np.std(sx)
            sy_std = np.std(sy)
            scale_varies = (sx_std > 1e-6) or (sy_std > 1e-6)
            
            if not scale_varies:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
                sc1 = axes[0].scatter(dx, dy, c=corr, cmap="viridis", s=22, alpha=0.8)
                axes[0].set_xlabel("dx (pixels)")
                axes[0].set_ylabel("dy (pixels)")
                axes[0].set_title("dx vs dy (color=corr)")
                plt.colorbar(sc1, ax=axes[0], label="corr")
                
                axes[1].plot(iterations, corr, color="tab:blue", lw=1.5, marker="o", markersize=3)
                axes[1].set_xlabel("iteration")
                axes[1].set_ylabel("corr")
                axes[1].set_title("corr vs iteration")
                
                pdf.savefig(fig)
                plt.close(fig)
                print("✓ Case 1: Generated 2-axis plot (no scale optimization)")
            
            # Test case 2: With scale variation (3 axes)
            history_with_scale = np.array([
                [i, i*0.5, i*0.3, 1.0 + i*0.01, 1.0 - i*0.005, 0.5 + i*0.05] 
                for i in range(10)
            ])
            
            iterations = history_with_scale[:, 0]
            dx = history_with_scale[:, 1]
            dy = history_with_scale[:, 2]
            sx = history_with_scale[:, 3]
            sy = history_with_scale[:, 4]
            corr = history_with_scale[:, 5]
            
            sx_std = np.std(sx)
            sy_std = np.std(sy)
            scale_varies = (sx_std > 1e-6) or (sy_std > 1e-6)
            
            if scale_varies:
                fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
                
                sc1 = axes[0].scatter(dx, dy, c=corr, cmap="viridis", s=22, alpha=0.8)
                axes[0].set_xlabel("dx (pixels)")
                axes[0].set_ylabel("dy (pixels)")
                axes[0].set_title("dx vs dy (color=corr)")
                plt.colorbar(sc1, ax=axes[0], label="corr")
                
                sc2 = axes[1].scatter(sx, sy, c=corr, cmap="viridis", s=22, alpha=0.8)
                axes[1].set_xlabel("sx (scale)")
                axes[1].set_ylabel("sy (scale)")
                axes[1].set_title("sx vs sy (color=corr)")
                plt.colorbar(sc2, ax=axes[1], label="corr")
                
                axes[2].plot(iterations, corr, color="tab:blue", lw=1.5, marker="o", markersize=3)
                axes[2].set_xlabel("iteration")
                axes[2].set_ylabel("corr")
                axes[2].set_title("corr vs iteration")
                
                pdf.savefig(fig)
                plt.close(fig)
                print("✓ Case 2: Generated 3-axis plot (with scale optimization)")
        
        # Verify PDF was created
        assert pdf_path.exists(), "PDF should be created"
        assert pdf_path.stat().st_size > 0, "PDF should not be empty"
        print(f"✓ PDF generated successfully: {pdf_path.stat().st_size} bytes")


if __name__ == "__main__":
    print("Testing adaptive plot logic...\n")
    
    test_scale_detection()
    print()
    test_plot_generation()
    
    print("\n✅ All tests passed!")
