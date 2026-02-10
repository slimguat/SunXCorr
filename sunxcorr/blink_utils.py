"""Utility functions for creating blink animations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sunpy.map import GenericMap


def make_blink_animation(
    before_map: GenericMap,
    after_map: GenericMap,
    output_path: str | Path,
    title: str = "Before/After Comparison",
    interval: int = 800,
    n_cycles: int = 3,
) -> None:
    """
    Create a blink animation GIF comparing before and after maps.

    Parameters
    ----------
    before_map : GenericMap
        Map before correction
    after_map : GenericMap
        Map after correction
    output_path : str or Path
        Output path for GIF file
    title : str
        Title for the animation
    interval : int
        Interval between frames in milliseconds
    n_cycles : int
        Number of blink cycles
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data
    before_data = np.asarray(before_map.data)
    after_data = np.asarray(after_map.data)

    # Normalize for display
    vmin = min(np.nanpercentile(before_data, 1), np.nanpercentile(after_data, 1))
    vmax = max(np.nanpercentile(before_data, 99), np.nanpercentile(after_data, 99))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(before_data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\nBefore")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Animation function
    frames = []
    for _ in range(n_cycles):
        frames.extend(["before", "after"])

    def update(frame_name):
        if frame_name == "before":
            im.set_data(before_data)
            ax.set_title(f"{title}\nBefore")
        else:
            im.set_data(after_data)
            ax.set_title(f"{title}\nAfter")
        return [im]

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=1000 // interval)
    anim.save(output_path, writer=writer)
    plt.close(fig)
