# SunXCorr — Minimal Docs

This repository contains the `sunxcorr` package (in `Cross_correlation/sunxcorr`) for solar image coalignment and synthetic raster generation.

Quick start
-----------
- Install the environment (use the provided `envs/SAFFRON_ENV` or create a new venv).
- Run the focused test suite (only collects tests under `Cross_correlation/tests`):

```bash
python -m pytest
```

Running quick tests (no pytest)
-------------------------------
To run the small quick-runner that avoids whole-repo pytest collection:

```bash
/path/to/python Cross_correlation/tests/run_tests_quick.py
```

Next steps
----------
- Expand docstrings in the `sunxcorr` package.
- Add Sphinx docs and examples (not included yet).
- Convert the quick-runner tests to be fully pytest-friendly if desired.
