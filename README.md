# SunXCorr — Usage, Development, and Tutorials

This repository (SunXCorr) contains tools for cross-correlation and coalignment of solar images. The README documents how to set up, run, test, and develop the project, and includes tutorials showing common workflows.

## Table of contents
- [Quick start](#quick-start)
- [Setup (env + deps)](#setup-env-deps)
- [How the algorithm works (high level)](#how-the-algorithm-works)
- [Tests and static checks (mypy, pre-commit, pytest)](#tests-and-static-checks)
- [Development workflow](#development-workflow)
- [Tutorials](#tutorials)
- [Troubleshooting](#troubleshooting)

---

<a name="quick-start"></a>
## Quick start

1. Clone the repository and change into the project folder:

```bash
git clone <repo-url> SunXCorr
cd SunXCorr
```

2. Create and activate a Python 3.11 virtual environment (recommended):

```bash
python -m venv location/to/env
source location/to/env/bin/activate
python -m pip install --upgrade pip
```

3. Install runtime dependencies.

- If you use Poetry (recommended):

```bash
poetry install
```

- If you prefer pip and `requirements.txt`:

```bash
pip install -r requirements.txt
```

4. Run a quick smoke test (this uses a small, fast test runner that avoids heavy imports):

```bash
python tests/run_tests_quick.py
```

If the quick runner succeeds, you have a working core installation.

---

<a name="setup-env-deps"></a>
## Setup (env + deps)

- Recommended Python: 3.11
- Create a virtual environment and install the runtime `requirements.txt` in `SunXCorr`.
- If you need JIT-accelerated functions, install `numba` and a matching `numpy` in that environment.

Example full setup (optional):

```bash
# create + activate a venv
python -m venv ./env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r SunXCorr/requirements.txt
```

Notes:
- `numba` requires a working C toolchain. If you rely on JIT for performance, install `numba` in the same environment where `numpy` is installed.
- Dev tools such as formatters/linters (black, isort, pre-commit) are optional. Install them only when you are developing or contributing. They are not required to run SunXCorr:

```bash
# optional, for development only
pip install pre-commit black isort
pre-commit install
```

---

<a name="how-the-algorithm-works"></a>
## How the algorithm works (high level)

SunXCorr uses cross-correlation to find the best relative shifts (and optionally scales) between a target map and a reference map or between a target map and a synthetic raster built from multiple maps.

- Cross-correlation is performed either in single-map mode (shift-only search) or in synthetic-raster mode (shift + scale search).
- SingleMap processes perform translations (shifts) only and do not rescale the image. They can search a wider shift space, which makes them robust when the reference map is larger than the target.
- SyntheticRaster processes include scale corrections and operate on maps of the same size as the target — therefore their allowed shift search space must be smaller than the map size.

FITS metadata requirements (important): the map headers must contain time and axis metadata for correct handling and output updates:

- `CRVAL3`: a float representing the time reference value (seconds since the Unix epoch / 1970-01-01). This value is used for time alignment and downstream metadata updates.
- `CDELT3`: typically `1` (units per pixel along the time axis).
- `CUNIT3`: e.g. `"s"` for seconds.
- `PC3_1` (or appropriate WCS PC matrix entries): may be used to encode exposure/scan direction (exposure time may be reversed if the scan direction is reversed).

JIT-accelerated functions: the `sunxcorr` package provides typed-Python implementations and optional `numba` wrappers for runtime acceleration. The first call to a numba-jitted function warms up the JIT and may take longer; subsequent calls are faster.

Example usage (Python):

```python
from sunxcorr.correlation_jit import correlation_for_params_jit
import numpy as np

# Prepare small synthetic maps (see tests for examples)
map1 = np.ones((64, 64), dtype=np.float32)
map2 = np.ones((64, 64), dtype=np.float32)

result = correlation_for_params_jit(map1, map2, dx=0.0, dy=0.0, scale=1.0)
print(result)
```

---

<a name="tests-and-static-checks"></a>
## Tests and static checks

- Type checking: `mypy` is configured via `mypy.ini`. Legacy data lives under `SunXCorr/data_storage/` and may be ignored by the repo mypy config.

Run mypy for the package:

```bash
python -m mypy SunXCorr
```

- Pre-commit and formatters are optional for contributors. Install them only when developing.

- Tests: the project includes a quick runner and a full `pytest` suite.

Run the quick runner:

```bash
python SunXCorr/tests/run_tests_quick.py
```

Run the full pytest suite:

```bash
pytest SunXCorr
```

Git LFS note (important before running tests): some large data used by tests and examples may be stored with Git LFS. After cloning, fetch LFS files:

```bash
git clone <repo-url> SunXCorr
cd SunXCorr
git lfs install --local
git lfs pull
```

If you run tests immediately after cloning without pulling LFS objects, tests may fail because example data files are missing.

If `pytest` hangs during collection due to heavy imports (e.g., `astropy`), use the quick runner or run individual tests to isolate failures.

---

<a name="development-workflow"></a>
## Development workflow

1. Create a feature branch.
2. Run `mypy` and fix typing issues in small steps. The project uses a pattern where runtime-critical code keeps a numba-compiled copy while the typed-Python core exists for linters and mypy.
3. Run `pre-commit run --all-files` to auto-format and lint.
4. Run tests (`run_tests_quick.py` first, then `pytest`).
5. Commit and push.

Commit example:

```bash
git checkout -b fix/foo
git add <files>
git commit -m "Fix: ..."
git push origin fix/foo
```

---

<a name="tutorials"></a>
## Tutorials

Below we provide compact, practical tutorials for the most common workflows: single-map cross-correlation, synthetic-raster correlation, and an orchestrated multi-step pipeline.

Algorithm overview (read before running examples):

- Cross-correlation finds the best relative translation (and optionally scale) between maps by computing similarity metrics over a search grid.
- Single-map workflows only search for translations (shifts). They can explore large shift spaces since the reference map may be larger than the target.
- Synthetic-raster workflows include scale search (scale is unitless; 1.0 = no scale). Synthetic rasters are the same size as the target, so their shift search range must be smaller than the map dimensions.

Quick examples below assume you have valid FITS maps where the header includes the time axis metadata (`CRVAL3`, `CDELT3`, `CUNIT3`) as described above.

Cross-correlation with a single map

Preparation: have two maps and the paths available, e.g. `Path/to/your/reference_map.fits` and `Path/to/your/target_map.fits`. We want to correct the `target_map` using `reference_map`.

Example (Single map class usage):

```python
from sunxcorr import SingleMapProcess
import astropy.units as u

xcorr_single = SingleMapProcess(
	max_shift=100.0 * u.arcsec,        # required: astropy Quantity
	bin_kernel=1.0 * u.arcsec,         # required: astropy Quantity; if >1 data is binned
	min_corr=0.0,                      # minimum correlation threshold to stop early
	plateau_iters=2,                   # stop if top correlation repeats plateau_iters times
	n_workers=1,                       # default 1; set to 4 or more for parallel runs
	verbose=1,                         # 0=quiet, 1=normal, 2=debug, 3=plots (saved to output_directory)
)

xcorr_single.node_id = "example_single"
xcorr_single.name = "single_coalign"
xcorr_single.target_map = "Path/to/your/target_map.fits"
xcorr_single.reference_map = "Path/to/your/reference_map.fits"
xcorr_single.output_directory = "./data_storage/debug_output"

xcorr_single.execute()
result = xcorr_single.get_final_result()
# result is a ProcessResult-like object; the corrected map (with updated CRVALs, CDELTs, CRPIXs, PCs) is available in result.output_map
```

Notes on parameters:
- `max_shift` and `bin_kernel` must be `astropy.units.Quantity` objects (e.g., `100.0 * u.arcsec`).
- `min_corr` is a stopping threshold (previously called `max_corr` in older docs); the search stops early if correlation >= `min_corr` and plateau detection triggers.
- `plateau_iters` stops the search if the top correlation value repeats for this many iterations.
- `verbose` levels: `-q` for quiet (equivalent to `verbose=0`), `-v` or `-vv` increase verbosity.

The Process API:
- `execute()` runs the process.
- `get_final_result()` returns a `ProcessResult` containing metadata and the corrected output map.

Synthetic raster correlation

For synthetic raster workflows provide a sequence of maps (not a single reference). Synthetic processes allow small scale corrections as well:

```python
from sunxcorr import SyntheticRasterProcess
import astropy.units as u
import os

root2 = SyntheticRasterProcess(
	max_shift=100.0 * u.arcsec,
	scale_step=0.00,                 # 0.0 means no scale search; positive step searches scale around 1.0
	n_neighbors=os.cpu_count() * 2,
	plateau_iters=2,
)

# add inputs, execute as with SingleMapProcess
```

`scale_step` is a unitless increment for the scale factor (1.0 = no scaling). `n_neighbors` controls the local search density.

Orchestrator: multi-step highly accurate pipeline

Below is a recommended staged pipeline that first locates a coarse solution in a large search space and then refines it with finer kernels and optional synthetic-raster steps.

```python
import os
import astropy.units as u
from sunxcorr import SingleMapProcess, SyntheticRasterProcess, Orchestrator

root2 = Orchestrator()

# 1. coarse wide-range search (works even if the target is far away)
root2.add_child(SingleMapProcess(
	max_shift=700.0 * u.arcsec,
	bin_kernel=50.0 * u.arcsec,
	n_neighbors=48,
	min_corr=0.7,
))

# 2. refine in smaller space without binning
root2.add_child(SingleMapProcess(
	max_shift=100.0 * u.arcsec,
	bin_kernel=1.0 * u.arcsec,
	n_neighbors=48,
))

# 3. synthetic raster (no scale) to improve results
root2.add_child(SyntheticRasterProcess(
	max_shift=100.0 * u.arcsec,
	scale_step=0.00,
	n_neighbors=os.cpu_count()*2,
	plateau_iters=2,
))

# 4. recompute synthetic raster with more neighbors
root2.add_child(SyntheticRasterProcess(
	max_shift=100.0 * u.arcsec,
	scale_step=0.00,
	n_neighbors=os.cpu_count()*4,
	plateau_iters=2,
))

# 5. final scaling search in a tighter space (scale around 1.2 - 1.8 as example)
root2.add_child(SyntheticRasterProcess(
	max_shift=50.0 * u.arcsec,
	scale_step=0.01,
	n_neighbors=os.cpu_count()*10,
	plateau_iters=2,
))

root2.execute()
root2.cleanup()
```

Add a placeholder example artificial shifted raster and a protocol/visualization picture (or GIF) in the package `img/` folder. These images show the map before/after each step and can be used in tutorials.

CLI verbosity flags

- `-q`: quiet mode (equivalent to `verbose=0`)
- `-v`: verbose
- `-vv`: very verbose / debug

Script-style runs

You can call the same API from a small script. Example minimal script:

```python
from sunxcorr.single_map_process import run_single_map_minimal
run_single_map_minimal()
```

Before calling any API the README recommends reading the algorithm overview and verifying your FITS headers contain the time axis metadata described above.

---

<a name="troubleshooting"></a>
## Troubleshooting

- If `mypy` reports many import errors for legacy modules under `data_storage`, the repo config currently excludes those. You can un-ignore them selectively by editing `mypy.ini`.
- If `pytest` hangs during import, run the quick runner `Cross_correlation/tests/run_tests_quick.py` to avoid pytest's package-level collection.
- Numba: if functions fail to compile, verify `numba` is installed and matches the Python and `numpy` ABI used in your environment.

---

<a name="contribution-notes"></a>
## Contribution notes

- Prefer small, focused typing fixes. Use the typed-core + numba-wrapper pattern in `sunxcorr` when performance is critical.
- Add tests for any bugfix or new feature.

---

If you want, I can now:

- Expand any tutorial into a runnable notebook example.
- Add more examples showing JIT warm-up and profiling.

---

Created and maintained as part of the SunXCorr project.
