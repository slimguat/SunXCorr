# SunXCorr

Fast solar image cross-correlation and alignment library.

## Installation

Clone the repository and install with pip:

```
pip install .
```

````markdown
# SunXCorr

Fast solar image cross-correlation and alignment library.

This README provides developer-oriented instructions to run the full test-suite, work with the built-in synthetic test helpers (single synthetic runs), operate the `orchestrator` for multi-frame sequences, and use common developer tooling (linters and static checks). It also explains what a Pull Request (PR) is and the recommended workflow when contributing.

---

## Installation

Clone the repository and install with pip inside a virtual environment (recommended):

```bash
git clone <your-repo-url>
cd Cross_correlation
python -m venv env
source env/bin/activate
pip install -e .
```

If you have the project-specific environment layout, you can use the provided env: `/home/smzergua/workshop/2026/envs/SAFFRON_ENV` and call the interpreter directly.

## Requirements
- numpy
- scipy
- numba (used on hot code paths for performance)
- astropy
- sunpy
- matplotlib

Install the developer tools used in this repo (formatters and type checkers):

```bash
/home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/python -m pip install --upgrade pip
/home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/python -m pip install black isort mypy pytest
```

---

## Running Tests

All tests are executed with `pytest`. It is important to run tests from the repository root so imports resolve correctly.

Run the full test-suite (quiet):

```bash
cd Cross_correlation
/home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/python -m pytest -q
```

Run a single test file or test function to speed up iteration:

```bash
/home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/python -m pytest tests/test_synthetic_raster_process.py::test_synthetic_raster_process_short_sequence -q
```

Notes:
- `pytest` will create a `.pytest_cache/` directory — this is normal and is ignored by `.gitignore`.
- If a test appears to hang in multiprocessing code, enable worker debug logging by setting `SUNXCORR_DEBUG_LOG=1` before running the test; this will write diagnostic output to `/tmp/sunxcorr_worker_debug.log` (opt-in only).

---

## Single Synthetic Test — Quick Tutorial

This project includes utilities used by the test-suite to generate synthetic frames and validate correlation/coalignment behavior. Use the following recipe to run a minimal single synthetic example:

1. Activate the environment and change to the project directory:

```bash
source /home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/activate
cd Cross_correlation
```

2. Run a minimal pipeline from Python (example):

```python
from sunxcorr.single_map_process import run_single_map_minimal
run_single_map_minimal()
```

3. If you prefer a script-style run, wrap the same call in a small script and run it via the venv python.

Useful test references:
- `tests/test_single_map_process.py` — shows a minimal single-map pipeline used by CI tests.
- `tests/test_synthetic_raster_process.py` — integration test that builds a short synthetic sequence and runs the orchestrator over it.

---

## Using the `orchestrator` for multi-frame synthetic sequences

The `orchestrator` is a convenience layer that dispatches the single-map processing across sequences and supports a persistent worker pool that reduces Numba warm-up overhead.

Typical usage (high-level):

```python
from sunxcorr.orchestrator import Orchestrator

frames = [...]  # list of synthetic frames or astropy Maps
orch = Orchestrator(n_workers=4)
orch.run_sequence(frames)
```

Notes:
- `Orchestrator` can be configured with `n_workers` to control the number of persistent worker processes.
- Persistent workers keep shared payloads (raw arrays) in a Manager-backed dict so they avoid re-sending large arrays repeatedly.
- Worker debug logging is opt-in via `SUNXCORR_DEBUG_LOG`.

---

## Developer tooling: linters and static checks

What are linters and why run them?

- Linters and formatters automatically check and/or reformat code to maintain consistency and catch common issues early. They are not tests but they improve code quality and reduce review friction.

Common tools used with this project:
- `black` — formats Python code to a consistent style.
- `isort` — sorts and groups imports.
- `mypy` — optional static type checker for Python type hints.
- `flake8` — additional linting rules (optional).

Commands to run them (using the project environment):

```bash
/home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/python -m isort . --profile black
/home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/python -m black .
/home/smzergua/workshop/2026/envs/SAFFRON_ENV/bin/python -m mypy -p sunxcorr --ignore-missing-imports
```

Tips:
- Run formatters before committing to keep diffs small and consistent.
- `mypy` will create `.mypy_cache/` — this is ignored in the repo.

---

## What is a PR (Pull Request)?

- A Pull Request (PR) is how you propose changes to be merged into the main repository (on platforms like GitHub or GitLab). Typical workflow:

1. Create a branch: `git checkout -b feat/your-feature`.
2. Make changes, run tests and linters locally.
3. Commit and push your branch: `git push origin feat/your-feature`.
4. Open a PR on the remote and request reviewers.
5. Address comments and merge once approved.

Best practices for PRs in this repository:
- Keep PRs focused on a single concern.
- Ensure `pytest` and `mypy` pass locally before opening the PR.
- Include a short description of what changed and why in the PR body.

---

## Troubleshooting

- If tests fail in CI but pass locally, check environment differences (Python version, installed packages).
- If the worker processes hang or behave unexpectedly, enable `SUNXCORR_DEBUG_LOG=1` and inspect `/tmp/sunxcorr_worker_debug.log` for diagnostic messages.
- Remove stale caches if needed:

```bash
rm -rf .pytest_cache .mypy_cache
```

---

## Next steps (optional)

- I can add short example notebooks demonstrating a single synthetic run and an orchestrator sequence.
- I can also add a `docs/` page with the same content and a small CI workflow example.

If you want either of those, tell me which and I'll add them.

Author: Slimane MZERGUAT

````
