# Coaligner Reference

`Coaligner` wraps the full workflow needed to co-align a target SunPy map onto a reference map. It handles field-of-view padding, binning, reprojection, and an adaptive cross-correlation search that can reuse a persistent multiprocessing pool across multiple alignment attempts.

## High-Level Responsibilities
- **Data preparation (`prepare_binned_data`)** enlarges and smooths the target map, reprojects the reference onto the same extended grid, and stores the coarse (binned) arrays along with WCS metadata in `xcorr_binned_kwargs`.
- **Cross-correlation search (`_run_shift_search`)** runs a guided exploration over integer pixel shifts for either the binned or one-map phase. Convenience wrappers (`run_binned_cross_correlation`, `run_one_map_cross_correlation`, `run_global_cross_correlation`) prepare data and invoke the shared routine, while `xcorr_to_value` remains as a compatibility shim for the original API.
- **Worker management (`_ensure_corr_workers` / `_shutdown_corr_workers`)** maintains a shared pool of `_corr_worker_loop` processes that evaluate correlation candidates without reloading the image data on each call.
- **Lifecycle helpers (`close`, `__del__`)** guarantee that any live workers are terminated cleanly when the aligner is no longer needed.

## Persistent Worker Pool
- The instance owns `_task_queue`, `_result_queue`, and `_corr_workers`. `_ensure_corr_workers` lazily spins up `n_jobs` processes the first time a correlation routine requests multiprocessing, caching a payload key so the pool is reused only while the same `(ref_img, target_img, center)` are valid.
- Each worker runs `_corr_worker_loop`, holding the image data in memory and blocking on `_task_queue` for `(job_id, dx, dy, sx, sy)` messages. Results are pushed back through `_result_queue`.
- Workers survive across multiple invocations (and even across future cross-correlation variants) until `close()` or `_shutdown_corr_workers()` is called, at which point `None` sentinels are broadcast and the processes are joined.

## Cross-Correlation Search (`_run_shift_search`)

### Inputs and Runtime Configuration
- Consumes the arrays and metadata staged in the relevant kwargs dictionary (`xcorr_binned_kwargs` or `xcorr_one_map_kwargs`).
- Reads shift bounds, initial guess, neighbor budget, convergence targets (`max_corr`, `corr_atol`, `corr_rtol`, `plateau_iters`), and the requested worker count.
- Establishes the interpolation center (`binned_crpix1/2`) when available, then calls `_ensure_corr_workers` so the multiprocessing pool is ready to receive tasks.

### Shift Grid Construction
- Generates the full integer lattice defined by `shift_range` via `np.meshgrid` and stores both `all_points` (for fallbacks) and a distance-ranked `neighbor_offsets` list used to sample nearby candidates first.

### Helper Utilities
- `in_bounds` / `clamp_point` keep shifts inside the legal search window.
- `record_corr` centralizes cache/history updates so each new evaluation is logged exactly once.
- `evaluate_points` fan-outs uncached points: it returns cached correlations immediately, batches the rest to the worker pool (or computes serially), and rehydrates the `(dx, dy, corr)` triples in request order.
- `gather_neighbors` finds up to `n_neighbors` unseen neighbors around the current center, while `first_unvisited` falls back to the nearest unexplored point relative to the present center when the gradient stalls.
- `compute_gradient_step` derives a gradient-like step from the weighted local correlations, defaulting toward the global best if the gradient collapses.

### Search Loop
- Seeds the cache with the initial shift (`dx0`, `dy0`), then iteratively:
  1. Evaluates neighbor batches (respecting the total grid size to avoid over-sampling).
  2. Updates the local/global best correlations and phases.
  3. Detects plateaus using the absolute/relative tolerance pair; repeated stagnation triggers termination once `plateau_iters` consecutive flat steps occur.
  4. Chooses the next center via the gradient step, clamping within bounds and skipping shifts that already exist in the cache. If movement is blocked, the fallback locator elects the closest unvisited point.
- The loop exits when the plateau condition is satisfied or when every grid point has been evaluated.

### Result Packaging
- Persists the outcome under `self.results["binning"]`, logging the best shift, per-evaluation history, visited centers, sample counts, and the final phase (`search` vs `plateau`).
- Marks `self.procedures["binning"] = True` and prints a summary when `verbose` is enabled.
- Worker cleanup is explicit: call `coaligner.close()` once all cross-correlation work (current and upcoming) is complete to release the persistent processes.
