"""Optimization functions for coalignment (shift and/or scale search)."""

# import heapq
import math
import hashlib
from collections import defaultdict
# from itertools import count
from typing import Any, Dict, List, Tuple, Optional#, Iterable
from multiprocessing import Queue #, Manager, Process
# from functools import partial

import numpy as np
from .utils import _vprint

from slimfunc_correlation_effort import correlation_for_params, correlation_for_params_jit, build_corr_context_jit
from coalign_helpers import (
    clamp_point,
    # clamp_point_nd,
    point_in_bounds,
    # point_in_bounds_nd,
    build_shift_structures,
    gather_neighbors,
    first_unvisited,
    compute_gradient_step,
    # compute_gradient_step_nd,
    update_phase_state,
)




def _make_payload_key(ref_img: np.ndarray, target_img: np.ndarray, center: Tuple[float, float] | None) -> str:
    """Generate a unique key for a worker payload based on data characteristics."""
    hasher = hashlib.sha256()
    hasher.update(ref_img.tobytes())
    hasher.update(target_img.tobytes())
    if center:
        hasher.update(str(center).encode())
    return hasher.hexdigest()[:16]


def _corr_worker_loop(
    task_queue: Queue,
    result_queue: Queue,
    shared_payloads: Dict[str, Dict[str, Any]],
) -> None:
    """
    Persistent worker process for parallel correlation evaluation.
    
    Runs in infinite loop, processing batches of correlation jobs until
    receiving None sentinel. Maintains cached payload to avoid repeated
    data loading.
    """
    current_payload_id: str | None = None
    ref_img: np.ndarray | None = None
    target_img: np.ndarray | None = None
    center: Tuple[float, float] | None = None
    ctx_jit = None

    def _load_payload(payload_id: str) -> bool:
        nonlocal current_payload_id, ref_img, target_img, center, ctx_jit
        if payload_id == current_payload_id and ref_img is not None and ctx_jit is not None:
            return True

        payload = shared_payloads.get(payload_id)
        if payload is None:
            return False

        ref_img = payload.get("ref_img")
        target_img = payload.get("target_img")
        center = payload.get("center")
        current_payload_id = payload_id

        if ref_img is None or target_img is None:
            ctx_jit = None
            return False

        ref_img = np.ascontiguousarray(ref_img, dtype=np.float64)
        target_img = np.ascontiguousarray(target_img, dtype=np.float64)

        ctx_jit = build_corr_context_jit(ref_img, target_img)

        # warm-up compile
        _ = correlation_for_params_jit(ref_img, target_img, 0.0, 0.0, 1.0, 1.0, center=center, ctx=ctx_jit)
        return True
    
    while True:
        task = task_queue.get()
        if task is None:  # Shutdown sentinel
            break
        
        job_id, payload_ids, dxs, dys, sxs, sys = task
        batch_results: List[Tuple[float, float, float]] = []
        
        for payload_id, dx, dy, sx, sy in zip(payload_ids, dxs, dys, sxs, sys):
            if payload_id != current_payload_id:
                if not _load_payload(payload_id):
                    continue
            
            if ref_img is None or target_img is None:
                continue
            
            corr_val = correlation_for_params_jit(
                ref_img, target_img, float(dx), float(dy), float(sx), float(sy),
                center=center, ctx=ctx_jit
            )
            batch_results.append((dx, dy, corr_val))
        
        result_queue.put((job_id, batch_results))


def optimize_shift_and_scale(
    target_data: np.ndarray,
    reference_data: np.ndarray,
    max_shift: int | Tuple[int, int],
    scale_step: float = 0.0,
    scale_range: Tuple[Tuple[float, float], Tuple[float, float]] | Tuple[float, float] | None = None,
    workers: Any = None,
    center_pix: Tuple[float, float] | None = None,
    n_neighbors: int = 50,
    max_corr: float = -1.0,
    corr_atol: float = 1e-4,
    corr_rtol: float = 1e-3,
    plateau_iters: int = 6,
    dx0: float = 0.0,
    dy0: float = 0.0,
    sx0: float | None = None,
    sy0: float | None = None,
    verbose: int = 0,
    debug_ctx: Any = None,
    task_queue: Optional[Queue] = None,
    result_queue: Optional[Queue] = None,
    shared_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, float], int, List[Dict[str, float]]]:
    """
    Optimize shift (and optionally scale) using persistent worker architecture.
    
    Workers, queues, and shared memory are managed externally and passed in.
    This allows workers to persist across multiple optimization calls.
    """
    # Detect search dimensionality
    is_4d = scale_step > 0 and scale_range is not None
    
    if isinstance(max_shift, tuple):
        shift_x, shift_y = max_shift
    else:
        shift_x, shift_y = max_shift, max_shift
    
    ref_img = reference_data.astype(np.float64)
    target_img = target_data.astype(np.float64)
    history: List[Dict[str, float]] = []
    
    force_plateau = max_corr < 0.0
    if force_plateau:
        max_corr = float('inf')
    
    # Use provided Manager infrastructure (workers already spawned externally)
    if task_queue is None or result_queue is None or shared_payloads is None:
        raise ValueError("task_queue, result_queue, and shared_payloads must be provided")
    
    # Store payload in shared memory
    payload_id = _make_payload_key(ref_img, target_img, center_pix)
    shared_payloads[payload_id] = {
        "ref_img": ref_img,
        "target_img": target_img,
        "center": center_pix,
    }
    
    # Determine number of workers from workers parameter
    n_workers = workers._processes if workers and hasattr(workers, '_processes') else 1
    _vprint(verbose, 2, f"Using {n_workers} persistent workers (managed externally)")
    
    next_job_id = 0
    
    if not is_4d:
        # 2D shift-only search
        all_points, neighbor_offsets = build_shift_structures(shift_x, shift_y)
        total_points = (2 * shift_x + 1) * (2 * shift_y + 1)
        
        _vprint(verbose, 2, f"Using {n_workers} workers with batch processing...")
        _vprint(verbose, 2, f"Total shift points: {total_points}...")
        
        cache: Dict[Tuple[int, int], float] = {}
        center_trace: List[Tuple[int, int]] = []
        iteration_number = 0
        
        def record_corr(dx: int, dy: int, corr_val: float) -> None:
            cache[(dx, dy)] = corr_val
            history.append({
                'iteration': iteration_number,
                'dx': float(dx),
                'dy': float(dy),
                'corr': float(corr_val),
            })
            if debug_ctx is not None:
                debug_ctx.add_point(int(dx), int(dy), corr_val)
        
        def evaluate_points(points: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
            nonlocal next_job_id
            results: List[Tuple[int, int, float]] = []
            uncached = []
            
            for dx, dy in points:
                if (dx, dy) in cache:
                    results.append((dx, dy, cache[(dx, dy)]))
                else:
                    uncached.append((dx, dy))
            
            if uncached:
                # Batch processing with persistent workers
                batch_size = max(1, int(math.ceil(len(uncached) / max(1, n_workers))))
                job_map: Dict[int, List[Tuple[int, int]]] = {}
                pending_jobs: set[int] = set()
                
                # Submit batches to workers
                for i in range(0, len(uncached), batch_size):
                    chunk = uncached[i:i + batch_size]
                    job_id = next_job_id
                    next_job_id += 1
                    
                    dxs = [float(pt[0]) for pt in chunk]
                    dys = [float(pt[1]) for pt in chunk]
                    sxs = [1.0] * len(chunk)
                    sys = [1.0] * len(chunk)
                    payload_ids = [payload_id] * len(chunk)
                    
                    task_queue.put((job_id, payload_ids, dxs, dys, sxs, sys))
                    pending_jobs.add(job_id)
                    job_map[job_id] = chunk
                
                # Collect results from workers
                while pending_jobs:
                    job_id, batch = result_queue.get()
                    pending_jobs.discard(job_id)
                    chunk_points = job_map.pop(job_id, [])
                    
                    for (dx, dy, corr_val), (point_dx, point_dy) in zip(batch, chunk_points):
                        record_corr(int(dx), int(dy), corr_val)
                        results.append((int(dx), int(dy), corr_val))
            
            return results
        
        # Initialize
        current_center = clamp_point((int(round(dx0)), int(round(dy0))), shift_x, shift_y)
        if current_center not in cache:
            evaluate_points([current_center])
        center_trace.append(current_center)
        
        best_point, best_corr = max(cache.items(), key=lambda item: item[1])
        global_best_corr = best_corr
        search_phase = 2 if force_plateau else 1
        plateau_counter = 0
        last_best_increase = best_corr
        stall_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Main search loop
        while True:
            if len(cache) >= total_points:
                _vprint(verbose, 2, "All shift positions evaluated; stopping.")
                break
            iteration_number += 1
            _vprint(verbose, 2, f"Iteration {iteration_number}: current center {current_center}, cache size {len(cache)}/{total_points}, phase {search_phase}...")
            
            neighbor_limit = min(n_neighbors, total_points - len(cache))
            new_points = gather_neighbors(
                current_center, neighbor_offsets, cache, neighbor_limit, shift_x, shift_y
            )
            evaluated = evaluate_points(new_points)
            
            local_points = [(current_center[0], current_center[1], cache[current_center])]
            local_points.extend(evaluated)
            
            if debug_ctx is not None:
                debug_ctx.render_iteration(current_center, search_phase)
            
            best_point, best_corr = max(cache.items(), key=lambda item: item[1])
            if best_corr > global_best_corr:
                global_best_corr = best_corr
            _vprint(verbose, 2, f"Best local corr: {cache[current_center]:.5f}, best overall corr: {global_best_corr:.5f} at shift {best_point}.")
            
            search_phase, plateau_counter, last_best_increase, plateau_met, phase_switched = update_phase_state(
                search_phase, best_corr, last_best_increase, plateau_counter,
                max_corr, corr_atol, corr_rtol, plateau_iters
            )
            if phase_switched and not force_plateau:
                _vprint(verbose, 2, f"Switching to plateau phase at corr={best_corr:.5f}")
            if plateau_met:
                _vprint(verbose, 2, "Plateau condition met; stopping.")
                break
            
            step_vec = compute_gradient_step(local_points, current_center, best_point)
            candidate_center = clamp_point(
                (current_center[0] + int(step_vec[0]), current_center[1] + int(step_vec[1])),
                shift_x, shift_y
            )
            
            while candidate_center in cache and candidate_center != current_center:
                stall_counts[candidate_center] += 1
                if stall_counts[candidate_center] >= 2:
                    fallback_local = first_unvisited(candidate_center, all_points, cache)
                    if fallback_local is not None:
                        candidate_center = fallback_local
                        break
                    break
                next_candidate = clamp_point(
                    (candidate_center[0] + int(step_vec[0]), candidate_center[1] + int(step_vec[1])),
                    shift_x, shift_y
                )
                if next_candidate == candidate_center:
                    break
                candidate_center = next_candidate
                if candidate_center == current_center or not point_in_bounds(candidate_center, shift_x, shift_y):
                    break
            
            if candidate_center == current_center or not point_in_bounds(candidate_center, shift_x, shift_y) or candidate_center in cache:
                fallback = first_unvisited(current_center, all_points, cache)
                if fallback is None:
                    break
                candidate_center = fallback
            
            current_center = candidate_center
            center_trace.append(current_center)
            if current_center not in cache:
                evaluate_points([current_center])
        
        # Workers remain alive for subsequent calls (managed externally)
        _vprint(verbose, 1, f"Best correlation found: {best_corr:.6f} at shift ({best_point[0]}, {best_point[1]}).")
        return {
            'dx': float(best_point[0]),
            'dy': float(best_point[1]),
            'corr': float(best_corr),
        }, iteration_number, history
    
    else:
        # 4D shift+scale search
        # Parse scale range
        if isinstance(scale_range, tuple) and len(scale_range) == 2:
            if isinstance(scale_range[0], tuple):
                # ((sx_min, sx_max), (sy_min, sy_max))
                sx_min, sx_max = scale_range[0]
                sy_min, sy_max = scale_range[1]
            else:
                # (s_min, s_max) - same for both axes
                sx_min, sx_max = scale_range
                sy_min, sy_max = scale_range
        else:
            raise ValueError(f"Invalid scale_range: {scale_range}")
        
        # Build 4D grid
        sx_vals = np.arange(sx_min, sx_max + scale_step/2, scale_step)
        sy_vals = np.arange(sy_min, sy_max + scale_step/2, scale_step)
        dx_vals = list(range(-shift_x, shift_x + 1))
        dy_vals = list(range(-shift_y, shift_y + 1))
        
        total_points = len(dx_vals) * len(dy_vals) * len(sx_vals) * len(sy_vals)
        _vprint(verbose, 2, f"4D search: {len(dx_vals)}×{len(dy_vals)}×{len(sx_vals)}×{len(sy_vals)} = {total_points} points")
        _vprint(verbose, 2, f"Using {n_workers} workers with batch processing")
        
        cache: Dict[Tuple[int, int, float, float], float] = {}
        iteration_number = 0
        
        def record_corr(dx: int, dy: int, sx: float, sy: float, corr_val: float) -> None:
            cache[(dx, dy, sx, sy)] = corr_val
            history.append({
                'iteration': iteration_number,
                'dx': float(dx),
                'dy': float(dy),
                'sx': float(sx),
                'sy': float(sy),
                'corr': float(corr_val),
            })
            if debug_ctx is not None:
                debug_ctx.add_point(int(dx), int(dy), corr_val)
        
        def evaluate_points(points: List[Tuple[int, int, float, float]]) -> List[Tuple[int, int, float, float, float]]:
            nonlocal next_job_id
            results: List[Tuple[int, int, float, float, float]] = []
            uncached = []
            
            for dx, dy, sx, sy in points:
                if (dx, dy, sx, sy) in cache:
                    results.append((dx, dy, sx, sy, cache[(dx, dy, sx, sy)]))
                else:
                    uncached.append((dx, dy, sx, sy))
            
            if uncached:
                # Batch processing with persistent workers
                batch_size = max(1, int(math.ceil(len(uncached) / max(1, n_workers))))
                job_map: Dict[int, List[Tuple[int, int, float, float]]] = {}
                pending_jobs: set[int] = set()
                
                # Submit batches to workers
                for i in range(0, len(uncached), batch_size):
                    chunk = uncached[i:i + batch_size]
                    job_id = next_job_id
                    next_job_id += 1
                    
                    dxs = [float(pt[0]) for pt in chunk]
                    dys = [float(pt[1]) for pt in chunk]
                    sxs = [float(pt[2]) for pt in chunk]
                    sys = [float(pt[3]) for pt in chunk]
                    payload_ids = [payload_id] * len(chunk)
                    
                    task_queue.put((job_id, payload_ids, dxs, dys, sxs, sys))
                    pending_jobs.add(job_id)
                    job_map[job_id] = chunk
                
                # Collect results from workers
                while pending_jobs:
                    job_id, batch = result_queue.get()
                    pending_jobs.discard(job_id)
                    chunk_points = job_map.pop(job_id, [])
                    
                    for (dx, dy, corr_val), (point_dx, point_dy, point_sx, point_sy) in zip(batch, chunk_points):
                        record_corr(int(dx), int(dy), float(point_sx), float(point_sy), corr_val)
                        results.append((int(dx), int(dy), float(point_sx), float(point_sy), corr_val))
            
            return results
        
        # Initialize from starting point
        current_center = (
            clamp_point((int(round(dx0)), int(round(dy0))), shift_x, shift_y),
            (float(sx0) if sx0 is not None else 1.0, float(sy0) if sy0 is not None else 1.0)
        )
        
        start_dx, start_dy = current_center[0]
        start_sx, start_sy = current_center[1]
        if (start_dx, start_dy, start_sx, start_sy) not in cache:
            evaluate_points([(start_dx, start_dy, start_sx, start_sy)])
        
        best_key = max(cache.items(), key=lambda item: item[1])[0]
        best_corr = cache[best_key]
        global_best_corr = best_corr
        search_phase = 2 if force_plateau else 1
        plateau_counter = 0
        last_best_increase = best_corr
        
        # Main search loop - simple greedy neighbor search
        while True:
            if len(cache) >= total_points:
                _vprint(verbose, 2, "All 4D positions evaluated; stopping.")
                break
            
            iteration_number += 1
            _vprint(verbose, 2, f"Iteration {iteration_number}: center {current_center}, cache {len(cache)}/{total_points}, phase {search_phase}")
            
            # Generate neighbors in 4D space
            center_dx, center_dy = current_center[0]
            center_sx, center_sy = current_center[1]
            
            neighbors = []
            for ddx in [-1, 0, 1]:
                for ddy in [-1, 0, 1]:
                    for dsx in [-scale_step, 0, scale_step]:
                        for dsy in [-scale_step, 0, scale_step]:
                            if ddx == 0 and ddy == 0 and dsx == 0 and dsy == 0:
                                continue
                            
                            new_dx = center_dx + ddx
                            new_dy = center_dy + ddy
                            new_sx = center_sx + dsx
                            new_sy = center_sy + dsy
                            
                            if (new_dx < -shift_x or new_dx > shift_x or
                                new_dy < -shift_y or new_dy > shift_y or
                                new_sx < sx_min or new_sx > sx_max or
                                new_sy < sy_min or new_sy > sy_max):
                                continue
                            
                            if (new_dx, new_dy, new_sx, new_sy) not in cache:
                                neighbors.append((new_dx, new_dy, new_sx, new_sy))
            
            # Limit neighbors
            if len(neighbors) > n_neighbors:
                neighbors = neighbors[:n_neighbors]
            
            if not neighbors:
                _vprint(verbose, 2, "No more neighbors; stopping.")
                break
            
            evaluated = evaluate_points(neighbors)
            
            if debug_ctx is not None:
                debug_ctx.render_iteration((center_dx, center_dy), search_phase)
            
            # Find best
            best_key = max(cache.items(), key=lambda item: item[1])[0]
            best_corr = cache[best_key]
            if best_corr > global_best_corr:
                global_best_corr = best_corr
            
            _vprint(verbose, 2, f"Best corr: {best_corr:.5f} at {best_key}")
            
            # Update phase
            search_phase, plateau_counter, last_best_increase, plateau_met, phase_switched = update_phase_state(
                search_phase, best_corr, last_best_increase, plateau_counter,
                max_corr, corr_atol, corr_rtol, plateau_iters
            )
            if plateau_met:
                _vprint(verbose, 2, "Plateau met; stopping.")
                break
            
            # Move to best point
            current_center = ((best_key[0], best_key[1]), (best_key[2], best_key[3]))
        
        # Workers remain alive for subsequent calls (managed externally)
        _vprint(verbose, 1, f"Best correlation: {best_corr:.6f} at shift=({best_key[0]}, {best_key[1]}), scale=({best_key[2]:.3f}, {best_key[3]:.3f})")
        return {
            'dx': float(best_key[0]),
            'dy': float(best_key[1]),
            'sx': float(best_key[2]),
            'sy': float(best_key[3]),
            'corr': float(best_corr),
        }, iteration_number, history
