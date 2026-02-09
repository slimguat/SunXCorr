"""Geometry, search, and plateau helpers for the coalignment solver.

These functions encapsulate math-heavy logic used by `Coaligner` so the main
class can focus on orchestration. Each helper has a focused responsibility and
is documented for reuse.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

import numpy as np


def build_shift_structures(
    shift_x: int,
    shift_y: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
  """Generate all search grid points and sorted neighbor offsets.
  
  Creates a complete rectangular search grid and an ordered list of relative
  offsets for neighbor exploration during optimization.
  
  Parameters
  ----------
  shift_x : int
      Maximum shift in x-direction (pixels)
  shift_y : int
      Maximum shift in y-direction (pixels)
  
  Returns
  -------
  all_pts : List[Tuple[int, int]]
      All (dx, dy) points in the search grid, flattened from meshgrid
  offsets : List[Tuple[int, int]]
      Relative offsets sorted by distance, excluding (0, 0)
  
  Notes
  -----
  The offsets list is sorted by Euclidean distance, with ties broken by
  Manhattan distance. This ordering ensures neighbors are explored in a
  sensible radial pattern from any center point.
  
  Examples
  --------
  >>> all_pts, offsets = build_shift_structures(2, 2)
  >>> len(all_pts)
  25
  >>> offsets[0:3]  # Closest neighbors first
  [(1, 0), (0, 1), (-1, 0)]
  """
  x_steps, y_steps = np.meshgrid(
    np.arange(-shift_x, shift_x + 1),
    np.arange(-shift_y, shift_y + 1),
    indexing="xy",
  )
  grid = np.array([x_steps.flatten(), y_steps.flatten()], dtype=np.int64).T
  all_pts = [(int(dx), int(dy)) for dx, dy in grid]
  offsets = [
    (dx, dy)
    for dx in range(-shift_x, shift_x + 1)
    for dy in range(-shift_y, shift_y + 1)
    if not (dx == 0 and dy == 0)
  ]
  offsets.sort(key=lambda xy: (xy[0] * xy[0] + xy[1] * xy[1], abs(xy[0]) + abs(xy[1])))
  return all_pts, offsets


def point_in_bounds(point: Tuple[int, int], shift_x: int, shift_y: int) -> bool:
  """Check if a 2D point lies within the rectangular search bounds.
  
  Parameters
  ----------
  point : Tuple[int, int]
      Point coordinates (dx, dy)
  shift_x : int
      Maximum allowed x-shift
  shift_y : int
      Maximum allowed y-shift
  
  Returns
  -------
  bool
      True if point is within [-shift_x, shift_x] × [-shift_y, shift_y]
  """
  return (-shift_x <= point[0] <= shift_x) and (-shift_y <= point[1] <= shift_y)


def point_in_bounds_nd(point: Tuple[int, ...], axis_ranges: Sequence[int]) -> bool:
  """Check if an N-dimensional point lies within hypercube bounds.
  
  Parameters
  ----------
  point : Tuple[int, ...]
      N-dimensional point coordinates
  axis_ranges : Sequence[int]
      Maximum range along each axis
  
  Returns
  -------
  bool
      True if all coordinates satisfy -range ≤ coord ≤ range
  """
  return all(-rng <= coord <= rng for coord, rng in zip(point, axis_ranges))


def clamp_point(point: Tuple[int, int], shift_x: int, shift_y: int) -> Tuple[int, int]:
  """Clamp a 2D point to lie within rectangular search bounds.
  
  Parameters
  ----------
  point : Tuple[int, int]
      Input point coordinates (dx, dy)
  shift_x : int
      Maximum allowed x-shift
  shift_y : int
      Maximum allowed y-shift
  
  Returns
  -------
  Tuple[int, int]
      Clamped coordinates within [-shift_x, shift_x] × [-shift_y, shift_y]
  """
  return (
    int(np.clip(point[0], -shift_x, shift_x)),
    int(np.clip(point[1], -shift_y, shift_y)),
  )


def clamp_point_nd(point: Tuple[int, ...], axis_ranges: Sequence[int]) -> Tuple[int, ...]:
  """Clamp an N-dimensional point within hypercube bounds.
  
  Parameters
  ----------
  point : Tuple[int, ...]
      N-dimensional point coordinates
  axis_ranges : Sequence[int]
      Maximum range along each axis
  
  Returns
  -------
  Tuple[int, ...]
      Clamped coordinates with each component within [-range, range]
  """
  return tuple(int(np.clip(coord, -rng, rng)) for coord, rng in zip(point, axis_ranges))


def gather_neighbors(
    center: Tuple[int, int],
    neighbor_offsets: List[Tuple[int, int]],
    cache: Mapping[Tuple[int, int], float],
    limit: int,
    shift_x: int,
    shift_y: int,
) -> List[Tuple[int, int]]:
  """Collect up to `limit` unevaluated neighbor points around a center.
  
  Parameters
  ----------
  center : Tuple[int, int]
      Current center point (dx, dy)
  neighbor_offsets : List[Tuple[int, int]]
      Ordered list of relative offsets to try
  cache : Mapping[Tuple[int, int], float]
      Dictionary of already-evaluated points
  limit : int
      Maximum number of neighbors to return
  shift_x : int
      Maximum x-shift for bounds checking
  shift_y : int
      Maximum y-shift for bounds checking
  
  Returns
  -------
  List[Tuple[int, int]]
      Up to `limit` valid, unevaluated neighbor coordinates
  
  Notes
  -----
  Neighbors are selected in the order given by `neighbor_offsets`. Points
  outside bounds or already in cache are skipped. Returns an empty list if
  limit ≤ 0.
  """
  if limit <= 0:
    return []
  neighbors: List[Tuple[int, int]] = []
  for dx, dy in neighbor_offsets:
    if len(neighbors) >= limit:
      break
    candidate = (center[0] + dx, center[1] + dy)
    if not point_in_bounds(candidate, shift_x, shift_y):
      continue
    if candidate in cache:
      continue
    neighbors.append(candidate)
  return neighbors


def compute_gradient_step(
    local_points: Sequence[Tuple[int, int, float]],
    current_center: Tuple[int, int],
    best_point: Tuple[int, int],
) -> Tuple[float, float]:
  """Estimate a gradient-based search direction from correlation samples.
  
  Computes a weighted gradient direction based on local correlation values,
  prioritizing directions toward improved correlation.
  
  Parameters
  ----------
  local_points : Sequence[Tuple[int, int, float]]
      List of (dx, dy, correlation) samples
  current_center : Tuple[int, int]
      Current optimizer position
  best_point : Tuple[int, int]
      Best point found in this iteration
  
  Returns
  -------
  Tuple[float, float]
      Suggested (step_x, step_y) direction in continuous coordinates
  
  Notes
  -----
  If best_point differs from current_center, returns the direct vector.
  Otherwise, computes a weighted average of directions toward points with
  higher correlation, using correlation improvements as weights.
  """
  if best_point != current_center:
    return (
      float(best_point[0] - current_center[0]),
      float(best_point[1] - current_center[1]),
    )

  center_corr = None
  for px, py, corr in local_points:
    if (px, py) == current_center:
      center_corr = corr
      break
  if center_corr is None:
    return (0.0, 0.0)

  grad_x = 0.0
  grad_y = 0.0
  weight_sum = 0.0
  for px, py, corr in local_points:
    if (px, py) == current_center:
      continue
    delta = corr - center_corr
    if delta <= 0:
      continue
    grad_x += (px - current_center[0]) * delta
    grad_y += (py - current_center[1]) * delta
    weight_sum += delta

  if weight_sum == 0.0:
    return (0.0, 0.0)
  return (grad_x / weight_sum, grad_y / weight_sum)


def compute_gradient_step_nd(
    local_points: Sequence[Tuple[Tuple[int, ...], float]],
    current_center: Tuple[int, ...],
    best_point: Tuple[int, ...],
) -> Tuple[float, ...]:
  """Estimate a gradient direction in N-dimensional parameter space.
  
  Generalization of compute_gradient_step to arbitrary dimensions.
  
  Parameters
  ----------
  local_points : Sequence[Tuple[Tuple[int, ...], float]]
      List of (point, correlation) tuples where point is N-dimensional
  current_center : Tuple[int, ...]
      Current N-dimensional center position
  best_point : Tuple[int, ...]
      Best N-dimensional point found
  
  Returns
  -------
  Tuple[float, ...]
      N-dimensional gradient step direction
  
  Notes
  -----
  Uses the same weighted-average logic as compute_gradient_step, but
  operates on points in N-dimensional space.
  """
  if best_point != current_center:
    return tuple(float(best_point[idx] - current_center[idx]) for idx in range(len(current_center)))

  center_corr = None
  for point, corr in local_points:
    if point == current_center:
      center_corr = corr
      break
  if center_corr is None:
    return tuple(0.0 for _ in current_center)

  grad = np.zeros(len(current_center), dtype=float)
  weight_sum = 0.0
  for point, corr in local_points:
    if point == current_center:
      continue
    delta = corr - center_corr
    if delta <= 0:
      continue
    diff = np.array(point, dtype=float) - np.array(current_center, dtype=float)
    grad += diff * delta
    weight_sum += delta

  if weight_sum == 0.0:
    return tuple(0.0 for _ in current_center)
  step = grad / weight_sum
  return tuple(float(component) for component in step)


def first_unvisited(
    anchor: Tuple[int, int],
    all_points: Sequence[Tuple[int, int]],
    cache: Mapping[Tuple[int, int], float],
) -> Tuple[int, int] | None:
  """Find the closest unevaluated point to an anchor position.
  
  Parameters
  ----------
  anchor : Tuple[int, int]
      Reference point for distance calculation
  all_points : Sequence[Tuple[int, int]]
      Complete list of candidate points
  cache : Mapping[Tuple[int, int], float]
      Dictionary of already-evaluated points
  
  Returns
  -------
  Tuple[int, int] or None
      Closest unevaluated point, or None if all points are in cache
  
  Notes
  -----
  Points are sorted by Manhattan distance first, then by Euclidean distance
  squared as a tiebreaker. This ensures consistent ordering and prioritizes
  axis-aligned exploration.
  """
  ordered_points = sorted(
    all_points,
    key=lambda pt: (
      abs(pt[0] - anchor[0]) + abs(pt[1] - anchor[1]),
      pt[0] * pt[0] + pt[1] * pt[1],
    ),
  )
  for pt in ordered_points:
    if pt not in cache:
      return pt
  return None


def update_phase_state(
    phase: int,
    best_corr: float,
    last_best_increase: float,
    plateau_counter: int,
    max_corr_goal: float,
    corr_atol: float,
    corr_rtol: float,
    plateau_iters: int,
) -> Tuple[int, int, float, bool, bool]:
  """Update optimization phase based on convergence criteria.
  
  Tracks correlation improvements and plateau detection, transitioning from
  search phase (1) to plateau phase (2) when appropriate.
  
  Parameters
  ----------
  phase : int
      Current phase (1=search, 2=plateau)
  best_corr : float
      Current best correlation value
  last_best_increase : float
      Correlation at last significant improvement
  plateau_counter : int
      Number of iterations without significant improvement
  max_corr_goal : float
      Target correlation to reach before phase transition
  corr_atol : float
      Absolute tolerance for correlation improvement
  corr_rtol : float
      Relative tolerance for correlation improvement
  plateau_iters : int
      Number of plateau iterations before phase change
  
  Returns
  -------
  phase : int
      Updated phase number
  plateau_counter : int
      Updated plateau iteration counter
  last_best_increase : float
      Updated correlation at last improvement
  plateau_met : bool
      True if plateau convergence criteria are met
  phase_switched : bool
      True if phase transition occurred
  
  Notes
  -----
  A significant improvement is defined as exceeding max(corr_atol,
  |last_best| * corr_rtol). Phase transitions occur when:
  - Search phase reaches max_corr_goal
  - Plateau counter exceeds plateau_iters threshold
  """
  plateau_met = False
  phase_switched = False

  corr_delta = best_corr - last_best_increase
  improvement_needed = max(corr_atol, abs(last_best_increase) * corr_rtol)
  if corr_delta > improvement_needed:
    plateau_counter = 0
    last_best_increase = best_corr
  else:
    plateau_counter += 1
    if plateau_counter >= plateau_iters:
      if phase == 1:
        phase = 2
        plateau_counter = 0
        phase_switched = True
      else:
        plateau_met = True

  goal_reached = best_corr >= max_corr_goal
  if goal_reached and phase == 1:
    phase = 2
    plateau_counter = 0
    phase_switched = True
  elif goal_reached and phase == 2 and plateau_counter >= plateau_iters:
    plateau_met = True

  return phase, plateau_counter, last_best_increase, plateau_met, phase_switched


def phase_label(phase: int) -> str:
  """Return a human-readable label for the optimization phase.
  
  Parameters
  ----------
  phase : int
      Phase number (1 or 2)
  
  Returns
  -------
  str
      "search phase" for phase 1, "plateau phase" for phase 2,
      or "phase {phase}" for other values
  """
  if phase == 1:
    return "search phase"
  if phase == 2:
    return "plateau phase"
  return f"phase {phase}"
