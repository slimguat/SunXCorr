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
  """Return all shift points and nearby offsets for a rectangular search grid."""
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
  """Return True when `point` is within the allowed shift bounds."""
  return (-shift_x <= point[0] <= shift_x) and (-shift_y <= point[1] <= shift_y)


def point_in_bounds_nd(point: Tuple[int, ...], axis_ranges: Sequence[int]) -> bool:
  """Return True when a multi-dimensional point lies within all bounds."""
  return all(-rng <= coord <= rng for coord, rng in zip(point, axis_ranges))


def clamp_point(point: Tuple[int, int], shift_x: int, shift_y: int) -> Tuple[int, int]:
  """Clamp `point` so it always remains within the search rectangle."""
  return (
    int(np.clip(point[0], -shift_x, shift_x)),
    int(np.clip(point[1], -shift_y, shift_y)),
  )


def clamp_point_nd(point: Tuple[int, ...], axis_ranges: Sequence[int]) -> Tuple[int, ...]:
  """Clamp a point in N-D space within the specified bounds."""
  return tuple(int(np.clip(coord, -rng, rng)) for coord, rng in zip(point, axis_ranges))


def gather_neighbors(
    center: Tuple[int, int],
    neighbor_offsets: List[Tuple[int, int]],
    cache: Mapping[Tuple[int, int], float],
    limit: int,
    shift_x: int,
    shift_y: int,
) -> List[Tuple[int, int]]:
  """Collect up to `limit` unseen neighbor points around `center`."""
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
  """Estimate a gradient-based direction from sampled correlation points."""
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
  """Estimate a gradient direction in N-D space."""
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
  """Return the closest point to `anchor` that has not been evaluated yet."""
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
  """Update plateau bookkeeping and return the new state tuple."""
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
  """Return a human-readable description of the current search phase."""
  if phase == 1:
    return "search phase"
  if phase == 2:
    return "plateau phase"
  return f"phase {phase}"
