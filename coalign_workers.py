"""Worker-process helpers for distributing correlation evaluations."""

from __future__ import annotations

import datetime
import random
from multiprocessing import Queue
from typing import Any, Dict, Tuple

import numpy as np

from slimfunc_correlation_effort import correlation_for_params


def make_worker_payload_key(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    center: Tuple[float, float] | None,
) -> Tuple[Any, ...]:
  """Build a tuple that uniquely identifies the payload shared with workers."""
  center_key: Tuple[float, float] | None = None
  if center is not None:
    center_key = (float(center[0]), float(center[1]))
  return (
    ref_img.shape,
    str(ref_img.dtype),
    int(ref_img.ctypes.data),
    target_img.shape,
    str(target_img.dtype),
    int(target_img.ctypes.data),
    center_key,
  )


def generate_payload_id() -> str:
  """Return a globally unique payload identifier based on time and randomness."""
  timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
  suffix = f"{random.randint(0, 999_999_999):09d}"
  return f"{timestamp}-{suffix}"


def store_shared_payload(
  shared_payloads: Dict[str, Dict[str, Any]] | None,
    current_payload_id: str | None,
    ref_img: np.ndarray,
    target_img: np.ndarray,
    center: Tuple[float, float] | None,
) -> str | None:
  """Store payload in shared memory and return the identifier used."""
  if shared_payloads is None:
    return None
  shared_dict = shared_payloads  # typed alias so mypy sees mutations
  if current_payload_id is not None:
    shared_dict.pop(current_payload_id, None)
  payload_id = generate_payload_id()
  shared_dict[payload_id] = {
    "ref_img": ref_img,
    "target_img": target_img,
    "center": center,
  }
  return payload_id


def corr_worker_loop(
  task_queue: Queue,
  result_queue: Queue,
  shared_payloads: Dict[str, Dict[str, Any]] | None,
  initial_payload_id: str | None,
) -> None:
  """Process queued correlation jobs until a sentinel `None` task is received."""
  current_payload_id: str | None = None
  ref_img: np.ndarray | None = None
  target_img: np.ndarray | None = None
  center: Tuple[float, float] | None = None

  def _load_payload(payload_id: str | None) -> bool:
    nonlocal current_payload_id, ref_img, target_img, center
    if payload_id is None or shared_payloads is None:
      return False
    if payload_id == current_payload_id:
      return ref_img is not None and target_img is not None
    payload = shared_payloads.get(payload_id)
    if payload is None:
      return False
    ref_img = payload.get("ref_img")
    target_img = payload.get("target_img")
    center = payload.get("center")
    current_payload_id = payload_id
    return ref_img is not None and target_img is not None

  if initial_payload_id:
    _load_payload(initial_payload_id)

  while True:
    task = task_queue.get()
    if task is None:
      break
    job_id, payload_ids, dxs, dys, sxs, sys = task
    batch_results: list[tuple[int, int, float]] = []
    for payload_id, dx, dy, sx, sy in zip(payload_ids, dxs, dys, sxs, sys):
      if payload_id != current_payload_id:
        if not _load_payload(payload_id):
          continue
      if ref_img is None or target_img is None:
        continue
      corr_val = correlation_for_params(
        ref_img,
        target_img,
        float(dx),
        float(dy),
        float(sx),
        float(sy),
        center=center,
      )
      batch_results.append((int(round(dx)), int(round(dy)), float(corr_val)))
    result_queue.put((job_id, batch_results))
