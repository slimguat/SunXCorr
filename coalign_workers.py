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
  """Create a unique identifier tuple for worker payload data.
  
  Builds a hashable key from array metadata (shape, dtype, memory address)
  and center coordinates. Used to detect when payload data changes and
  workers need to reload their shared memory references.
  
  Parameters
  ----------
  ref_img : np.ndarray
      Reference image array
  target_img : np.ndarray
      Target image array
  center : Tuple[float, float] or None
      Optional center coordinates for correlation calculation
  
  Returns
  -------
  Tuple[Any, ...]
      Hashable key containing (ref_shape, ref_dtype, ref_address,
      target_shape, target_dtype, target_address, center_tuple)
  
  Notes
  -----
  The memory address (from ctypes.data) is used for identity checking,
  not for direct memory access. This allows workers to detect when the
  underlying data has changed even if shape and dtype remain the same.
  """
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
  """Generate a globally unique identifier for shared payload data.
  
  Creates a unique string combining UTC timestamp (microsecond precision)
  and a random suffix to ensure uniqueness even with concurrent processes.
  
  Returns
  -------
  str
      Unique payload identifier in format "YYYYMMDDHHMMSSffffff-RRRRRRRRR"
      where ffffff is microseconds and RRRRRRRRR is a 9-digit random number
  
  Examples
  --------
  >>> payload_id = generate_payload_id()
  >>> len(payload_id)
  30
  >>> payload_id.split('-')  # doctest: +SKIP
  ['20260206153045123456', '042837561']
  """
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
  """Store image payload in shared memory and return the identifier.
  
  Removes the old payload (if any) and stores new image data with a unique
  identifier for worker processes to access.
  
  Parameters
  ----------
  shared_payloads : Dict[str, Dict[str, Any]] or None
      Shared dictionary mapping payload IDs to data dictionaries.
      If None, shared memory is disabled and function returns None.
  current_payload_id : str or None
      ID of current payload to remove before storing new one
  ref_img : np.ndarray
      Reference image to share
  target_img : np.ndarray
      Target image to share
  center : Tuple[float, float] or None
      Optional center coordinates
  
  Returns
  -------
  str or None
      New payload ID if successful, None if shared_payloads is None
  
  Notes
  -----
  The payload dictionary contains keys 'ref_img', 'target_img', and 'center'.
  Old payloads are removed to prevent unbounded memory growth.
  """
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
  """Worker process loop for parallel correlation evaluation.
  
  Continuously processes correlation tasks from a queue until receiving a
  sentinel None value. Each task evaluates correlation at multiple (dx, dy)
  shift positions and returns results via the result queue.
  
  Parameters
  ----------
  task_queue : Queue
      Input queue providing tasks as tuples:
      (job_id, payload_ids, dxs, dys, sxs, sys)
  result_queue : Queue
      Output queue for results as tuples:
      (job_id, batch_results) where batch_results is
      List[(dx, dy, correlation)]
  shared_payloads : Dict[str, Dict[str, Any]] or None
      Shared dictionary mapping payload IDs to image data.
      If None, worker operates without shared memory.
  initial_payload_id : str or None
      Optional initial payload to preload before processing tasks
  
  Notes
  -----
  The worker maintains a cache of the current payload (ref_img, target_img,
  center) and only reloads when the payload_id changes. This avoids
  redundant data copying for batches using the same images.
  
  Tasks are tuples containing:
  - job_id : int - Identifier for this batch
  - payload_ids : List[str] - Payload ID for each correlation
  - dxs : List[float] - X-shifts
  - dys : List[float] - Y-shifts
  - sxs : List[float] - X-scales
  - sys : List[float] - Y-scales
  
  The loop terminates when task_queue provides None, allowing graceful
  shutdown of worker processes.
  
  See Also
  --------
  store_shared_payload : Function to populate shared_payloads dictionary
  correlation_for_params : Core correlation function called by workers
  """
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
