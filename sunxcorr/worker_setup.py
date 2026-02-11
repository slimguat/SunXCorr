"""Persistent worker helpers used by SunXCorr.

This module creates manager-backed shared structures and spawns persistent
worker processes that execute correlation workloads. Workers are intended
to be started once (at the root orchestrator) and reused across many
optimization calls to avoid process spawn overhead.

Examples
--------
>>> from sunxcorr.worker_setup import setup_persistent_workers
>>> callable(setup_persistent_workers)
True
"""

from __future__ import annotations

import os
import time
from multiprocessing import Manager, Process, Queue
from typing import Any, List, Tuple

from .optimization import _corr_worker_loop


def _debug_log(msg: str) -> None:
    try:
        # Only write debug log to file when explicitly enabled to avoid
        # unbounded log growth. Enable via environment variable:
        # `export SUNXCORR_DEBUG_LOG=1`
        if os.environ.get("SUNXCORR_DEBUG_LOG"):
            with open("/tmp/sunxcorr_worker_debug.log", "a", encoding="utf-8") as fh:
                fh.write(f"{time.time():.3f} {msg}\n")
    except Exception:
        pass


def setup_persistent_workers(
    n_workers: int,
) -> Tuple[Queue[Any], Queue[Any], Any, List[Process]]:
    """
    Setup persistent worker infrastructure with Manager, queues, and worker processes.

    This should be called once at the root level before optimization begins.
    Workers remain alive until explicitly shutdown.

    Parameters
    ----------
    n_workers : int
        Number of worker processes to spawn

    Returns
    -------
    tuple
        (task_queue, result_queue, shared_payloads, worker_processes)
        - task_queue: Queue for submitting jobs to workers
        - result_queue: Queue for receiving results from workers
        - shared_payloads: Manager.dict() for storing ref/target images
        - worker_processes: List of Process objects

    Examples
    --------
    >>> from sunxcorr.worker_setup import setup_persistent_workers
    >>> callable(setup_persistent_workers)
    True
    """
    # Create Manager and shared data structures
    manager = Manager()
    shared_payloads = manager.dict()
    task_queue: Queue[Any] = Queue()
    result_queue: Queue[Any] = Queue()

    # Spawn persistent worker processes
    worker_processes: List[Process] = []
    for _ in range(n_workers):
        proc = Process(
            target=_corr_worker_loop, args=(task_queue, result_queue, shared_payloads)
        )
        proc.daemon = True
        proc.start()
        _debug_log(f"spawned worker pid={proc.pid}")
        worker_processes.append(proc)

    return task_queue, result_queue, shared_payloads, worker_processes


def shutdown_persistent_workers(
    task_queue: Queue[Any], worker_processes: List[Process], timeout: float = 2.0
) -> None:
    """
    Gracefully shutdown persistent worker processes.

    Parameters
    ----------
    task_queue : Queue
        Task queue to send shutdown sentinels
    worker_processes : List[Process]
        List of worker Process objects to shutdown
    timeout : float
        Timeout in seconds for graceful join before termination
    """
    # Send None sentinel to each worker
    for _ in worker_processes:
        task_queue.put(None)

    # Wait for workers to finish, then terminate if needed
    for proc in worker_processes:
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.terminate()
