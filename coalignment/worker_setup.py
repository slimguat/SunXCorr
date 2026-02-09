"""Utility functions for setting up persistent worker architecture."""

from multiprocessing import Manager, Process, Queue
from typing import  List#,Any, Dict,
from .optimization import _corr_worker_loop


def setup_persistent_workers(n_workers: int) -> tuple:
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
    """
    # Create Manager and shared data structures
    manager = Manager()
    shared_payloads = manager.dict()
    task_queue = Queue()
    result_queue = Queue()
    
    # Spawn persistent worker processes
    worker_processes: List[Process] = []
    for _ in range(n_workers):
        proc = Process(target=_corr_worker_loop, args=(task_queue, result_queue, shared_payloads))
        proc.daemon = True
        proc.start()
        worker_processes.append(proc)
    
    return task_queue, result_queue, shared_payloads, worker_processes


def shutdown_persistent_workers(
    task_queue: Queue,
    worker_processes: List[Process],
    timeout: float = 2.0
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
