"""Base class for coalignment tree nodes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from multiprocessing import cpu_count

from sunpy.map import GenericMap

from .process_result import ProcessResult
from .worker_setup import setup_persistent_workers
from help_funcs import _vprint


class _WorkerPoolInfo:
    """Lightweight stand-in to report worker count without creating a Pool."""

    def __init__(self, n_workers: int) -> None:
        self._processes = n_workers

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


class CoalignmentNode(ABC):
    """
    Base class for coalignment process tree.
    
    Can be either:
    - Composite (orchestrator with children)
    - Leaf (process with execution logic)
    
    Attributes
    ----------
    node_id : str
        Unique identifier
    node_name : str
        Human-readable name
    parent : CoalignmentNode | None
        Parent node
    children : List[CoalignmentNode]
        Child nodes (empty for leaf processes)
    base_target_map : GenericMap | None
        Original uncorrected target map
    current_working_map : GenericMap | None
        Latest corrected map (maintained by root)
    result : ProcessResult | None
        Result from this node's execution
    worker_pool : Any | None
        Shared worker pool for parallel processing
    debug_writer : Any | None
        Shared debug PDF writer
    verbose : int | None
        Verbosity level
    output_directory : Path | None
        Output directory for artifacts
    is_executed : bool
        Whether this node has been executed
    """
    
    def __init__(self):
        self.node_id: str = ""
        self.node_name: str = ""
        self.parent: CoalignmentNode | None = None
        self.children: List[CoalignmentNode] = []
        
        # Data (optional)
        self.base_target_map: GenericMap | None = None
        self.current_working_map: GenericMap | None = None
        
        # Result
        self.result: ProcessResult | None = None
        
        # Resources (inherited if None)
        self.worker_pool = None
        self.debug_writer = None
        self.verbose: int | None = None
        self.output_directory: Path | None = None
        self.n_workers: int | None = None
        
        # Persistent worker infrastructure (managed at root)
        self.task_queue = None
        self.result_queue = None
        self.shared_payloads = None
        self.worker_processes = None
        
        # State
        self.is_executed: bool = False

    def _ensure_persistent_workers(self) -> None:
        """Create persistent workers at the root if missing."""
        if self.task_queue is not None and self.result_queue is not None and self.shared_payloads is not None:
            return
        _vprint(self.get_verbose_level(), 2, f"[{self.node_name}] Setting up persistent workers at {self.node_name}...")
        n_workers = self.n_workers if self.n_workers is not None else max(1, cpu_count() - 1)
        task_queue, result_queue, shared_payloads, worker_procs = setup_persistent_workers(n_workers)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.shared_payloads = shared_payloads
        self.worker_processes = worker_procs
        if self.worker_pool is None:
            self.worker_pool = _WorkerPoolInfo(n_workers)

    def _validate_persistent_worker_state(self) -> None:
        """Ensure persistent worker components are all present or all absent."""
        components = {
            "task_queue": self.task_queue,
            "result_queue": self.result_queue,
            "shared_payloads": self.shared_payloads,
            "worker_processes": self.worker_processes,
        }
        present = [name for name, value in components.items() if value is not None]
        if present and len(present) != len(components):
            missing = [name for name, value in components.items() if value is None]
            red = "\033[31m"
            reset = "\033[0m"
            raise RuntimeError(
                f"{red}Persistent worker state is inconsistent. Present: {present}, missing: {missing}.{reset}"
            )
    
    def execute(self) -> None:
        """
        Execute this node and all descendants.
        Recursively processes tree structure.
        """
        working_map = self.get_working_map()
        
        if self.children:
            # Composite: execute children
            self._execute_children()
        else:
            # Leaf: execute own process logic
            self._execute_own_process(working_map)
    
    def _execute_children(self) -> None:
        """Execute all children in sequence."""
        for child in self.children:
            verbose = self.get_verbose_level()
            _vprint(verbose, 1, f"\n[{self.node_name}] Executing child: {child.node_name}")
            
            child.execute()
            
            # Update working map after each child (at all composite levels)
            if child.result is not None:
                self.current_working_map = child.result.output_map
    
    @abstractmethod
    def _execute_own_process(self, working_map: GenericMap) -> None:
        """
        Execute process logic (override in leaf subclasses).
        Must store result in self.result, not return it.
        """
        raise NotImplementedError(
            f"Leaf node {self.node_id} must implement _execute_own_process()"
        )
    
    # ==================
    # TREE MANAGEMENT
    # ==================
    
    def add_child(self, child: 'CoalignmentNode') -> None:
        """Add a child process to this node."""
        if child.is_ancestor_of(self):
            raise ValueError(f"Cannot add {child.node_id}: would create cycle")
        
        self.children.append(child)
        child.parent = self
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    def is_ancestor_of(self, node: 'CoalignmentNode') -> bool:
        """Check if this node is an ancestor of another node."""
        current = node.parent
        while current is not None:
            if current == self:
                return True
            current = current.parent
        return False
    
    # ==================
    # DATA ACCESS (Memory-safe)
    # ==================
    
    def get_base_target_map(self) -> GenericMap:
        """Get original target map (searches up tree)."""
        if self.base_target_map is not None:
            return self.base_target_map
        elif self.parent is not None:
            return self.parent.get_base_target_map()
        else:
            raise ValueError(
                f"Process {self.node_id} requires base_target_map but none found in tree"
            )
    
    def get_working_map(self) -> GenericMap:
        """Get current working map (latest correction)."""
        if self.is_root():
            if self.current_working_map is not None:
                return self.current_working_map
            else:
                return self.get_base_target_map()
        else:
            return self.parent._get_latest_child_output()
    
    def _get_latest_child_output(self) -> GenericMap:
        """Get output from last executed child."""
        for child in reversed(self.children):
            if child.is_executed and child.result is not None:
                return child.result.output_map
        return self.get_working_map()
    
    def get_reference_map(self) -> GenericMap:
        """Get reference map (searches up tree, may not exist on this node)."""
        if hasattr(self, 'reference_map') and self.reference_map is not None:
            return self.reference_map
        elif self.parent is not None:
            return self.parent.get_reference_map()
        else:
            raise ValueError(
                f"Process {self.node_id} requires reference_map but none found in tree"
            )
    
    def get_reference_sequence(self) -> List[GenericMap]:
        """Get reference sequence (searches up tree, may not exist on this node)."""
        if hasattr(self, 'reference_sequence') and self.reference_sequence is not None:
            return self.reference_sequence
        elif self.parent is not None:
            return self.parent.get_reference_sequence()
        else:
            raise ValueError(
                f"Process {self.node_id} requires reference_sequence but none found in tree"
            )
    
    # ==================
    # RESOURCE ACCESS
    # ==================
    
    def get_worker_pool(self):
        """Get worker pool (searches up tree or creates at root)."""
        if self.worker_pool is not None:
            return self.worker_pool
        elif self.parent is not None:
            return self.parent.get_worker_pool()
        else:
            self._ensure_persistent_workers()
            self._validate_persistent_worker_state()
            return self.worker_pool
    
    def get_task_queue(self):
        """Get task queue for persistent workers (searches up tree)."""
        if self.task_queue is not None:
            self._validate_persistent_worker_state()
            return self.task_queue
        elif self.parent is not None:
            return self.parent.get_task_queue()
        else:
            self._ensure_persistent_workers()
            self._validate_persistent_worker_state()
            return self.task_queue
    
    def get_result_queue(self):
        """Get result queue for persistent workers (searches up tree)."""
        if self.result_queue is not None:
            self._validate_persistent_worker_state()
            return self.result_queue
        elif self.parent is not None:
            return self.parent.get_result_queue()
        else:
            self._ensure_persistent_workers()
            self._validate_persistent_worker_state()
            return self.result_queue
    
    def get_shared_payloads(self):
        """Get shared payloads dict for persistent workers (searches up tree)."""
        if self.shared_payloads is not None:
            self._validate_persistent_worker_state()
            return self.shared_payloads
        elif self.parent is not None:
            return self.parent.get_shared_payloads()
        else:
            self._ensure_persistent_workers()
            self._validate_persistent_worker_state()
            return self.shared_payloads
    
    def get_debug_writer(self):
        """Get debug PDF writer (searches up tree)."""
        if self.debug_writer is not None:
            return self.debug_writer
        elif self.parent is not None:
            return self.parent.get_debug_writer()
        else:
            return None
    
    def get_verbose_level(self) -> int:
        """Get verbosity level (searches up tree)."""
        if self.verbose is not None:
            return self.verbose
        elif self.parent is not None:
            return self.parent.get_verbose_level()
        else:
            return 0
    
    def get_output_directory(self) -> Path:
        """Get output directory (searches up tree)."""
        if self.output_directory is not None:
            return self.output_directory
        elif self.parent is not None:
            return self.parent.get_output_directory()
        else:
            return Path("./coalignment_output")
    
    # ==================
    # RESULTS COLLECTION
    # ==================
    
    def get_final_result(self) -> ProcessResult:
        """Get final result (last executed leaf process)."""
        if self.result is not None:
            return self.result
        elif self.children:
            for child in reversed(self.children):
                if child.is_executed:
                    return child.get_final_result()
        raise ValueError("No executed process found")
    
    def get_all_results(self) -> List[ProcessResult]:
        """Recursively collect all results from tree."""
        results = []
        if self.result is not None:
            results.append(self.result)
        for child in self.children:
            results.extend(child.get_all_results())
        return results

    def cleanup_workers(self) -> None:
        """Close and join worker pool and persistent workers."""
        verbose = self.get_verbose_level()
        
        # Shutdown persistent workers first
        if hasattr(self, 'worker_processes') and self.worker_processes:
            _vprint(verbose, 2, f"[{self.node_name}] Shutting down {len(self.worker_processes)} persistent workers...")
            if hasattr(self, 'task_queue') and self.task_queue:
                # Send shutdown signal to each worker
                for _ in self.worker_processes:
                    self.task_queue.put(None)
            else:
                _vprint(
                    verbose,
                    1,
                    f"[{self.node_name}] Warning: task_queue missing; terminating workers directly.",
                )
            
            # Wait for workers to finish, then terminate if needed
            for proc in self.worker_processes:
                # proc.join(timeout=5.0)
                if proc.is_alive():
                    _vprint(verbose, 1, f"[{self.node_name}] Warning: Worker process did not terminate gracefully")
                    proc.terminate()
            
            self.worker_processes.clear()
            self.task_queue = None
            self.result_queue = None
            self.shared_payloads = None
            self.worker_processes = None
            _vprint(verbose, 2, f"[{self.node_name}] Persistent workers shut down")
        
        # Close the worker pool
        if hasattr(self, 'worker_pool') and self.worker_pool is not None:
            _vprint(verbose, 2, f"[{self.node_name}] Closing worker pool...")
            self.worker_pool.close()
            self.worker_pool.join()
            self.worker_pool = None
            _vprint(verbose, 2, f"[{self.node_name}] Worker pool closed")
        elif verbose >= 3:
            _vprint(verbose, 3, f"[{self.node_name}] No worker pool to close")

    def cleanup_debug(self) -> None:
        """Close debug writer if active."""
        verbose = self.get_verbose_level()
        
        if hasattr(self, 'debug_writer') and self.debug_writer is not None:
            debug_pdf = None
            if hasattr(self, '_shared_debug_pdf_path'):
                debug_pdf = self._shared_debug_pdf_path
            
            _vprint(verbose, 2, f"[{self.node_name}] Closing debug writer...")
            self.debug_writer.close()
            self.debug_writer = None
            
            if debug_pdf:
                _vprint(verbose, 1, f"Saved PDF: {debug_pdf}")
            _vprint(verbose, 2, f"[{self.node_name}] Debug writer closed")
        elif verbose >= 3:
            _vprint(verbose, 3, f"[{self.node_name}] No debug writer to close")

    def cleanup(self) -> None:
        """Clean up all resources (workers and debug output)."""
        self.cleanup_workers()
        self.cleanup_debug()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup on deletion
