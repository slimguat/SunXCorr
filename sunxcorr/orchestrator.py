"""Orchestrator node for managing coalignment processes."""

from __future__ import annotations

from .base_node import CoalignmentNode


class Orchestrator(CoalignmentNode):
    """
    Orchestrator node that manages child processes.
    
    This is a concrete implementation of CoalignmentNode that only
    orchestrates children and does not perform its own processing.
    
    Automatically sets up persistent workers if not provided.
    """
    
    def __init__(self, n_workers: int = 48):
        super().__init__()
        self.node_id = "orchestrator"
        self.node_name = "Orchestrator"
        self._owns_workers = False
        
    
    def _execute_own_process(self, *args, **kwargs) -> None:
        """
        Orchestrators don't execute their own logic.
        This should never be called since orchestrators always have children.
        """
        raise RuntimeError(
            "Orchestrator nodes should only have children. "
            "This method should never be called."
        )
