"""Data lineage tracking and graph construction."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

LOGGER = logging.getLogger(__name__)


@dataclass
class LineageNode:
    id: str
    type: str # dataset, model, job
    name: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class LineageEdge:
    source: str
    target: str
    type: str # generated_by, trained_on, derived_from


class LineageGraph:
    """Tracks dependencies between data, models, and jobs."""

    def __init__(self) -> None:
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []

    def add_node(self, node_id: str, type: str, name: str, metadata: Optional[Dict] = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = LineageNode(node_id, type, name, metadata or {})

    def add_edge(self, source: str, target: str, type: str) -> None:
        self.edges.append(LineageEdge(source, target, type))

    def get_upstream(self, node_id: str) -> Set[str]:
        """Find all upstream dependencies (recursive)."""
        upstream = set()
        queue = [node_id]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self.edges:
                if edge.target == current:
                    upstream.add(edge.source)
                    queue.append(edge.source)
                    
        return upstream

    def get_downstream(self, node_id: str) -> Set[str]:
        """Find all downstream dependents (recursive)."""
        downstream = set()
        queue = [node_id]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self.edges:
                if edge.source == current:
                    downstream.add(edge.target)
                    queue.append(edge.target)
                    
        return downstream

    def to_json(self) -> str:
        return json.dumps({
            "nodes": [node.__dict__ for node in self.nodes.values()],
            "edges": [edge.__dict__ for edge in self.edges],
        }, indent=2)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "LineageGraph":
        graph = cls()
        with open(path, "r") as f:
            data = json.load(f)
            for node in data["nodes"]:
                graph.nodes[node["id"]] = LineageNode(**node)
            for edge in data["edges"]:
                graph.edges.append(LineageEdge(**edge))
        return graph

# Global instance (persisted to disk in real scenario)
_LINEAGE_GRAPH = LineageGraph()

def track_lineage(
    source_id: str,
    source_type: str,
    target_id: str,
    target_type: str,
    relation: str,
    source_name: str = "",
    target_name: str = "",
) -> None:
    """Record a lineage relationship."""
    _LINEAGE_GRAPH.add_node(source_id, source_type, source_name or source_id)
    _LINEAGE_GRAPH.add_node(target_id, target_type, target_name or target_id)
    _LINEAGE_GRAPH.add_edge(source_id, target_id, relation)
    LOGGER.info(f"lineage_tracked: {source_id} -> {target_id} ({relation})")

