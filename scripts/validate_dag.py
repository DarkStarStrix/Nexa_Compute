#!/usr/bin/env python3
"""Validate that NexaCompute module dependencies form a DAG (no cycles).

Per Scaling Policy Section 11: Module dependencies must form a DAG.
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Module dependency rules (per architecture.md)
CORE_MODULES = {"core", "config"}
DOMAIN_MODULES = {"data", "models", "training", "evaluation"}
ORCHESTRATION_MODULES = {"orchestration"}
CROSS_CUTTING_MODULES = {"monitoring", "api"}

# Expected dependency rules
ALLOWED_DEPENDENCIES = {
    "core": set(),  # No dependencies
    "config": set(),  # No dependencies
    "data": {"core"},
    "models": {"core"},
    "training": {"core", "models", "data"},
    "evaluation": {"core", "models", "data"},
    "orchestration": {"core", "data", "models", "training", "evaluation"},
    "monitoring": {"core"},
    "api": {"core", "monitoring", "orchestration"},
}


def extract_imports(file_path: Path) -> Set[str]:
    """Extract module imports from a Python file."""
    imports = set()
    try:
        with file_path.open("r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    # Extract top-level module name
                    module_parts = node.module.split(".")
                    if module_parts[0] == "nexa_compute" and len(module_parts) > 1:
                        imports.add(module_parts[1])
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
    
    return imports


def find_module_dependencies(module_name: str, src_root: Path) -> Set[str]:
    """Find all dependencies for a module."""
    module_dir = src_root / "nexa_compute" / module_name
    if not module_dir.exists():
        return set()
    
    dependencies = set()
    for py_file in module_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        imports = extract_imports(py_file)
        dependencies.update(imports)
    
    # Filter to only nexa_compute modules
    dependencies = {d for d in dependencies if d in ALLOWED_DEPENDENCIES}
    return dependencies


def detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Detect cycles in dependency graph using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()
    path = []
    
    def dfs(node: str) -> None:
        if node in rec_stack:
            # Found cycle
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, set()):
            dfs(neighbor)
        
        rec_stack.remove(node)
        path.pop()
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return cycles


def main() -> int:
    """Validate DAG structure."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "nexa_compute"
    
    if not src_root.exists():
        print(f"Error: Source directory not found: {src_root}")
        return 1
    
    # Build dependency graph
    graph: Dict[str, Set[str]] = defaultdict(set)
    
    for module in ALLOWED_DEPENDENCIES.keys():
        deps = find_module_dependencies(module, src_root)
        graph[module] = deps
        
        # Check against allowed dependencies
        allowed = ALLOWED_DEPENDENCIES[module]
        violations = deps - allowed
        if violations:
            print(f"⚠️  Module '{module}' has disallowed dependencies: {violations}")
            print(f"   Allowed: {allowed}")
            print(f"   Found: {deps}")
    
    # Detect cycles
    cycles = detect_cycles(graph)
    
    if cycles:
        print("❌ Cycle detected in dependency graph!")
        for cycle in cycles:
            print(f"   Cycle: {' → '.join(cycle)}")
        return 1
    
    print("✅ Dependency graph is acyclic (DAG)")
    print("\nDependency graph:")
    for module in sorted(graph.keys()):
        deps = graph[module]
        if deps:
            print(f"  {module:15} → {', '.join(sorted(deps))}")
        else:
            print(f"  {module:15} → (no dependencies)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

