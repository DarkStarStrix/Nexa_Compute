"""Test CLI for running module-specific test suites.

Per Scaling Policy Section 7: Every module must be represented in the test CLI.
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_TEST_MAP = {
    "core": [
        "tests/test_resilience.py",
        "tests/test_integration.py::test_storage_paths_integration",
    ],
    "data": [
        "tests/test_rust_modules.py::test_data_core",
        "tests/test_rust_modules.py::test_quality_core",
        "tests/test_rust_modules.py::test_pack_core",
        "tests/test_rust_modules.py::test_stats_core",
    ],
    "training": [
        "tests/test_integration.py",
    ],
    "evaluation": [
        "tests/test_integration.py",
    ],
    "models": [
        "tests/test_registry.py",
    ],
    "orchestration": [
        "tests/test_golden_pipeline.py",
    ],
    "monitoring": [
        "tests/monitoring/test_observability.py",
    ],
    "api": [
        "tests/test_api.py",
    ],
    "config": [
        "tests/test_config.py",
    ],
    "all": [],  # Will run all tests
}


def run_tests(module: str, verbose: bool = False) -> int:
    """Run tests for a specific module."""
    if module == "all":
        test_paths = ["tests/"]
    elif module in MODULE_TEST_MAP:
        test_paths = MODULE_TEST_MAP[module]
    else:
        print(f"Unknown module: {module}")
        print(f"Available modules: {', '.join(MODULE_TEST_MAP.keys())}")
        return 1
    
    if not test_paths:
        print(f"No tests defined for module: {module}")
        return 0
    
    cmd = ["pytest", "-v" if verbose else "-q"] + test_paths
    print(f"Running tests for module '{module}':")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


def list_modules() -> None:
    """List all available modules."""
    print("Available test modules:")
    for module in sorted(MODULE_TEST_MAP.keys()):
        test_count = len(MODULE_TEST_MAP[module])
        print(f"  {module:15} ({test_count} test file(s))")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run module-specific test suites per Scaling Policy"
    )
    parser.add_argument(
        "module",
        nargs="?",
        default="all",
        help="Module to test (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available modules",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_modules()
        return 0
    
    return run_tests(args.module, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
