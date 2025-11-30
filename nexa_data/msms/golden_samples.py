"""Golden sample regression suite for transform logic validation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .transforms import clean_and_canonicalize
from .config import PreprocessingConfig, PipelineConfig

logger = logging.getLogger(__name__)


class GoldenSampleTest:
    """Single golden sample test case."""

    def __init__(self, sample_id: str, input_npz_path: Path, expected_json_path: Path):
        """Initialize golden sample test.

        Args:
            sample_id: Unique identifier for this test
            input_npz_path: Path to input .npz file
            expected_json_path: Path to expected output JSON
        """
        self.sample_id = sample_id
        self.input_npz_path = input_npz_path
        self.expected_json_path = expected_json_path

    def load_input(self) -> Dict:
        """Load input data from .npz file."""
        data = np.load(self.input_npz_path, allow_pickle=True)
        return {key: data[key] for key in data.keys()}

    def load_expected(self) -> Dict:
        """Load expected output from JSON file."""
        with open(self.expected_json_path) as f:
            return json.load(f)

    def run(self, preprocessing: PreprocessingConfig) -> Dict:
        """Run transform pipeline and compare to expected output.

        Returns:
            Test result dictionary with pass/fail status and details
        """
        try:
            input_data = self.load_input()
            expected_output = self.load_expected()

            from .metrics import PipelineMetrics
            metrics = PipelineMetrics()

            actual_output = clean_and_canonicalize(
                self.sample_id,
                input_data,
                preprocessing,
                metrics,
            )

            if actual_output is None:
                return {
                    "sample_id": self.sample_id,
                    "passed": False,
                    "error": "Transform returned None (sample was filtered)",
                }

            result = self._compare_outputs(actual_output, expected_output)

            return {
                "sample_id": self.sample_id,
                "passed": result["passed"],
                "error": result.get("error"),
                "differences": result.get("differences", []),
            }
        except Exception as e:
            return {
                "sample_id": self.sample_id,
                "passed": False,
                "error": str(e),
            }

    def _compare_outputs(self, actual: Dict, expected: Dict) -> Dict:
        """Compare actual and expected outputs.

        Returns:
            Comparison result with pass/fail status
        """
        differences = []

        for key in set(actual.keys()) | set(expected.keys()):
            if key not in actual:
                differences.append(f"Missing key in actual: {key}")
                continue
            if key not in expected:
                differences.append(f"Extra key in actual: {key}")
                continue

            actual_val = actual[key]
            expected_val = expected[key]

            if isinstance(expected_val, (list, np.ndarray)):
                actual_arr = np.array(actual_val) if not isinstance(actual_val, np.ndarray) else actual_val
                expected_arr = np.array(expected_val) if not isinstance(expected_val, np.ndarray) else expected_val

                if actual_arr.shape != expected_arr.shape:
                    differences.append(f"{key}: shape mismatch {actual_arr.shape} vs {expected_arr.shape}")
                    continue

                if not np.allclose(actual_arr, expected_arr, rtol=1e-5, atol=1e-8):
                    max_diff = np.abs(actual_arr - expected_arr).max()
                    differences.append(f"{key}: values differ (max diff: {max_diff:.2e})")
            elif isinstance(expected_val, (int, float)):
                if not np.isclose(actual_val, expected_val, rtol=1e-5, atol=1e-8):
                    differences.append(f"{key}: value differs {actual_val} vs {expected_val}")
            else:
                if actual_val != expected_val:
                    differences.append(f"{key}: value differs {actual_val} vs {expected_val}")

        return {
            "passed": len(differences) == 0,
            "differences": differences,
        }


class GoldenSampleSuite:
    """Golden sample regression test suite."""

    def __init__(self, golden_samples_dir: Path, preprocessing: PreprocessingConfig):
        """Initialize golden sample suite.

        Args:
            golden_samples_dir: Directory containing golden sample test cases
            preprocessing: Preprocessing configuration to use
        """
        self.golden_samples_dir = Path(golden_samples_dir)
        self.preprocessing = preprocessing
        self.tests: List[GoldenSampleTest] = []

    def discover_tests(self) -> None:
        """Discover all golden sample tests in directory."""
        if not self.golden_samples_dir.exists():
            logger.warning(f"Golden samples directory does not exist: {self.golden_samples_dir}")
            return

        npz_files = sorted(self.golden_samples_dir.glob("*.npz"))
        for npz_path in npz_files:
            sample_id = npz_path.stem
            json_path = self.golden_samples_dir / f"{sample_id}.expected.json"

            if not json_path.exists():
                logger.warning(f"Missing expected output for {sample_id}: {json_path}")
                continue

            test = GoldenSampleTest(sample_id, npz_path, json_path)
            self.tests.append(test)

        logger.info(f"Discovered {len(self.tests)} golden sample tests")

    def run_all(self) -> Dict:
        """Run all golden sample tests.

        Returns:
            Test suite results dictionary
        """
        if not self.tests:
            self.discover_tests()

        if not self.tests:
            return {
                "passed": True,
                "total": 0,
                "passed_count": 0,
                "failed_count": 0,
                "message": "No golden samples found",
            }

        results = []
        passed_count = 0

        for test in self.tests:
            result = test.run(self.preprocessing)
            results.append(result)
            if result["passed"]:
                passed_count += 1

        failed_count = len(results) - passed_count
        all_passed = failed_count == 0

        return {
            "passed": all_passed,
            "total": len(results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "results": results,
        }

    def save_report(self, results: Dict, report_path: Path) -> None:
        """Save test results to report file.

        Args:
            results: Test suite results dictionary
            report_path: Path to save report
        """
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved golden sample report to {report_path}")


def run_golden_sample_suite(
    cfg: PipelineConfig,
    run_dir: Path,
    golden_samples_dir: Optional[Path] = None,
) -> bool:
    """Run golden sample regression suite.

    Args:
        cfg: Pipeline configuration
        run_dir: Run directory for reports
        golden_samples_dir: Optional override for golden samples directory

    Returns:
        True if all tests passed, False otherwise
    """
    if golden_samples_dir is None:
        golden_samples_dir = Path("data/golden_samples")

    suite = GoldenSampleSuite(golden_samples_dir, cfg.preprocessing)
    results = suite.run_all()

    report_dir = run_dir / "golden_samples"
    report_path = report_dir / f"report_{cfg.dataset_name}.json"
    suite.save_report(results, report_path)

    if not results["passed"]:
        logger.error(
            f"Golden sample suite failed: {results['failed_count']}/{results['total']} tests failed"
        )
        for result in results.get("results", []):
            if not result["passed"]:
                logger.error(f"  Failed: {result['sample_id']} - {result.get('error', 'See differences')}")

    return results["passed"]

