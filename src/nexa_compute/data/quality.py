"""Data quality validation and monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from nexa_compute.monitoring.alerts import AlertSeverity, send_alert

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    checks: Dict[str, bool]
    errors: List[str]


class DataValidator:
    """Validates data against schema and quality rules."""

    def validate_schema(
        self,
        data: pd.DataFrame,
        schema: Dict[str, str],
    ) -> ValidationResult:
        """Check if dataframe matches expected schema."""
        checks = {}
        errors = []
        
        # Check columns exist
        missing_cols = set(schema.keys()) - set(data.columns)
        if missing_cols:
            checks["columns_exist"] = False
            errors.append(f"Missing columns: {missing_cols}")
        else:
            checks["columns_exist"] = True
            
        # Check types (simplified)
        type_mismatches = []
        for col, expected_type in schema.items():
            if col not in data.columns:
                continue
                
            # Very basic type check mapping
            actual = str(data[col].dtype)
            if expected_type == "int" and "int" not in actual:
                type_mismatches.append(f"{col}: expected int, got {actual}")
            elif expected_type == "float" and "float" not in actual and "int" not in actual:
                type_mismatches.append(f"{col}: expected float, got {actual}")
                
        if type_mismatches:
            checks["types_match"] = False
            errors.extend(type_mismatches)
        else:
            checks["types_match"] = True
            
        return ValidationResult(
            passed=all(checks.values()),
            checks=checks,
            errors=errors,
        )

    def validate_quality(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any],
        dataset_name: str = "unknown",
    ) -> ValidationResult:
        """Run quality checks (nulls, ranges, uniqueness)."""
        checks = {}
        errors = []
        
        # Check nulls
        if "max_null_fraction" in rules:
            threshold = rules["max_null_fraction"]
            null_frac = data.isnull().mean().max()
            if null_frac > threshold:
                checks["nulls"] = False
                errors.append(f"Null fraction {null_frac:.2f} exceeds {threshold}")
            else:
                checks["nulls"] = True
                
        # Check uniqueness
        if "unique_cols" in rules:
            for col in rules["unique_cols"]:
                if col in data.columns:
                    if not data[col].is_unique:
                        checks[f"unique_{col}"] = False
                        errors.append(f"Column {col} contains duplicates")
                    else:
                        checks[f"unique_{col}"] = True
                        
        passed = all(checks.values())
        
        if not passed:
            send_alert(
                title=f"Data Quality Failure: {dataset_name}",
                message=f"Validation failed with {len(errors)} errors",
                severity=AlertSeverity.WARNING,
                source="data_validator",
                metadata={"errors": errors[:5]}, # Truncate errors
            )
            
        return ValidationResult(
            passed=passed,
            checks=checks,
            errors=errors,
        )

