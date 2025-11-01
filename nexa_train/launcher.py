"""Unified launcher for Nexa training backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from nexa_train.backends import get_backend, list_backends


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _coerce_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw


def _apply_override(params: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value, got: {override}")
    key_path, raw_value = override.split("=", 1)
    parts = key_path.split(".")
    cursor = params
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = _coerce_value(raw_value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Nexa training backends")
    parser.add_argument("--config", type=Path, required=False, help="Path to launcher YAML config")
    parser.add_argument("--backend", type=str, default=None, help="Override backend name")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override backend params using key=value (dot notation supported)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved backend params and exit")
    parser.add_argument("--list-backends", action="store_true", help="List available backends and exit")
    return parser


def _resolve_backend_payload(
    payload: Dict[str, Any],
    overrides: Sequence[str],
    backend_override: str | None,
) -> tuple[str, Dict[str, Any]]:
    backend_section = payload.get("backend") or {}
    backend_name = backend_override or backend_section.get("name") or "hf"
    backend_params = dict(backend_section.get("params") or {})
    for override in overrides:
        _apply_override(backend_params, override)
    return backend_name, backend_params


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_backends:
        print("Available backends:")
        for name in list_backends():
            print(f"  - {name}")
        return 0

    if args.config is None:
        parser.error("--config is required unless --list-backends is supplied")

    payload = _load_yaml(args.config)
    backend_name, backend_params = _resolve_backend_payload(payload, args.override, args.backend)

    if args.dry_run:
        rendered = {
            "backend": {
                "name": backend_name,
                "params": backend_params,
            }
        }
        print(yaml.safe_dump(rendered, sort_keys=False))
        return 0

    backend_fn = get_backend(backend_name)
    print(f"[nexa-train] Launching backend '{backend_name}' with config {args.config}")
    result = backend_fn(backend_params)
    manifest_path = result.get("manifest_path") if isinstance(result, dict) else None
    if manifest_path:
        print(f"[nexa-train] Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


