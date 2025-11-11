"""Mix tool-augmented episodes with a slice of the original SFT dataset."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import typer

LOGGER = logging.getLogger(__name__)

from nexa_compute.core.project_registry import DEFAULT_PROJECT_REGISTRY, ProjectRegistryError


@dataclass(slots=True)
class MixConfig:
    """Configuration describing the desired mix ratios."""

    tool_ratio: float = 0.85

    def __post_init__(self) -> None:
        if not 0.5 < self.tool_ratio < 0.99:
            raise ValueError("tool_ratio must lie between 0.5 and 0.99.")


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _deterministic_score(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _convert_sft_records(records: Iterable[dict], split: str, *, count: int) -> List[dict]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    frame = frame.copy()
    frame["score"] = frame["prompt"].astype(str).map(_deterministic_score)
    frame = frame.sort_values("score").head(count)
    converted: List[dict] = []
    for _, row in frame.iterrows():
        converted.append(
            {
                "id": f"sft-{row['id']}",
                "split": split,
                "category": "sft_base",
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                "source_prompt_id": row["id"],
                "metadata": {
                    "domain": row.get("domain"),
                    "template_name": row.get("template_name"),
                    "source": "original_sft",
                },
            }
        )
    return converted


def _mix_split(
    *,
    tool_records: List[dict],
    sft_records: List[dict],
    split: str,
    config: MixConfig,
) -> List[dict]:
    tool_records = [dict(record, metadata={**record.get("metadata", {}), "source": "tool_episode"}) for record in tool_records]
    tool_count = len(tool_records)
    if tool_count == 0:
        LOGGER.warning("No tool episodes provided for split '%s'.", split)
        return []

    desired_total = int(round(tool_count / config.tool_ratio))
    original_needed = max(1, desired_total - tool_count)
    original_available = len(sft_records)
    original_take = min(original_needed, original_available)
    LOGGER.info(
        "Split=%s tool=%s original_needed=%s original_available=%s taking=%s",
        split,
        tool_count,
        original_needed,
        original_available,
        original_take,
    )
    combined = tool_records + sft_records[:original_take]
    return combined


def _save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summarise(records: List[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    frame = pd.DataFrame(records)
    if frame.empty:
        return summary
    for split, group in frame.groupby("split"):
        categories = group["category"].value_counts(normalize=True).to_dict()
        summary[split] = {
            "count": int(len(group)),
            "category_distribution": categories,
        }
    return summary


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_slug: str = typer.Option(
        "scientific_assistant",
        "--project-slug",
        help="Project namespace to operate on.",
    ),
    tool_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing generated tool episodes.",
    ),
    sft_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing original SFT splits.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Destination directory for mixed datasets.",
    ),
    tool_ratio: float = typer.Option(
        0.85,
        min=0.6,
        max=0.95,
        help="Fraction of tool episodes in the final dataset.",
    ),
) -> None:
    """Mix tool episodes with a subset of the original SFT dataset."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = MixConfig(tool_ratio=tool_ratio)

    try:
        project_meta = DEFAULT_PROJECT_REGISTRY.get(project_slug)
    except ProjectRegistryError as exc:
        raise typer.BadParameter(str(exc))

    processed_root = project_meta.processed_data_dir
    tool_dir = tool_dir or processed_root / "tool_protocol"
    sft_dir = sft_dir or processed_root / "distillation/sft_datasets"
    output_dir = output_dir or processed_root / "tool_protocol"

    tool_train = _load_jsonl(tool_dir / "episodes_train.jsonl")
    tool_val = _load_jsonl(tool_dir / "episodes_validation.jsonl")
    tool_test = _load_jsonl(tool_dir / "episodes_test.jsonl")

    sft_train = _load_jsonl(sft_dir / "sft_scientific_v1_train.jsonl")
    sft_val = _load_jsonl(sft_dir / "sft_scientific_v1_validation.jsonl")
    sft_test = _load_jsonl(sft_dir / "sft_scientific_v1_test.jsonl")

    mixed_train = _mix_split(
        tool_records=tool_train,
        sft_records=_convert_sft_records(sft_train, "train", count=len(tool_train)),
        split="train",
        config=config,
    )
    mixed_val = _mix_split(
        tool_records=tool_val,
        sft_records=_convert_sft_records(sft_val, "validation", count=max(100, len(tool_val))),
        split="validation",
        config=config,
    )
    mixed_test = _mix_split(
        tool_records=tool_test,
        sft_records=_convert_sft_records(sft_test, "test", count=max(100, len(tool_test))),
        split="test",
        config=config,
    )

    _save_jsonl(mixed_train, output_dir / "sft_toolproto_v1_train.jsonl")
    _save_jsonl(mixed_val, output_dir / "sft_toolproto_v1_validation.jsonl")
    _save_jsonl(mixed_test, output_dir / "sft_toolproto_v1_test.jsonl")

    summary = {
        "tool_ratio": config.tool_ratio,
        "train": _summarise(mixed_train).get("train", {}),
        "validation": _summarise(mixed_val).get("validation", {}),
        "test": _summarise(mixed_test).get("test", {}),
    }
    (output_dir / "sft_toolproto_v1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "Mixed dataset written. train=%s validation=%s test=%s",
        len(mixed_train),
        len(mixed_val),
        len(mixed_test),
    )


if __name__ == "__main__":
    app()

