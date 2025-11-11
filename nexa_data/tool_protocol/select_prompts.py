"""Select prompts from the base SFT corpus that benefit from tool usage."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import typer

from nexa_compute.core.project_registry import DEFAULT_PROJECT_REGISTRY, ProjectRegistryError

LOGGER = logging.getLogger(__name__)


_TOOL_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "units": [
        re.compile(r"\bconvert\b"),
        re.compile(r"\bunit[s]?\b"),
        re.compile(r"\bkelvin\b"),
        re.compile(r"\bpascal\b"),
        re.compile(r"\bmol\b"),
        re.compile(r"\b%c\b"),
    ],
    "literature": [
        re.compile(r"\bdoi\b"),
        re.compile(r"\bcite\b"),
        re.compile(r"\breference\b"),
        re.compile(r"\bliterature\b"),
        re.compile(r"\bpaper\b"),
        re.compile(r"\bstudy\b"),
    ],
    "simulation": [
        re.compile(r"\bsimulat(e|ion)\b"),
        re.compile(r"\brun\b.*\bpython\b"),
        re.compile(r"\bmodel\b"),
        re.compile(r"\bcalculate\b"),
        re.compile(r"\bcompute\b"),
        re.compile(r"\bplot\b"),
    ],
    "hybrid": [
        re.compile(r"\bbattery\b"),
        re.compile(r"\bporosity\b"),
        re.compile(r"\bdiffusion\b"),
        re.compile(r"\bkinetic[s]?\b"),
        re.compile(r"\btemperature\b"),
        re.compile(r"\bc-rate\b"),
        re.compile(r"\bcurrent density\b"),
    ],
}

_TOOL_RATIOS = {
    "hybrid": 0.6,
    "literature": 0.2,
    "simulation": 0.15,
    "units": 0.05,
}

_SPLIT_WEIGHTS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}


@dataclass
class SelectionConfig:
    """Configuration describing how many prompts to sample."""

    target_total: int = 600
    tool_ratios: Dict[str, float] = None  # type: ignore[assignment]
    split_weights: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.tool_ratios = self.tool_ratios or _TOOL_RATIOS.copy()
        self.split_weights = self.split_weights or _SPLIT_WEIGHTS.copy()
        ratio_sum = sum(self.tool_ratios.values())
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Tool ratios must sum to 1.0, received {ratio_sum}")


class PromptSelector:
    """Select prompts that should include tool usage in synthetic episodes."""

    def __init__(self, dataframe: pd.DataFrame, *, config: SelectionConfig | None = None) -> None:
        self._df = dataframe.copy()
        self._config = config or SelectionConfig()
        if "prompt" not in self._df.columns:
            raise ValueError("Input dataframe must contain a 'prompt' column.")
        if "split" not in self._df.columns:
            raise ValueError("Input dataframe must contain a 'split' column.")
        self._df["prompt_norm"] = self._df["prompt"].astype(str).str.lower()
        self._df["tool_candidate"] = self._df["prompt_norm"].map(self._classify_prompt)
        self._df["det_score"] = self._df["prompt_norm"].map(_deterministic_score)

    def select(self) -> pd.DataFrame:
        """Return a dataframe containing the chosen prompts."""

        tool_frames = []
        for tool_name, ratio in self._config.tool_ratios.items():
            target = int(round(self._config.target_total * ratio))
            if target <= 0:
                continue
            tool_frame = self._df[self._df["tool_candidate"] == tool_name]
            if tool_frame.empty:
                LOGGER.warning("No prompts matched tool category '%s'.", tool_name)
                continue
            tool_frames.append(self._select_for_tool(tool_frame, tool_name, target))
        if not tool_frames:
            raise RuntimeError("No prompts selected; adjust configuration or patterns.")
        selection = pd.concat(tool_frames, ignore_index=True)
        selection = selection.sort_values("det_score").drop_duplicates(subset=["prompt"])
        return selection

    def _select_for_tool(self, frame: pd.DataFrame, tool_name: str, target: int) -> pd.DataFrame:
        """Select prompts for a single tool category respecting split weights."""

        selections = []
        for split, weight in self._config.split_weights.items():
            split_target = max(1, int(round(target * weight)))
            candidates = frame[frame["split"] == split].sort_values("det_score")
            if candidates.empty:
                LOGGER.warning("No candidates for tool '%s' in split '%s'.", tool_name, split)
                continue
            selections.append(candidates.head(split_target))
        if selections:
            return pd.concat(selections, ignore_index=True)
        return frame.sort_values("det_score").head(target)

    @staticmethod
    def _classify_prompt(prompt: str) -> str:
        """Classify a prompt into one of the target tool categories."""

        for tool_name, patterns in _TOOL_PATTERNS.items():
            if any(pattern.search(prompt) for pattern in patterns):
                return tool_name
        return "hybrid"


def _deterministic_score(text: str) -> int:
    """Generate a deterministic ordering score derived from text."""

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _load_dataset(path: Path) -> pd.DataFrame:
    """Load the JSONL dataset into a DataFrame."""

    LOGGER.info("Loading dataset from %s", path)
    return pd.read_json(path, lines=True)


def _save_records(records: Iterable[Dict[str, object]], destination: Path) -> None:
    """Persist selected prompts to JSONL."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_slug: str = typer.Option(
        "scientific_assistant",
        "--project-slug",
        help="Project namespace to operate on.",
    ),
    dataset_path: Optional[Path] = typer.Option(
        None,
        help="Source JSONL dataset containing prompt/response pairs.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        help="Destination path for the selected prompt metadata.",
    ),
    target_total: int = typer.Option(
        600,
        min=100,
        max=1000,
        help="Desired number of prompts to select for tool episodes.",
    ),
) -> None:
    """CLI entry-point for selecting tool-worthy prompts."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        project_meta = DEFAULT_PROJECT_REGISTRY.get(project_slug)
    except ProjectRegistryError as exc:
        raise typer.BadParameter(str(exc))

    processed_root = project_meta.processed_data_dir
    default_dataset = processed_root / "distillation/sft_datasets/sft_scientific_v1.jsonl"
    default_output = processed_root / "tool_protocol/selected_prompts.jsonl"

    dataset_path = dataset_path or default_dataset
    output_path = output_path or default_output

    dataframe = _load_dataset(dataset_path)
    selector = PromptSelector(dataframe, config=SelectionConfig(target_total=target_total))
    selection = selector.select()
    records = selection.drop(columns=["prompt_norm", "det_score"]).to_dict(orient="records")
    _save_records(records, output_path)
    LOGGER.info("Selected %s prompts written to %s", len(records), output_path)


if __name__ == "__main__":
    app()

