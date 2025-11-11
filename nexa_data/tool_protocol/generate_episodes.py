"""Programmatically construct multi-turn tool episodes from selected prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import typer

from nexa_compute.core.project_registry import DEFAULT_PROJECT_REGISTRY, ProjectRegistryError
from nexa_tools.sandbox import SandboxRunner
from nexa_tools.units import UnitConverter

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeMessage:
    """Single conversational turn."""

    role: str
    content: str


@dataclass
class Episode:
    """Container for a multi-turn sample."""

    id: str
    split: str
    category: str
    messages: List[EpisodeMessage]
    source_prompt_id: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        """Convert to a serialisable dictionary."""

        return {
            "id": self.id,
            "split": self.split,
            "category": self.category,
            "messages": [message.__dict__ for message in self.messages],
            "source_prompt_id": self.source_prompt_id,
            "metadata": self.metadata,
        }


class EpisodeGenerator:
    """Generate tool-augmented episodes from base prompts."""

    def __init__(self, *, rng_seed: int = 20251111) -> None:
        self._sandbox = SandboxRunner()
        self._unit_converter = UnitConverter()
        self._rng = random.Random(rng_seed)

    def generate(self, dataframe: pd.DataFrame) -> List[Episode]:
        """Generate a collection of episodes from the selection dataframe."""

        episodes: List[Episode] = []
        for _, row in dataframe.iterrows():
            try:
                episode = self._build_episode(row)
                episodes.append(episode)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to build episode for prompt %s", row.get("id"))
        return episodes

    def generate_repairs(self, count: int = 50) -> List[Episode]:
        """Produce repair-focused episodes to teach error recovery."""

        repairs: List[Episode] = []
        for index in range(count):
            scenario = index % 3
            if scenario == 0:
                repairs.append(self._build_malformed_json_repair(index))
            elif scenario == 1:
                repairs.append(self._build_missing_fetch_repair(index))
            else:
                repairs.append(self._build_python_error_repair(index))
        return repairs

    def _build_episode(self, row: pd.Series) -> Episode:
        tool_category = row.get("tool_candidate", "hybrid")
        if tool_category == "units":
            return self._build_units_episode(row)
        if tool_category == "literature":
            return self._build_literature_episode(row)
        if tool_category == "simulation":
            return self._build_simulation_episode(row)
        return self._build_hybrid_episode(row)

    def _build_units_episode(self, row: pd.Series) -> Episode:
        prompt = str(row["prompt"])
        value, from_unit, to_unit = _parse_conversion_request(prompt)
        call_id = uuid4().hex
        args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}

        toolcall = _format_toolcall(
            call_id,
            tool="units.convert",
            args=args,
            return_format="json",
        )

        conversion = self._unit_converter.convert(value, from_unit=from_unit, to_unit=to_unit)
        toolresult_payload = {
            "id": call_id,
            "ok": True,
            "data": conversion,
            "error": None,
            "meta": {"ms": _pseudo_latency(call_id)},
        }

        tool_result = _format_toolresult(toolresult_payload)
        final_report = _format_final_report(
            task="unit_conversion",
            inputs={"prompt": prompt, "value": value, "from_unit": from_unit, "to_unit": to_unit},
            plan=[{"step": 1, "desc": "Use Pint to convert the requested units accurately."}],
            methods=[{"name": "pint.UnitRegistry", "code_ref": "tool://units.convert"}],
            metrics=[
                {
                    "name": "converted_value",
                    "value": conversion["value"],
                    "unit": conversion["unit"],
                }
            ],
            evidence=[],
            limitations=[],
            next_steps=[],
        )

        messages = [
            EpisodeMessage(role="user", content=prompt),
            EpisodeMessage(role="assistant", content=toolcall),
            EpisodeMessage(role="tool", content=tool_result),
            EpisodeMessage(role="assistant", content=final_report),
        ]
        return Episode(
            id=uuid4().hex,
            split=row["split"],
            category="units",
            messages=messages,
            source_prompt_id=row.get("id"),
        )

    def _build_literature_episode(self, row: pd.Series) -> Episode:
        prompt = str(row["prompt"])
        search_id = uuid4().hex
        search_results = _synth_search_results(prompt, top_k=3)

        search_call = _format_toolcall(
            search_id,
            tool="papers.search",
            args={"query": prompt, "top_k": 3, "corpus": "crossref"},
            return_format="json",
        )
        search_result = _format_toolresult(
            {
                "id": search_id,
                "ok": True,
                "data": {"results": search_results},
                "error": None,
                "meta": {"ms": _pseudo_latency(search_id)},
            }
        )

        chosen = search_results[0]
        fetch_id = uuid4().hex
        fetch_payload = _synth_fetch_result(chosen["doi"], chosen["title"], chosen["year"])
        fetch_call = _format_toolcall(
            fetch_id,
            tool="papers.fetch",
            args={"doi": chosen["doi"]},
            return_format="json",
        )
        fetch_result = _format_toolresult(
            {
                "id": fetch_id,
                "ok": True,
                "data": fetch_payload,
                "error": None,
                "meta": {"ms": _pseudo_latency(fetch_id)},
            }
        )

        final_report = _format_final_report(
            task="literature_grounding",
            inputs={"prompt": prompt},
            plan=[
                {"step": 1, "desc": "Search recent literature for relevant studies."},
                {"step": 2, "desc": "Review the highest-signal DOI and capture key claims."},
            ],
            methods=[
                {"name": "Crossref Search", "code_ref": "tool://papers.search"},
                {"name": "Crossref Fetch", "code_ref": "tool://papers.fetch"},
            ],
            metrics=[],
            evidence=[
                {
                    "doi": fetch_payload["doi"],
                    "title": fetch_payload["title"],
                    "where_used": "supports background section",
                }
            ],
            limitations=["Search limited to Crossref synthetic results."],
            next_steps=["Validate findings against domain-specific databases."],
        )

        messages = [
            EpisodeMessage(role="user", content=prompt),
            EpisodeMessage(role="assistant", content=search_call),
            EpisodeMessage(role="tool", content=search_result),
            EpisodeMessage(role="assistant", content=fetch_call),
            EpisodeMessage(role="tool", content=fetch_result),
            EpisodeMessage(role="assistant", content=final_report),
        ]

        return Episode(
            id=uuid4().hex,
            split=row["split"],
            category="literature",
            messages=messages,
            source_prompt_id=row.get("id"),
            metadata={
                "search_results": search_results,
                "fetch_payload": fetch_payload,
            },
        )

    def _build_simulation_episode(self, row: pd.Series) -> Episode:
        prompt = str(row["prompt"])
        params = _derive_simulation_params(prompt)
        call_id = uuid4().hex
        code = _generate_simulation_code(params)
        sandbox_result = self._sandbox.run(code, timeout_s=10)

        tool_call = _format_toolcall(
            call_id,
            tool="python.run",
            args={"code": code, "timeout_s": 10},
            return_format="json",
        )
        tool_result = _format_toolresult(
            {
                "id": call_id,
                "ok": sandbox_result.stderr == "",
                "data": {
                    "stdout": sandbox_result.stdout,
                    "stderr": sandbox_result.stderr,
                    "artifacts": sandbox_result.artifacts,
                },
                "error": None if sandbox_result.stderr == "" else {"message": sandbox_result.stderr},
                "meta": {"ms": _pseudo_latency(call_id)},
            }
        )

        capacity = _extract_capacity_from_stdout(sandbox_result.stdout)
        final_report = _format_final_report(
            task="python_simulation",
            inputs={"prompt": prompt, **params},
            plan=[
                {"step": 1, "desc": "Implement parameterised simulation in Python."},
                {"step": 2, "desc": "Execute script to compute relevant metrics."},
            ],
            methods=[{"name": "Python Sandbox", "code_ref": sandbox_result.artifacts[0] if sandbox_result.artifacts else "tool://python.run"}],
            metrics=[
                {"name": "simulated_capacity_mAh_g", "value": capacity or params["baseline_capacity"], "unit": "mAh/g"}
            ],
            evidence=[],
            limitations=["Model assumes linear scaling and ignores side reactions."],
            next_steps=["Validate simulation against experimental datasets."],
        )

        messages = [
            EpisodeMessage(role="user", content=prompt),
            EpisodeMessage(role="assistant", content=tool_call),
            EpisodeMessage(role="tool", content=tool_result),
            EpisodeMessage(role="assistant", content=final_report),
        ]
        return Episode(
            id=uuid4().hex,
            split=row["split"],
            category="simulation",
            messages=messages,
            source_prompt_id=row.get("id"),
        )

    def _build_hybrid_episode(self, row: pd.Series) -> Episode:
        prompt = str(row["prompt"])
        messages: List[EpisodeMessage] = [EpisodeMessage(role="user", content=prompt)]
        if self._should_use_think(prompt):
            think_call_id = uuid4().hex
            think_call = _format_toolcall(
                think_call_id,
                tool="think",
                args={"budget_tokens": 256, "goal": "Outline best tool sequence for the task."},
                return_format="json",
            )
            think_result = _format_toolresult(
                {
                    "id": think_call_id,
                    "ok": True,
                    "data": {
                        "notes": "Prioritise literature grounding before quantitative modelling.",
                        "budget_tokens": 256,
                    },
                    "error": None,
                    "meta": {"ms": _pseudo_latency(think_call_id)},
                }
            )
            messages.extend(
                [
                    EpisodeMessage(role="assistant", content=think_call),
                    EpisodeMessage(role="tool", content=think_result),
                ]
            )

        # Literature step (excluding its own user prompt and final report)
        literature_episode = self._build_literature_episode(row)
        messages.extend(literature_episode.messages[1:-1])

        # Python simulation step
        params = _derive_simulation_params(prompt)
        call_id = uuid4().hex
        code = _generate_simulation_code(params)
        sandbox_result = self._sandbox.run(code, timeout_s=12)

        py_call = _format_toolcall(
            call_id,
            tool="python.run",
            args={"code": code, "timeout_s": 12},
            return_format="json",
        )
        py_result = _format_toolresult(
            {
                "id": call_id,
                "ok": sandbox_result.stderr == "",
                "data": {
                    "stdout": sandbox_result.stdout,
                    "stderr": sandbox_result.stderr,
                    "artifacts": sandbox_result.artifacts,
                },
                "error": None if sandbox_result.stderr == "" else {"message": sandbox_result.stderr},
                "meta": {"ms": _pseudo_latency(call_id)},
            }
        )

        messages.extend(
            [
                EpisodeMessage(role="assistant", content=py_call),
                EpisodeMessage(role="tool", content=py_result),
            ]
        )

        capacity = _extract_capacity_from_stdout(sandbox_result.stdout)
        fetch_payload = literature_episode.metadata.get("fetch_payload", {})
        final_report = _format_final_report(
            task="hybrid_analysis",
            inputs={"prompt": prompt, **params},
            plan=[
                {"step": 1, "desc": "Survey recent literature for governing mechanisms."},
                {"step": 2, "desc": "Parameterise simulation using referenced data."},
                {"step": 3, "desc": "Execute numerical model and report metrics."},
            ],
            methods=[
                {"name": "Crossref Search", "code_ref": "tool://papers.search"},
                {"name": "Crossref Fetch", "code_ref": "tool://papers.fetch"},
                {"name": "Python Sandbox", "code_ref": sandbox_result.artifacts[0] if sandbox_result.artifacts else "tool://python.run"},
            ],
            metrics=[
                {
                    "name": "simulated_capacity_mAh_g",
                    "value": capacity or params["baseline_capacity"],
                    "unit": "mAh/g",
                }
            ],
            evidence=[
                {
                    "doi": fetch_payload.get("doi", "10.0000/placeholder"),
                    "title": fetch_payload.get("title", "Synthetic reference"),
                    "where_used": "Assumption for active material performance.",
                }
            ],
            limitations=["Simplified diffusion assumptions and no thermal coupling."],
            next_steps=["Run full DFN model with validated kinetics."],
        )
        messages.append(EpisodeMessage(role="assistant", content=final_report))

        return Episode(
            id=uuid4().hex,
            split=row["split"],
            category="hybrid",
            messages=messages,
            source_prompt_id=row.get("id"),
        )

    def _should_use_think(self, prompt: str) -> bool:
        """Decide whether to insert a think step based on prompt hash."""

        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        return int(digest[:2], 16) % 5 == 0

    def _build_malformed_json_repair(self, index: int) -> Episode:
        prompt = "Summarise recent advances in binder-free cathodes."
        bad_call_id = uuid4().hex
        bad_toolcall = "~~~toolcall\n{ 'id': '%s', 'tool': 'papers.search' }\n~~~" % bad_call_id

        error_result = _format_toolresult(
            {
                "id": bad_call_id,
                "ok": False,
                "data": None,
                "error": {"message": "Invalid JSON payload: Expecting property name enclosed in double quotes."},
                "meta": {"ms": 12},
            }
        )

        fixed_call = _format_toolcall(
            bad_call_id,
            tool="papers.search",
            args={"query": prompt, "top_k": 3, "corpus": "crossref"},
            return_format="json",
        )
        fixed_result = _format_toolresult(
            {
                "id": bad_call_id,
                "ok": True,
                "data": {"results": _synth_search_results(prompt, top_k=3)},
                "error": None,
                "meta": {"ms": 45},
            }
        )

        final_report = _format_final_report(
            task="repair_malformed_json",
            inputs={"prompt": prompt},
            plan=[{"step": 1, "desc": "Retry the tool call with valid JSON formatting."}],
            methods=[{"name": "Crossref Search", "code_ref": "tool://papers.search"}],
            metrics=[],
            evidence=[],
            limitations=["Initial request failed due to formatting error."],
            next_steps=["Continue literature grounded analysis."],
        )

        messages = [
            EpisodeMessage(role="user", content=prompt),
            EpisodeMessage(role="assistant", content=bad_toolcall),
            EpisodeMessage(role="tool", content=error_result),
            EpisodeMessage(role="assistant", content=fixed_call),
            EpisodeMessage(role="tool", content=fixed_result),
            EpisodeMessage(role="assistant", content=final_report),
        ]

        return Episode(
            id=f"repair-json-{index}",
            split="train",
            category="repair_malformed_json",
            messages=messages,
            metadata={"repair_type": "malformed_json"},
        )

    def _build_missing_fetch_repair(self, index: int) -> Episode:
        prompt = "Provide a literature-backed argument for higher porosity cathodes at 5C."
        search_id = uuid4().hex
        search_call = _format_toolcall(
            search_id,
            tool="papers.search",
            args={"query": prompt, "top_k": 2, "corpus": "crossref"},
            return_format="json",
        )
        results = _synth_search_results(prompt, top_k=2)
        search_result = _format_toolresult(
            {"id": search_id, "ok": True, "data": {"results": results}, "error": None, "meta": {"ms": 62}}
        )

        controller_warning = "~~~toolresult\n{\n  \"id\": \"%s\",\n  \"ok\": false,\n  \"data\": null,\n  \"error\": {\n    \"message\": \"DOI cited in final response without successful papers.fetch.\"\n  },\n  \"meta\": {\"ms\": 5}\n}\n~~~" % uuid4().hex

        fetch_id = uuid4().hex
        fetch_call = _format_toolcall(
            fetch_id,
            tool="papers.fetch",
            args={"doi": results[0]["doi"]},
            return_format="json",
        )
        fetch_result = _format_toolresult(
            {
                "id": fetch_id,
                "ok": True,
                "data": _synth_fetch_result(results[0]["doi"], results[0]["title"], results[0]["year"]),
                "error": None,
                "meta": {"ms": 31},
            }
        )

        final_report = _format_final_report(
            task="repair_missing_fetch",
            inputs={"prompt": prompt},
            plan=[
                {"step": 1, "desc": "Re-run citation fetch for DOI before final report."},
            ],
            methods=[{"name": "Crossref Fetch", "code_ref": "tool://papers.fetch"}],
            metrics=[],
            evidence=[
                {
                    "doi": results[0]["doi"],
                    "title": results[0]["title"],
                    "where_used": "Supports porosity-performance claim.",
                }
            ],
            limitations=["Initial attempt lacked DOI verification."],
            next_steps=["Proceed with final analysis using verified citation."],
        )

        messages = [
            EpisodeMessage(role="user", content=prompt),
            EpisodeMessage(role="assistant", content=search_call),
            EpisodeMessage(role="tool", content=search_result),
            EpisodeMessage(
                role="assistant",
                content="~~~final\n{\"task\": \"premature_final\", \"note\": \"(omitted)\"}\n~~~",
            ),
            EpisodeMessage(role="tool", content=controller_warning),
            EpisodeMessage(role="assistant", content=fetch_call),
            EpisodeMessage(role="tool", content=fetch_result),
            EpisodeMessage(role="assistant", content=final_report),
        ]

        return Episode(
            id=f"repair-fetch-{index}",
            split="train",
            category="repair_missing_fetch",
            messages=messages,
            metadata={"repair_type": "missing_fetch"},
        )

    def _build_python_error_repair(self, index: int) -> Episode:
        prompt = "Estimate diffusion-limited capacity of LFP at 5C and 298K."
        bad_call_id = uuid4().hex
        buggy_code = "import numpy as np\nprint(np.linspace(0, 10, num='fast'))"
        bad_call = _format_toolcall(
            bad_call_id,
            tool="python.run",
            args={"code": buggy_code, "timeout_s": 10},
            return_format="json",
        )
        bad_result = _format_toolresult(
            {
                "id": bad_call_id,
                "ok": False,
                "data": {
                    "stdout": "",
                    "stderr": "TypeError: 'num' must be an integer",
                    "artifacts": [],
                },
                "error": {"message": "TypeError during execution."},
                "meta": {"ms": 18},
            }
        )

        fixed_call_id = uuid4().hex
        fixed_code = (
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "time = np.linspace(0, 600, num=200)\n"
            "capacity = 165 * (1 - np.exp(-time / 120))\n"
            "print(f\"capacity={capacity[-1]:.2f}\")\n"
            "plt.plot(time, capacity)\n"
            "plt.xlabel('Time (s)')\n"
            "plt.ylabel('Capacity (mAh/g)')\n"
            "plt.savefig('lfp_diffusion.png')\n"
        )
        sandbox_result = self._sandbox.run(fixed_code, timeout_s=10)
        fixed_call = _format_toolcall(
            fixed_call_id,
            tool="python.run",
            args={"code": fixed_code, "timeout_s": 10},
            return_format="json",
        )
        fixed_result = _format_toolresult(
            {
                "id": fixed_call_id,
                "ok": sandbox_result.stderr == "",
                "data": {
                    "stdout": sandbox_result.stdout,
                    "stderr": sandbox_result.stderr,
                    "artifacts": sandbox_result.artifacts,
                },
                "error": None,
                "meta": {"ms": _pseudo_latency(fixed_call_id)},
            }
        )
        final_report = _format_final_report(
            task="repair_python_error",
            inputs={"prompt": prompt},
            plan=[
                {"step": 1, "desc": "Fix type error by providing an integer num parameter."},
                {"step": 2, "desc": "Re-run simulation and capture diffusion-limited capacity."},
            ],
            methods=[{"name": "Python Sandbox", "code_ref": sandbox_result.artifacts[0] if sandbox_result.artifacts else "tool://python.run"}],
            metrics=[
                {
                    "name": "diffusion_limited_capacity",
                    "value": _extract_capacity_from_stdout(sandbox_result.stdout) or 150.0,
                    "unit": "mAh/g",
                }
            ],
            evidence=[],
            limitations=["Simplified kinetic representation."],
            next_steps=["Cross-check results with particle-scale simulations."],
        )

        messages = [
            EpisodeMessage(role="user", content=prompt),
            EpisodeMessage(role="assistant", content=bad_call),
            EpisodeMessage(role="tool", content=bad_result),
            EpisodeMessage(role="assistant", content=fixed_call),
            EpisodeMessage(role="tool", content=fixed_result),
            EpisodeMessage(role="assistant", content=final_report),
        ]
        return Episode(
            id=f"repair-python-{index}",
            split="train",
            category="repair_python_error",
            messages=messages,
            metadata={"repair_type": "python_error"},
        )


def _format_toolcall(call_id: str, *, tool: str, args: Dict[str, Any], return_format: str) -> str:
    payload = {
        "id": call_id,
        "tool": tool,
        "args": args,
        "return_format": return_format,
    }
    return _wrap_block("toolcall", payload)


def _format_toolresult(payload: Dict[str, Any]) -> str:
    return _wrap_block("toolresult", payload)


def _format_final_report(
    *,
    task: str,
    inputs: Dict[str, Any],
    plan: List[Dict[str, Any]],
    methods: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    limitations: List[str],
    next_steps: List[str],
) -> str:
    payload = {
        "task": task,
        "inputs": inputs,
        "plan": plan,
        "methods": methods,
        "results": {
            "metrics": metrics,
            "figures": [],
            "tables": [],
        },
        "evidence": evidence,
        "limitations": limitations,
        "next_steps": next_steps,
    }
    return _wrap_block("final", payload)


def _wrap_block(tag: str, payload: Dict[str, Any]) -> str:
    return f"~~~{tag}\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n~~~"


def _parse_conversion_request(prompt: str) -> tuple[float, str, str]:
    pattern = re.compile(
        r"convert\s+(?P<value>[-+]?\d*\.?\d+)\s*(?P<from>[A-Za-z/%^0-9\-\s]+?)\s+to\s+(?P<to>[A-Za-z/%^0-9\-\s]+)",
        re.IGNORECASE,
    )
    match = pattern.search(prompt)
    if match:
        value = float(match.group("value"))
        from_unit = match.group("from").strip()
        to_unit = match.group("to").strip()
        return value, from_unit, to_unit
    return 1.0, "mol", "mmol"


def _synth_search_results(prompt: str, *, top_k: int) -> List[Dict[str, Any]]:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    results = []
    for index in range(top_k):
        suffix = digest[index * 8 : (index + 1) * 8]
        doi = f"10.1234/{suffix}"
        title = f"Study on {prompt[:40].title()} ({index + 1})"
        results.append(
            {
                "title": title,
                "authors": ["A. Researcher", "B. Scientist"],
                "year": 2015 + (index % 8),
                "doi": doi,
                "url": f"https://example.org/{suffix}",
                "abstract": f"Synthetic abstract fragment for {title}.",
            }
        )
    return results


def _synth_fetch_result(doi: str, title: str, year: int) -> Dict[str, Any]:
    return {
        "title": title,
        "abstract": f"Extended abstract for {title}.",
        "year": year,
        "bibtex": f"@article{{{doi.replace('/', '_')}},\n  title={{{title}}},\n  year={{{year}}}\n}}",
        "authors": ["A. Researcher", "B. Scientist"],
        "doi": doi,
        "url": f"https://example.org/{doi}",
        "source": "synthetic-crossref",
    }


def _pseudo_latency(seed: str) -> int:
    return 20 + int(seed[:4], 16) % 120


def _derive_simulation_params(prompt: str) -> Dict[str, Any]:
    numbers = _extract_numbers(prompt)
    c_rate = numbers[0] if numbers else 1.0
    temperature = numbers[1] if len(numbers) > 1 else 298.0
    porosity = 0.35 + 0.05 * math.sin(c_rate)
    baseline_capacity = 165.0
    return {
        "c_rate": c_rate,
        "temperature_K": temperature,
        "porosity": round(porosity, 3),
        "baseline_capacity": baseline_capacity,
    }


def _generate_simulation_code(params: Dict[str, Any]) -> str:
    gradient = 0.92 - 0.05 * params["porosity"]
    noise = 0.02 * params["c_rate"]
    return (
        "import json\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        f"c_rate = {params['c_rate']}\n"
        f"porosity = {params['porosity']}\n"
        f"temperature = {params['temperature_K']}\n"
        f"baseline_capacity = {params['baseline_capacity']}\n"
        f"gradient = {gradient:.5f}\n"
        f"noise = {noise:.5f}\n"
        "time = np.linspace(0, 600, num=200)\n"
        "capacity = baseline_capacity * (1 - gradient * np.log1p(c_rate) / 3)\n"
        "capacity = capacity - noise * np.sqrt(c_rate)\n"
        "capacity = max(capacity, 110)\n"
        "profile = capacity * (1 - np.exp(-time / 180))\n"
        "print(f\"capacity={profile[-1]:.2f}\")\n"
        "plt.plot(time, profile)\n"
        "plt.xlabel('Time (s)')\n"
        "plt.ylabel('Capacity (mAh/g)')\n"
        "plt.title('Simulated Capacity Curve')\n"
        "plt.tight_layout()\n"
        "plt.savefig('capacity_profile.png')\n"
    )


def _extract_capacity_from_stdout(stdout: str) -> float | None:
    match = re.search(r"capacity=([0-9]+(\.[0-9]+)?)", stdout)
    if match:
        return float(match.group(1))
    return None


def _extract_numbers(text: str) -> List[float]:
    return [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", text)]


def _save_episodes(episodes: Sequence[Episode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_record(), ensure_ascii=False) + "\n")


def _save_summary(episodes: Sequence[Episode], path: Path) -> None:
    summary: Dict[str, Any] = {}
    for split in ["train", "validation", "test"]:
        subset = [ep for ep in episodes if ep.split == split]
        summary[split] = {
            "count": len(subset),
            "category_distribution": pd.Series([ep.category for ep in subset]).value_counts(normalize=True).to_dict(),
        }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_slug: str = typer.Option(
        "scientific_assistant",
        "--project-slug",
        help="Project namespace to operate on.",
    ),
    selected_prompts: Optional[Path] = typer.Option(
        None,
        help="Path to selected prompts jsonl.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Directory to write generated episode files.",
    ),
    repair_count: int = typer.Option(50, help="Number of repair episodes to generate."),
) -> None:
    """CLI entry-point to generate tool episodes."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        project_meta = DEFAULT_PROJECT_REGISTRY.get(project_slug)
    except ProjectRegistryError as exc:
        raise typer.BadParameter(str(exc))

    project_tool_protocol = project_meta.processed_data_dir / "tool_protocol"
    selected_prompts = selected_prompts or project_tool_protocol / "selected_prompts.jsonl"
    output_dir = output_dir or project_tool_protocol

    dataframe = pd.read_json(selected_prompts, lines=True)
    generator = EpisodeGenerator()
    episodes = generator.generate(dataframe)
    repairs = generator.generate_repairs(repair_count)

    train_eps = [ep for ep in episodes if ep.split == "train"]
    val_eps = [ep for ep in episodes if ep.split == "validation"]
    test_eps = [ep for ep in episodes if ep.split == "test"]

    _save_episodes(train_eps, output_dir / "episodes_train.jsonl")
    _save_episodes(val_eps, output_dir / "episodes_validation.jsonl")
    _save_episodes(test_eps, output_dir / "episodes_test.jsonl")
    _save_episodes(repairs, output_dir / "episodes_repairs.jsonl")
    _save_summary(episodes, output_dir / "episodes_summary.json")

    LOGGER.info(
        "Generated episodes: train=%s validation=%s test=%s repairs=%s",
        len(train_eps),
        len(val_eps),
        len(test_eps),
        len(repairs),
    )


if __name__ == "__main__":
    app()

