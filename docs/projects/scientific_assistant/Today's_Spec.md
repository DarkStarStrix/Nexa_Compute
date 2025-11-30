# Post-Training Spec: Turn Your 10B Into a Tool-Using Scientific Assistant

## 0) Objective (what changes after post-train)

Make the model **act**, not just answer:

* Detect when to: search literature, run Python/sims, convert units, or think longer.
* Emit **valid tool calls**, integrate **real results**, and output a **structured, grounded FinalReport**.
* Stay concise, professional, reproducible.

---

## 1) Data You’ll Build (from your existing 100k Q&A)

You won’t redo the 100k. You’ll add a **tool-protocol dataset** (300–800 episodes) that teaches:

* **When** to use tools
* **How** to call them
* **How** to integrate results
* **How** to finalize a structured scientific report

### 1.1 Episode Types (balanced mix)

1. **Literature grounding**: `papers.search → papers.fetch → FinalReport`
2. **Computation/sim**: `python.run (plot/number) → FinalReport`
3. **Units/sanity**: `units.convert → FinalReport`
4. **Hybrid**: `search→fetch→python.run→FinalReport`
5. **Long-thought**: model uses a “think” step, then chooses a tool (see §2.3).

Target: 60% hybrid, 20% literature, 15% sim, 5% units.

### 1.2 Selecting prompts from the 100k

Pick ~300–800 Qs where tooling adds value:

* lithium-ion batteries (C-rate, porosity, temp), materials trends, simple kinetics/diffusion, unit conversions, citation-needed questions.
* Keep task diversity but map to a small tool set.

---

## 2) Protocol & Schemas (what the model must emit)

### 2.1 ToolCall (assistant → controller)

Fenced JSON block:

```
~~~toolcall
{
  "id": "uuid",
  "tool": "python.run | papers.search | papers.fetch | units.convert | think",
  "args": {},
  "return_format": "json | table | text"
}
~~~
```

### 2.2 ToolResult (controller → assistant)

```
~~~toolresult
{
  "id": "uuid",
  "ok": true,
  "data": {},
  "error": null,
  "meta": { "ms": 137 }
}
~~~
```

### 2.3 “Think” tool (lightweight deliberation)

Let the model ask for scratch-space when unsure:

```
tool = "think", args = {"budget_tokens": 256, "goal": "decide best next tool"}
```

Controller simply echoes the scratch back as a toolresult; model then commits to a real tool.

### 2.4 FinalReport (assistant → user)

```
~~~final
{
  "task": "battery_hypothesis_test",
  "inputs": {...},
  "plan": [{"step":1,"desc":"..."}, ...],
  "methods": [{"name":"...","code_ref":"artifact://sim_01.py"}],
  "results": {
    "metrics": [{"name":"capacity_at_5C","value":146.2,"unit":"mAh/g"}],
    "figures": [{"path":"artifact://plot.png","caption":"..."}],
    "tables": [{"name":"assumptions","rows":[["porosity",0.32],["r_um",0.25]]}]
  },
  "evidence": [{"doi":"10.XXXX/...","title":"...", "where_used":"assumption"}],
  "limitations": ["..."],
  "next_steps": ["..."]
}
~~~
```

---

## 3) Tooling Surface (v0 to train against)

Implement minimal endpoints (FastAPI):

* `python.run(code:str, timeout_s:int=10)` → `{stdout, stderr, artifacts[]}` (sandboxed; numpy/scipy/pandas/matplotlib preinstalled; no network).
* `papers.search(query:str, top_k:int, corpus:"arxiv|crossref|pubmed")` → `[{title, authors[], year, doi?, url, abstract}]`
* `papers.fetch(doi:str)` → `{title, abstract, year, bibtex, url}`
* `units.convert(value, from_unit, to_unit)` → `{value}` (backed by `pint`)
* `think(goal, budget_tokens)` → `{notes}` (controller returns the model’s own scratch to itself)

Guardrails: 10s timeout, memory cap, one tool per call, require `papers.fetch` before citing a DOI.

---

## 4) Episode Format (multi-turn SFT sample)

Store each as a list of messages:

```json
[
  {"role":"user","content":"Test if higher LFP porosity improves capacity at 5C, 298K."},
  {"role":"assistant","content":"~~~toolcall {...papers.search...}~~~"},
  {"role":"tool","content":"~~~toolresult {...results...}~~~"},
  {"role":"assistant","content":"~~~toolcall {...papers.fetch...}~~~"},
  {"role":"tool","content":"~~~toolresult {...verified doi...}~~~"},
  {"role":"assistant","content":"~~~toolcall {...python.run code that outputs a number+plot...}~~~"},
  {"role":"tool","content":"~~~toolresult {...stdout:'146.2', artifacts:['artifact://plot.png']}~~~"},
  {"role":"assistant","content":"~~~final {...structured report citing the verified DOI...}~~~"}
]
```

Also include ~50 **repair** episodes:

* malformed JSON → corrected
* cited DOI without fetch → forced fetch then final
* tool error → revised code then final

---

## 5) Dataset Construction Pipeline (fast path)

1. **Seed authoring**: Hand-write 20–40 gold traces covering each tool pattern; run the python code to produce real artifacts.
2. **Programmatic expansion** (×5–×10 each seed):

   * Parameter sweep (porosity, temp, C-rate, radii).
   * Swap corpora (arxiv/crossref).
   * Auto-generate simple python snippets that echo numeric outputs + save a plot.
   * Execute to capture real `toolresult` payloads.
3. **Pack to JSONL**: one episode per line with fields: `{"id", "domain", "messages":[...]}`
4. **Mix**: 80–90% tool episodes + 10–20% from the original 100k (unmodified) to retain voice.

---

## 6) Training (QLoRA SFT; no RLHF)

* Base: your merged 10B.
* Data: 300–800 episodes (multi-turn) + 10–20% original SFT.
* Loss weighting: +0.2–0.3 on tokens inside `~~~toolcall` and `~~~final`.
* Params (safe): r=16, α=16, target `{q,k,v,o,gate,up,down}`, lr=2e-4, cosine, warmup 3%, seq_len 4k–8k, bs=2, grad_acc 8–16, 1.5–2 epochs.

---

## 7) Controller & Coherence (runtime)

* Loop max 5 tool rounds; one JSON-repair attempt if invalid.
* Enforce **“no DOI in FinalReport unless papers.fetch succeeded”**.
* If a tool fails or budget exceeded, require a `limitations` entry.
* Save all artifacts (code, plots, csv, FinalReport.json) with COMPLETE markers.

---

## 8) Evaluation (gate what matters)

Per test prompt:

* JSON validity (ToolCall + FinalReport parse rate)
* Tool success rate (no exceptions)
* Citation validity (DOI resolves; title fuzzy-match)
* Numeric sanity (unit conversions; simple physics constraints)
* Steps used (should the model have used a tool? penalize spurious calls)
* Latency (≤ N tool rounds; time budget)

Score: weighted composite; set a pass threshold before promotion.

---

## 9) Deliverables to add to your repo

* `schemas/` → ToolCall, ToolResult, FinalReport JSON-Schema
* `tools/` → FastAPI server (`python.run`, `papers.*`, `units.convert`, `think`)
* `data/episodes_train.jsonl` → multi-turn SFT set
* `prompts/system.txt` + 3–5 few-shots
* `train/qlora_toolproto.yaml` → Axolotl/HF config
* `serve/controller.py` → parsing + tool loop
* `eval/toolproto.py` → the gates above

