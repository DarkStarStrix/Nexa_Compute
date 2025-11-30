"""
Rank data quality using LLM-as-a-judge.

This script evaluates a dataset (Parquet or JSONL) on quality metrics:
- Clarity
- Correctness
- Educational Value

Usage:
    python scripts/rank_data_quality.py --input data/raw/my_data.parquet --output data/processed/my_data_ranked.parquet
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from tqdm import tqdm

# Reuse existing client infrastructure
try:
    from nexa_eval.clients import OpenRouterClient, OpenRouterConfig, OpenRouterRequest
except ImportError:
    # Fallback or instruction if nexa_eval is not in pythonpath
    import sys
    sys.path.append(".")
    from nexa_eval.clients import OpenRouterClient, OpenRouterConfig, OpenRouterRequest

DEFAULT_MODEL = "openai/gpt-4o-mini"

QUALITY_SYSTEM_PROMPT = """You are an expert data quality auditor. You will be given a data sample (e.g., an instruction, a conversation, or a document).

Your task is to audit this data sample for quality.
Score the sample from 1 (poor) to 5 (excellent) on these dimensions:

1. **Clarity**: Is the text easy to understand, well-structured, and free of ambiguity?
2. **Correctness**: Does the information appear factually accurate and logical?
3. **Educational Value**: Is this data useful for training a model? Does it contain rich, non-trivial information?

Return a JSON object:
{
  "clarity": 1-5,
  "correctness": 1-5,
  "educational_value": 1-5,
  "reasoning": "concise explanation of the scores"
}

Do not include any additional text.
"""


@dataclass
class RankerSpec:
    """Configuration for the ranker model."""
    model_id: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 512


def parse_ranker_response(payload: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from ranker response."""
    try:
        # Clean up potential markdown code blocks
        cleaned = payload.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    required = ["clarity", "correctness", "educational_value", "reasoning"]
    if not all(key in parsed for key in required):
        return None
    
    try:
        return {
            "clarity": int(parsed["clarity"]),
            "correctness": int(parsed["correctness"]),
            "educational_value": int(parsed["educational_value"]),
            "reasoning": str(parsed["reasoning"]),
        }
    except (TypeError, ValueError):
        return None


def build_prompt(row: Dict[str, Any]) -> str:
    """Construct the prompt from a data row."""
    # Try to find common content fields
    content = ""
    if "text" in row:
        content = row["text"]
    elif "content" in row:
        content = row["content"]
    elif "prompt" in row and "response" in row:
        content = f"Prompt:\n{row['prompt']}\n\nResponse:\n{row['response']}"
    elif "instruction" in row and "output" in row:
        content = f"Instruction:\n{row['instruction']}\n\nOutput:\n{row['output']}"
    elif "messages" in row:
        # Handle chat format
        messages = row["messages"]
        if isinstance(messages, list):
            content = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
        else:
            content = str(messages)
    else:
        # Fallback: dump the whole row
        content = json.dumps(row, indent=2)

    return f"Data Sample:\n{content}\n"


def rank_data(
    df: pd.DataFrame,
    *,
    ranker: RankerSpec,
    client_config: Optional[OpenRouterConfig],
    dry_run: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Run the ranking process on the dataframe."""
    
    records = []
    rows = df.to_dict("records")
    
    progress = tqdm(total=len(rows), desc="Ranking Data Quality", unit="sample")

    if dry_run:
        for row in rows:
            records.append({
                **row,
                "clarity": 3,
                "correctness": 3,
                "educational_value": 3,
                "reasoning": "Dry run placeholder",
            })
            progress.update(1)
        progress.close()
        return pd.DataFrame(records)

    if client_config is None:
        raise ValueError("Client config required for non-dry run")

    def worker(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = build_prompt(row)
        request = OpenRouterRequest(
            prompt=prompt_text,
            system_prompt=QUALITY_SYSTEM_PROMPT,
            temperature=ranker.temperature,
            max_tokens=ranker.max_tokens,
        )
        
        attempts = 0
        backoff = client_config.retry_backoff
        
        while True:
            client = OpenRouterClient(client_config)
            try:
                response = client.generate([request], model=ranker.model_id, batch_size=1)[0]
                parsed = parse_ranker_response(response.output_text)
                
                if parsed is None:
                    # Failed to parse, return 0s
                    parsed = {
                        "clarity": 0,
                        "correctness": 0,
                        "educational_value": 0,
                        "reasoning": f"Failed to parse response: {response.output_text[:100]}...",
                    }
                
                return {
                    **row,
                    **parsed
                }
                
            except Exception as e:
                attempts += 1
                if attempts > client_config.max_retries:
                    # Return error row instead of crashing
                    return {
                        **row,
                        "clarity": -1,
                        "correctness": -1,
                        "educational_value": -1,
                        "reasoning": f"Error: {str(e)}",
                    }
                time.sleep(backoff)
                backoff *= 1.5
            finally:
                client.close()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, row) for row in rows]
        try:
            for future in as_completed(futures):
                records.append(future.result())
                progress.update(1)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise

    progress.close()
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Rank data quality using LLM.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet or jsonl file.")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without API calls.")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers.")
    
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    elif args.input.suffix == ".jsonl":
        df = pd.read_json(args.input, lines=True)
    elif args.input.suffix == ".json":
        df = pd.read_json(args.input)
    else:
        raise ValueError("Unsupported file format. Use .parquet, .json, or .jsonl")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to {args.limit} samples.")

    ranker = RankerSpec(model_id=args.model)
    client_config = None
    if not args.dry_run:
        client_config = OpenRouterConfig(model=ranker.model_id)

    print(f"Starting ranking with model {args.model}...")
    ranked_df = rank_data(
        df,
        ranker=ranker,
        client_config=client_config,
        dry_run=args.dry_run,
        max_workers=args.workers
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_parquet(args.output, index=False)
    print(f"Saved ranked data to {args.output}")
    
    # Show summary stats
    if not args.dry_run:
        print("\nSummary Statistics:")
        print(ranked_df[["clarity", "correctness", "educational_value"]].describe())


if __name__ == "__main__":
    main()
