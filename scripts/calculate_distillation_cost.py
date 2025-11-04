#!/usr/bin/env python3
"""Calculate cost for distillation job using GPT-mini as teacher model."""

import pandas as pd
from pathlib import Path

# Pricing (per 1M tokens)
INPUT_PRICE_PER_1M = 0.25  # $0.25 per 1M input tokens
OUTPUT_PRICE_PER_1M = 2.00  # $2.00 per 1M output tokens
CACHED_PRICE_PER_1M = 0.025  # $0.025 per 1M cached tokens

# Token estimation (rough: 1 token ≈ 4 characters for English text)
TOKENS_PER_CHAR = 0.25


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation)."""
    return int(len(str(text)) * TOKENS_PER_CHAR)


def calculate_cost(parquet_path: Path, system_prompt_path: Path, rows: int = None):
    """Calculate distillation cost."""
    
    # Load data
    df = pd.read_parquet(parquet_path)
    actual_rows = len(df)
    
    # Use provided row count or actual
    row_count = rows if rows else actual_rows
    print(f"Analyzing {row_count:,} rows (actual: {actual_rows:,})")
    
    # Load system prompt
    system_prompt = system_prompt_path.read_text() if system_prompt_path.exists() else ""
    system_prompt_tokens = estimate_tokens(system_prompt)
    
    print(f"\nSystem prompt tokens: {system_prompt_tokens:,}")
    
    # Analyze user prompts across all rows
    user_prompts = df['user_prompt'].astype(str)
    avg_user_prompt_chars = user_prompts.str.len().mean()
    avg_user_prompt_tokens = estimate_tokens(avg_user_prompt_chars)
    
    # System prompt with template name replaced (average)
    template_name = df['template_name'].mode().iloc[0] if len(df) > 0 else 'hypothesis'
    system_with_template = system_prompt.replace('{template_name}', str(template_name))
    system_prompt_tokens = estimate_tokens(system_with_template)
    
    print(f"Average user prompt tokens: {avg_user_prompt_tokens:,.0f}")
    print(f"System prompt tokens (with template): {system_prompt_tokens:,.0f}")
    
    # First request: full prompt (system + user) - uncached
    # Subsequent requests: only user prompt (system cached)
    first_user_prompt = str(df.iloc[0]['user_prompt']) if len(df) > 0 else ""
    first_request_tokens = system_prompt_tokens + estimate_tokens(first_user_prompt)
    
    # Subsequent requests use cached system prompt, only pay for user prompt
    user_only_tokens = avg_user_prompt_tokens
    
    # Estimate output tokens (reasoning + distilled response)
    # Typical response: 200-500 tokens, using 400 as conservative estimate
    avg_output_tokens = 400
    
    # Calculate totals
    # First request uses uncached system prompt
    # All subsequent requests use cached system prompt (only user prompt counted)
    uncached_input_tokens = first_request_tokens
    cached_input_tokens = user_only_tokens * (row_count - 1)  # System prompt cached, only user tokens
    total_output_tokens = avg_output_tokens * row_count
    
    # Total input tokens (for reporting)
    total_input_tokens = uncached_input_tokens + cached_input_tokens
    
    # Costs
    cost_uncached = (uncached_input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    cost_cached = (cached_input_tokens / 1_000_000) * CACHED_PRICE_PER_1M
    cost_output = (total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    
    total_cost = cost_uncached + cost_cached + cost_output
    
    print("\n" + "=" * 70)
    print("COST BREAKDOWN")
    print("=" * 70)
    print(f"\nInput Tokens:")
    print(f"  First request (uncached): {uncached_input_tokens:,.0f} tokens")
    print(f"    Cost @ ${INPUT_PRICE_PER_1M}/1M: ${cost_uncached:,.2f}")
    print(f"  Subsequent requests (cached, user prompt only): {cached_input_tokens:,.0f} tokens")
    print(f"    Cost @ ${CACHED_PRICE_PER_1M}/1M: ${cost_cached:,.2f}")
    print(f"  Total input tokens: {total_input_tokens:,.0f}")
    
    print(f"\nOutput Tokens:")
    print(f"  Total: {total_output_tokens:,.0f} tokens")
    print(f"  Cost @ ${OUTPUT_PRICE_PER_1M}/1M: ${cost_output:,.2f}")
    
    print("\n" + "-" * 70)
    print(f"TOTAL COST: ${total_cost:,.2f}")
    print("=" * 70)
    
    # Additional info
    print(f"\nPer-row cost: ${total_cost / row_count:.4f}")
    print(f"Average tokens per request:")
    print(f"  Input: {total_input_tokens / row_count:,.0f}")
    print(f"  Output: {avg_output_tokens:,.0f}")
    
    return {
        'rows': row_count,
        'input_tokens_uncached': uncached_input_tokens,
        'input_tokens_cached': cached_input_tokens,
        'output_tokens': total_output_tokens,
        'cost_uncached': cost_uncached,
        'cost_cached': cost_cached,
        'cost_output': cost_output,
        'total_cost': total_cost
    }


if __name__ == "__main__":
    import sys
    
    parquet_path = Path("data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet")
    system_prompt_path = Path("data/system_prompt_template.txt")
    
    # Allow override of row count
    rows = None
    if len(sys.argv) > 1:
        try:
            rows = int(sys.argv[1])
        except ValueError:
            pass
    
    print("=" * 70)
    print("NEXACOMPUTE DISTILLATION COST CALCULATOR")
    print("=" * 70)
    print(f"\nTeacher Model: GPT-mini")
    print(f"  Input: ${INPUT_PRICE_PER_1M}/1M tokens")
    print(f"  Output: ${OUTPUT_PRICE_PER_1M}/1M tokens")
    print(f"  Cached: ${CACHED_PRICE_PER_1M}/1M tokens")
    
    result = calculate_cost(parquet_path, system_prompt_path, rows)
    
    # Calculate for 120k-130k range if user specified
    if rows is None:
        print("\n" + "=" * 70)
        print("COST ESTIMATES FOR 120k-130k ROWS")
        print("=" * 70)
        for row_count in [120_000, 125_000, 130_000]:
            result_120k = calculate_cost(parquet_path, system_prompt_path, row_count)
            print(f"\n{row_count:,} rows: ${result_120k['total_cost']:,.2f}")

