---
title: Distillation Cost Analysis
slug: costs/distillation-cost-analysis
description: Cost breakdown for running NexaCompute distillation with GPT-mini.
---

# Distillation Cost Analysis

## Overview

This document calculates the cost for running the distillation pipeline using **GPT-mini** as the teacher model to generate QA pairs from the teacher inputs parquet.

## Setup

- **Teacher Model:** GPT-mini
- **Input Data:** `data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet`
- **System Prompt:** `data/system_prompt_template.txt`
- **Row Count:** 120k-130k (estimated final count)

## Pricing (GPT-mini)

- **Input tokens:** $0.25 per 1M tokens
- **Output tokens:** $2.00 per 1M tokens  
- **Cached tokens:** $0.025 per 1M tokens (90% discount on cached system prompt)

## Token Analysis

Based on actual data analysis:

### System Prompt
- **Length:** 323 characters
- **Tokens:** ~81 tokens (with template name replacement: ~79 tokens)
- **Status:** Can be cached after first request

### User Prompts
- **Average length:** 60 characters
- **Average tokens:** ~15 tokens per prompt
- **Range:** 37-99 characters (9-25 tokens)

### Full Request (First Request)
- **System prompt:** ~79 tokens
- **User prompt:** ~15 tokens
- **Total:** ~94 tokens (uncached)

### Subsequent Requests (Cached)
- **System prompt:** Cached (only $0.025/1M rate)
- **User prompt:** ~15 tokens (cached rate)
- **Total:** ~15 tokens per request (cached)

### Output Tokens
- **Estimated per response:** 400 tokens
  - Reasoning section: ~200 tokens
  - Distilled response: ~200 tokens
- **Conservative estimate** based on typical scientific distillation responses

## Cost Calculation

### For 125,000 Rows

#### Input Tokens
1. **First request (uncached):** ~94 tokens
   - Cost: (94 / 1,000,000) × $0.25 = **$0.00** (negligible)

2. **Subsequent requests (cached):** 124,999 × 15 tokens = 1,874,985 tokens
   - Cost: (1,874,985 / 1,000,000) × $0.025 = **$0.05**

3. **Total input cost:** ~**$0.05**

#### Output Tokens
- **Total output tokens:** 125,000 × 400 = 50,000,000 tokens
- **Cost:** (50,000,000 / 1,000,000) × $2.00 = **$100.00**

### Total Cost Summary

| Row Count | Input Cost | Output Cost | **Total Cost** |
|-----------|------------|-------------|----------------|
| 120,000   | $0.05      | $96.00      | **$96.05**     |
| 125,000   | $0.05      | $100.00     | **$100.05**    |
| 130,000   | $0.05      | $104.00     | **$104.05**    |

## Key Insights

1. **Output tokens dominate cost:** ~99.95% of total cost
2. **Caching is highly effective:** System prompt caching reduces input costs by 90%
3. **Per-row cost:** ~$0.0008 per row (essentially $0.0008 per QA pair generated)
4. **Cost efficiency:** Very reasonable for generating high-quality training data

## Cost Breakdown Percentage

- **Input tokens (uncached):** &lt;0.01%
- **Input tokens (cached):** &lt;0.01%  
- **Output tokens:** 99.98%

## Cost Optimization Strategies

1. **Use caching:** Always enable system prompt caching (already implemented)
2. **Batch processing:** Process in batches to optimize API usage
3. **Output length:** Consider if 400 tokens per response is necessary (could reduce if shorter responses acceptable)
4. **Retry logic:** Implement retry with exponential backoff to avoid wasted costs on failed requests

## Cost Comparison

At $100 for 125k QA pairs:
- **Cost per QA pair:** $0.0008
- **Cost per 1k QA pairs:** $0.80
- **Cost per 10k QA pairs:** $8.00

This is extremely cost-effective for generating high-quality training data via knowledge distillation.

## Running the Calculator

Use the cost calculator script:

```bash
# Calculate for actual row count (6,000)
python scripts/calculate_distillation_cost.py

# Calculate for specific row count (125,000)
python scripts/calculate_distillation_cost.py 125000
```

## Notes

- Token estimates use character-based approximation (1 token ≈ 4 characters)
- For production, consider using `tiktoken` for more accurate token counts
- Actual costs may vary slightly based on:
  - Actual response lengths (may be longer or shorter than 400 tokens)
  - API rate limits and retries
  - Caching effectiveness

