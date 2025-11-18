# Evaluation Issues Analysis: Gemini & GPT-5 Low Scores

## Root Causes Identified

### 1. **Gemini Responses Truncated** ✅ FIXED
**Problem**: All Gemini responses were being cut off at ~510 tokens (out of 512 max).

**Evidence**:
- Average output length: 89 characters (extremely short)
- Token usage shows most responses hit 510 tokens (75th percentile)
- Diagnostic test showed `finish_reason: "length"` indicating truncation

**Root Cause**: `max_tokens=512` was too low for scientific responses that need detailed explanations.

**Fix**: 
- Increased default `max_tokens` from 512 to 2048
- Updated `OpenRouterConfig.default_max_tokens` to 2048
- Added truncation warnings in the OpenRouter client

### 2. **GPT-5 Empty Responses** ⚠️ NEEDS REGENERATION
**Problem**: All GPT-5 outputs are empty strings, resulting in judge scores of 1.0 across all dimensions.

**Evidence**:
- 100% of GPT-5 outputs are empty (0 characters)
- Token usage shows exactly 512 tokens consumed for ALL responses
- Judge comments: "The model answer is missing"
- Diagnostic test shows GPT-5 API works correctly (returns 288 chars in test)

**Root Cause**: 
- API consumed tokens but content extraction failed or content was None
- Possible race condition or error handling issue during concurrent generation
- The `raw_response` field was being dropped, making debugging impossible

**Fixes Applied**:
- Added better error handling for None/empty content
- Preserved `raw_response` field in saved parquet files for debugging
- Added warnings when empty output but tokens consumed
- Improved content extraction with `.get("content") or ""` fallback

**Action Required**: Regenerate GPT-5 responses with the fixed code to capture actual content.

## Code Changes Made

### 1. `nexa_eval/clients/openrouter.py`
- Added `finish_reason` checking for truncation detection
- Improved content extraction: `message.get("content") or ""`
- Added warnings for empty responses and truncation
- Increased default `max_tokens` to 2048

### 2. `nexa_eval/generate_responses.py`
- Increased default `max_tokens` from 512 to 2048
- Preserved `raw_response` field in saved data
- Added warning when empty output but tokens consumed
- Better error handling in worker function

## Recommendations

1. **Regenerate Gemini responses** with `--max-tokens 2048` to get full responses
2. **Regenerate GPT-5 responses** with fixed code to capture actual content
3. **Re-run judge scoring** after regeneration
4. **Monitor warnings** during generation to catch truncation/empty response issues early

## Testing

Run diagnostic script to verify fixes:
```bash
python3 tmp_diagnose_api.py
```

Expected results:
- Gemini: Full responses without truncation warnings
- GPT-5: Actual content returned (not empty)

