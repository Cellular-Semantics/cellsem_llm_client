# Dashboard Cost Tracking Validation Plan

This document outlines comprehensive testing strategies for validating the cost tracking implementation against OpenAI and Anthropic dashboards.

## Overview

The CellSem LLM Client provides two cost tracking methods:
1. **Per-request estimation** via `FallbackCostCalculator` - precise for individual queries
2. **Aggregate API tracking** via `ApiCostTracker` - useful for overall usage monitoring

## ⚠️ Key Limitation: Shared API Key Issue

The `ApiCostTracker` returns **aggregate usage** for the entire API key, including:
- Other applications using the same key
- Manual API calls via web interfaces
- Other team members' usage
- Background services or scripts

**Validation Strategy**: Focus on patterns and relative changes rather than absolute values.

## Testing Scenarios

### 1. Small Fixed Tests (Precise Validation)

**Purpose**: Validate cost calculation accuracy with predictable token counts

**Test Cases**:
```python
# Test 1: Minimal prompt
prompt = "Hello"
# Expected: ~5 input + ~3 output tokens
# Cost: ~$0.000015 (gpt-3.5-turbo)

# Test 2: Simple question
prompt = "What is 2+2?"
# Expected: ~15 input + ~5 output tokens
# Cost: ~$0.000035 (gpt-3.5-turbo)

# Test 3: Single word answer
prompt = "Is the sky blue? Answer yes or no."
# Expected: ~20 input + ~3 output tokens
# Cost: ~$0.000048 (gpt-3.5-turbo)
```

**Validation Steps**:
1. Run test with dedicated API key (no other usage)
2. Record exact timestamps
3. Wait 5-10 minutes for dashboard updates
4. Compare:
   - Token counts (±10% acceptable due to tokenization)
   - Costs (±2% acceptable due to rounding)
   - Request counts (should match exactly)

### 2. Batch Tests (Dashboard Aggregation)

**Purpose**: Validate aggregation accuracy and dashboard reporting

**Test Cases**:
```python
# Test 1: Identical queries
for i in range(10):
    response, usage = agent.query_with_tracking(
        "Count to 5",
        cost_calculator=calculator
    )
    total_cost += usage.estimated_cost_usd

# Expected dashboard: +10 requests, predictable token increase

# Test 2: Mixed model comparison
gpt35_agent = OpenAIAgent(model="gpt-3.5-turbo")
gpt4_agent = OpenAIAgent(model="gpt-4")

# Same prompt to both models
prompt = "Explain photosynthesis in 30 words"
response1, usage1 = gpt35_agent.query_with_tracking(prompt, cost_calculator=calc)
response2, usage2 = gpt4_agent.query_with_tracking(prompt, cost_calculator=calc)

# Expected: GPT-4 cost ~20x higher than GPT-3.5-turbo
```

### 3. Provider-Specific Tests

#### OpenAI Dashboard Validation

**Dashboard Location**: https://platform.openai.com/usage

**Key Metrics**:
- **Prompt tokens** → `input_tokens`
- **Completion tokens** → `output_tokens`
- **Total requests** → API call count
- **Cost breakdown** by model

**Test Script**:
```python
def test_openai_dashboard_comparison():
    """Test OpenAI cost tracking against dashboard."""
    from cellsem_llm_client.agents.agent_connection import OpenAIAgent
    from cellsem_llm_client.tracking.cost_calculator import FallbackCostCalculator
    from cellsem_llm_client.tracking.api_trackers import ApiCostTracker
    from datetime import datetime, timedelta

    # Setup
    calculator = FallbackCostCalculator()
    calculator.load_default_rates()
    agent = OpenAIAgent(model="gpt-3.5-turbo", api_key="your_test_key")
    tracker = ApiCostTracker(openai_api_key="your_test_key")

    # Record start time
    start_time = datetime.now()

    # Run controlled test
    test_queries = [
        "Hello world",
        "What is AI?",
        "Explain quantum computing in 20 words"
    ]

    total_estimated_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for query in test_queries:
        response, usage = agent.query_with_tracking(query, cost_calculator=calculator)
        total_estimated_cost += usage.estimated_cost_usd
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

    end_time = datetime.now()

    # Wait for dashboard update (5-10 minutes)
    print("Waiting 8 minutes for dashboard update...")
    time.sleep(480)

    # Fetch API usage
    api_usage = tracker.get_openai_usage(start_time, end_time + timedelta(minutes=10))

    # Compare results
    print(f"Estimated cost: ${total_estimated_cost:.6f}")
    print(f"Estimated input tokens: {total_input_tokens}")
    print(f"Estimated output tokens: {total_output_tokens}")
    print(f"Expected requests: {len(test_queries)}")

    print(f"API total tokens: {api_usage.total_tokens}")
    print(f"API input tokens: {api_usage.total_input_tokens}")
    print(f"API output tokens: {api_usage.total_output_tokens}")
    print(f"API total requests: {api_usage.total_requests}")

    # Validation checks
    token_diff = abs(api_usage.total_tokens - (total_input_tokens + total_output_tokens))
    token_accuracy = 1 - (token_diff / (total_input_tokens + total_output_tokens))

    print(f"Token accuracy: {token_accuracy:.2%}")
    assert token_accuracy > 0.9, f"Token accuracy too low: {token_accuracy:.2%}"
    assert api_usage.total_requests >= len(test_queries), "Request count mismatch"
```

#### Anthropic Console Validation

**Dashboard Location**: https://console.anthropic.com/

**Key Metrics**:
- **Input tokens** → `input_tokens`
- **Output tokens** → `output_tokens`
- **Thinking tokens** → `thinking_tokens` (Claude 3.7+ models)
- **Cost in USD** → Direct cost comparison

**Test Script**:
```python
def test_anthropic_dashboard_comparison():
    """Test Anthropic cost tracking against console."""
    # Similar structure to OpenAI test
    agent = AnthropicAgent(model="claude-3-haiku-20240307", api_key="your_test_key")
    # ... rest of implementation
```

## Dashboard Interpretation Guide

### Expected Accuracies

| Metric | Expected Accuracy | Notes |
|--------|------------------|-------|
| Request Count | 100% | Should match exactly |
| Token Counts | ±10% | Tokenization differences acceptable |
| Cost Estimates | ±5% | Rate database vs real-time pricing |

### Red Flags

**🚨 Investigate if**:
- Token count difference >15%
- Missing requests in dashboard
- Cost difference >10%
- Dashboard shows zero usage after waiting period

**Common Causes**:
- API key confusion (staging vs prod)
- Timing issues (dashboard delays)
- Other applications using same key
- Rate database outdated

### Validation Checklist

**Pre-Test Setup**:
- [ ] Use dedicated test API keys
- [ ] Verify no other scripts running
- [ ] Record exact start/end timestamps
- [ ] Use simple, predictable prompts

**During Test**:
- [ ] Log all requests with timestamps
- [ ] Save exact prompts and responses
- [ ] Record estimated costs immediately

**Post-Test Analysis**:
- [ ] Wait 5-10 minutes for dashboard updates
- [ ] Screenshot dashboard for documentation
- [ ] Compare token counts within ±10%
- [ ] Verify cost calculations within ±5%
- [ ] Document any discrepancies

## Implementation Scripts

### Quick Validation Test

```python
#!/usr/bin/env python3
"""Quick dashboard validation test for cost tracking."""

import time
from datetime import datetime, timedelta
from cellsem_llm_client.agents.agent_connection import OpenAIAgent
from cellsem_llm_client.tracking.cost_calculator import FallbackCostCalculator

def quick_validation_test():
    """Run a quick test for manual dashboard comparison."""
    # Setup
    calculator = FallbackCostCalculator()
    calculator.load_default_rates()
    agent = OpenAIAgent(model="gpt-3.5-turbo")

    # Single predictable test
    start = datetime.now()
    print(f"Test started at: {start}")

    response, usage = agent.query_with_tracking(
        "Say hello in exactly 2 words",
        cost_calculator=calculator
    )

    print(f"Response: {response}")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Estimated cost: ${usage.estimated_cost_usd:.6f}")
    print(f"Model: {usage.model}")

    end = datetime.now()
    print(f"Test completed at: {end}")
    print("\nCheck OpenAI dashboard in 5-10 minutes:")
    print("https://platform.openai.com/usage")
    print(f"Look for usage between {start.strftime('%H:%M')} - {end.strftime('%H:%M')}")

if __name__ == "__main__":
    quick_validation_test()
```

### Comprehensive Test Suite

```python
#!/usr/bin/env python3
"""Comprehensive dashboard comparison test suite."""

def run_comprehensive_tests():
    """Run all validation tests."""
    # Test 1: Single query precision
    test_single_query_precision()

    # Test 2: Batch aggregation
    test_batch_aggregation()

    # Test 3: Multi-model comparison
    test_multi_model_comparison()

    # Test 4: Provider comparison
    test_provider_comparison()

# Implementation details for each test...
```

## Best Practices for Shared Keys

### Recommendation: Use Dedicated Test Keys

**Ideal Setup**:
- **Production keys**: For actual applications
- **Test keys**: For validation and development
- **Personal keys**: For individual development

### Pattern-Based Validation

When using shared keys, focus on **relative changes**:

```python
# Before test
baseline_usage = tracker.get_recent_usage("openai", hours=1)

# Run controlled test (10 identical queries)
for i in range(10):
    agent.query("Hello")

# After test
post_test_usage = tracker.get_recent_usage("openai", hours=1)

# Validate pattern
request_increase = post_test_usage.total_requests - baseline_usage.total_requests
assert request_increase >= 10, f"Expected +10 requests, got +{request_increase}"
```

## Reporting Template

### Test Report Format

```markdown
## Cost Tracking Validation Report

**Date**: 2024-01-15
**Tester**: [Name]
**API Key Type**: Dedicated test key

### Test Results

| Metric | Estimated | Dashboard | Difference | Status |
|--------|-----------|-----------|------------|---------|
| Requests | 10 | 10 | 0% | ✅ |
| Input Tokens | 150 | 147 | -2% | ✅ |
| Output Tokens | 50 | 52 | +4% | ✅ |
| Total Cost | $0.000450 | $0.000461 | +2.4% | ✅ |

### Issues Found
- None

### Recommendations
- Token counting accurate within acceptable range
- Cost calculation validated against dashboard
```

This comprehensive testing plan provides structured validation of cost tracking against provider dashboards while accounting for the shared API key limitation.