# Cost Tracking and Estimation

This guide explains how to track costs using:

- **Estimated per-request costs** (default, no real API needed)
- **Actual key-level usage** (provider-reported, delayed, key-wide)

## Estimated Costs (Per Request)

- Enable tracking on calls: `query_unified(..., track_usage=True)`.
- If you do **not** pass a `cost_calculator`, a `FallbackCostCalculator` with bundled rates is created by default (`auto_cost=True`).
- Opt out by setting `auto_cost=False`.
- Provide your own calculator for custom rates: `query_unified(..., track_usage=True, cost_calculator=my_calc, auto_cost=False)`.
- Rate freshness is exposed as `usage.rate_last_updated` (from the rate source access date).

Example:

```python
from cellsem_llm_client.agents import LiteLLMAgent

agent = LiteLLMAgent(model="gpt-4o", api_key="key")
result = agent.query_unified(
    message="Summarize this.",
    track_usage=True,  # auto cost estimation by default
)
print(result.usage.estimated_cost_usd, result.usage.rate_last_updated)
```

## Actual Usage (Key-Level)

Use `ApiCostTracker` for provider-reported usage; this is **key-wide** and delayed by a few minutes.

```python
from datetime import date, timedelta
from cellsem_llm_client.tracking.api_trackers import ApiCostTracker

tracker = ApiCostTracker(openai_api_key="sk-...", anthropic_api_key="ak-...")
end = date.today()
start = end - timedelta(days=1)

openai_usage = tracker.get_openai_usage(start_date=start, end_date=end)
print(openai_usage.total_cost, openai_usage.total_requests)
```

Notes:
- Reports aggregate all usage for the API key (not per request).
- Expect a short provider-side delay before usage appears.

## Rates and Updates

- Bundled rates live in `tracking/rates.json`; `FallbackCostCalculator.load_default_rates()` reads this file.
- `usage.rate_last_updated` shows when the rate data was last refreshed.
- A weekly GitHub Action runs `scripts/update_rates.py` to refresh rates; it opens a PR if rates change.
