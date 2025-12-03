"""Update bundled rate data for cost estimation.

This script refreshes `src/cellsem_llm_client/tracking/rates.json` with
the latest known pricing (hard-coded here) and stamps the current UTC
access date. It is intended to be run by CI on a schedule.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

RATE_FILE = Path("src/cellsem_llm_client/tracking/rates.json")

# Maintain current pricing here; update values as provider pricing changes.
CURRENT_RATES = [
    {
        "provider": "openai",
        "model": "gpt-4",
        "input_cost_per_1k_tokens": 0.03,
        "output_cost_per_1k_tokens": 0.06,
        "cached_cost_per_1k_tokens": None,
        "thinking_cost_per_1k_tokens": None,
        "source": {
            "name": "Provider Documentation",
            "url": "https://openai.com/pricing",
        },
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "input_cost_per_1k_tokens": 0.005,
        "output_cost_per_1k_tokens": 0.015,
        "cached_cost_per_1k_tokens": None,
        "thinking_cost_per_1k_tokens": None,
        "source": {
            "name": "Provider Documentation",
            "url": "https://openai.com/pricing",
        },
    },
    {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_cost_per_1k_tokens": 0.00015,
        "output_cost_per_1k_tokens": 0.0006,
        "cached_cost_per_1k_tokens": 0.000075,
        "thinking_cost_per_1k_tokens": None,
        "source": {
            "name": "Provider Documentation",
            "url": "https://openai.com/pricing",
        },
    },
    {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "input_cost_per_1k_tokens": 0.0015,
        "output_cost_per_1k_tokens": 0.002,
        "cached_cost_per_1k_tokens": None,
        "thinking_cost_per_1k_tokens": None,
        "source": {
            "name": "Provider Documentation",
            "url": "https://openai.com/pricing",
        },
    },
    {
        "provider": "anthropic",
        "model": "claude-3-sonnet",
        "input_cost_per_1k_tokens": 0.003,
        "output_cost_per_1k_tokens": 0.015,
        "cached_cost_per_1k_tokens": None,
        "thinking_cost_per_1k_tokens": 0.006,
        "source": {
            "name": "Provider Documentation",
            "url": "https://www.anthropic.com/pricing",
        },
    },
    {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "input_cost_per_1k_tokens": 0.00025,
        "output_cost_per_1k_tokens": 0.00125,
        "cached_cost_per_1k_tokens": None,
        "thinking_cost_per_1k_tokens": 0.0005,
        "source": {
            "name": "Provider Documentation",
            "url": "https://www.anthropic.com/pricing",
        },
    },
]


def main() -> None:
    """Write updated rate data with current access_date."""
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    output = []
    for entry in CURRENT_RATES:
        entry_copy = dict(entry)
        source = dict(entry_copy["source"])
        source["access_date"] = now
        entry_copy["source"] = source
        output.append(entry_copy)

    RATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RATE_FILE.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
