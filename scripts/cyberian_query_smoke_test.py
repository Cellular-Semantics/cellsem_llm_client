#!/usr/bin/env python3
"""Smoke test for CyberianAgent.query() in managed or external server mode.

Example:
  uv run python scripts/cyberian_query_smoke_test.py \
    --query "Reply with exactly: CYBERIAN_OK" \
    --port 3284
"""

from __future__ import annotations

import argparse
import sys
import time

from cellsem_llm_client.utils.config import create_cyberian_agent


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Smoke test standard query() via Cyberian/Codex in managed mode."
    )
    parser.add_argument(
        "--model",
        default="cyberian/codex",
        help="Model string used to route to Cyberian path (default: cyberian/codex).",
    )
    parser.add_argument(
        "--query",
        default="Reply with exactly: CYBERIAN_OK",
        help="Prompt sent to query().",
    )
    parser.add_argument(
        "--system-message",
        default=None,
        help="Optional system message passed to query().",
    )
    parser.add_argument(
        "--agent-type",
        default="codex",
        help="Agent type for agentapi server (default: codex).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3284,
        help="Port for managed agentapi server.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="End-to-end query timeout seconds.",
    )
    parser.add_argument(
        "--workdir-base",
        default=".cyberian-smoke",
        help="Working directory base for managed server logs/state.",
    )
    parser.add_argument(
        "--expect",
        default="CYBERIAN_OK",
        help="Expected marker to assert in response output.",
    )
    parser.add_argument(
        "--external-server",
        action="store_true",
        help="Use an already-running agentapi server (manage_server=False).",
    )
    parser.add_argument(
        "--skip-permissions",
        action="store_true",
        default=True,
        help="Pass skip-permissions flag when starting managed agent server.",
    )
    parser.add_argument(
        "--no-skip-permissions",
        action="store_false",
        dest="skip_permissions",
        help="Do not pass skip-permissions flag to managed agent server.",
    )
    return parser


def main() -> int:
    """Run smoke test."""
    args = build_parser().parse_args()
    manage_server = not args.external_server

    if manage_server:
        print("[smoke] creating agent (managed-server mode)")
    else:
        print("[smoke] creating agent (external-server mode; expecting existing agentapi)")
    print(f"[smoke] model={args.model} agent_type={args.agent_type} port={args.port}")

    agent = create_cyberian_agent(
        model=args.model,
        agent_type=args.agent_type,
        port=args.port,
        skip_permissions=args.skip_permissions,
        manage_server=manage_server,
        workdir_base=args.workdir_base,
        timeout=args.timeout,
    )

    started = time.time()
    try:
        response = agent.query(args.query, system_message=args.system_message)
    except Exception as exc:
        print(f"[smoke] ERROR: query failed: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - started
    print(f"[smoke] success in {elapsed:.1f}s")
    print("[smoke] response:")
    print(response)

    if args.expect and args.expect not in response:
        print(
            f"[smoke] ERROR: expected marker '{args.expect}' not found in response.",
            file=sys.stderr,
        )
        return 2

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
