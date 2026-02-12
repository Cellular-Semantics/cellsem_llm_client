"""MCP tool discovery via the ``mcp`` SDK and LiteLLM's experimental MCP client.

Provides ``MCPToolSource``, a sync context manager that connects to an MCP server,
discovers available tools, and wraps them as ``Tool`` objects whose handlers dispatch
calls back to the MCP session running in a background thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from litellm.experimental_mcp_client import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from cellsem_llm_client.tools.tool import Tool

logger = logging.getLogger(__name__)


class MCPToolSource:
    """Discovers MCP tools and wraps them as ``Tool`` objects for ``query_unified``.

    Uses the ``mcp`` SDK for transport/session management and LiteLLM's
    ``experimental_mcp_client.load_mcp_tools`` for schema conversion. The MCP
    session runs in a background thread with its own event loop so callers get a
    synchronous context-manager interface.

    Args:
        server: URL for SSE/streamable-HTTP servers, or command/path for stdio servers.
        transport: Transport type (``"auto"``, ``"sse"``, ``"stdio"``,
            ``"streamable_http"``). ``"auto"`` infers from *server*.
        **kwargs: Extra keyword arguments forwarded to the transport client
            (e.g. ``headers``, ``timeout``).

    Example::

        with MCPToolSource("https://example.com/mcp") as source:
            result = agent.query_unified(message="...", tools=source.tools)
    """

    def __init__(
        self,
        server: str,
        transport: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._server = server
        self._transport = transport
        self._kwargs = kwargs

        self._tools: list[Tool] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        # Signals the background loop to shut down
        self._shutdown = asyncio.Event()

    # ------------------------------------------------------------------
    # Transport detection
    # ------------------------------------------------------------------

    def _detect_transport(self) -> str:
        """Return the resolved transport type.

        Returns:
            One of ``"sse"``, ``"stdio"``, or ``"streamable_http"``.
        """
        if self._transport != "auto":
            return self._transport

        if self._server.startswith(("http://", "https://")):
            return "sse"
        return "stdio"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MCPToolSource:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-source"
        )
        self._thread.start()

        # Wait for the background loop to finish connecting
        self._ready.wait(timeout=60)
        if self._error is not None:
            raise RuntimeError(
                f"MCPToolSource failed to connect to {self._server}"
            ) from self._error
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._shutdown.set)
        if self._thread is not None:
            self._thread.join(timeout=15)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
        self._thread = None
        self._session = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[Tool]:
        """List of discovered ``Tool`` objects."""
        return list(self._tools)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Entry point for the daemon thread — runs the async session."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_lifecycle())
        except Exception as exc:
            logger.debug("MCP background loop exited: %s", exc)

    async def _async_lifecycle(self) -> None:
        """Connect, discover tools, then keep the session alive until shutdown."""
        transport_type = self._detect_transport()

        try:
            if transport_type == "sse":
                await self._lifecycle_sse()
            elif transport_type == "streamable_http":
                await self._lifecycle_streamable_http()
            else:
                await self._lifecycle_stdio()
        except Exception as exc:
            self._error = exc
            self._ready.set()

    async def _lifecycle_sse(self) -> None:
        """SSE transport lifecycle."""
        async with sse_client(self._server, **self._kwargs) as (read, write):
            async with ClientSession(read, write) as session:
                await self._init_session(session)
                await self._shutdown.wait()

    async def _lifecycle_streamable_http(self) -> None:
        """Streamable HTTP transport lifecycle."""
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(self._server, **self._kwargs) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await self._init_session(session)
                await self._shutdown.wait()

    async def _lifecycle_stdio(self) -> None:
        """Stdio transport lifecycle."""
        parts = self._server.split()
        params = StdioServerParameters(command=parts[0], args=parts[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await self._init_session(session)
                await self._shutdown.wait()

    async def _init_session(self, session: ClientSession) -> None:
        """Initialise the MCP session, discover tools, and signal ready."""
        self._session = session
        await session.initialize()

        openai_tools: list[dict[str, Any]] = await load_mcp_tools(
            session, format="openai"
        )  # type: ignore[assignment]
        self._tools = [self._wrap_tool(t) for t in openai_tools]

        self._ready.set()
        logger.info(
            "MCPToolSource connected to %s — discovered %d tool(s)",
            self._server,
            len(self._tools),
        )

    # ------------------------------------------------------------------
    # Tool wrapping
    # ------------------------------------------------------------------

    def _wrap_tool(self, openai_tool: dict[str, Any]) -> Tool:
        """Convert an OpenAI-format tool dict into a ``Tool`` with an MCP handler."""
        func = openai_tool["function"]
        name = func["name"]

        return Tool(
            name=name,
            description=func.get("description", ""),
            parameters=func.get("parameters", {}),
            handler=self._make_handler(name),
        )

    def _make_handler(self, tool_name: str) -> Callable[[dict[str, Any]], str | None]:
        """Build a sync handler that dispatches to ``session.call_tool``."""

        def handler(args: dict[str, Any]) -> str | None:
            if self._session is None or self._loop is None or self._loop.is_closed():
                raise RuntimeError("MCP session is not active")

            future: concurrent.futures.Future[str | None] = concurrent.futures.Future()

            async def _call() -> None:
                try:
                    assert self._session is not None
                    result = await self._session.call_tool(tool_name, args)
                    # Extract text from result content blocks
                    texts: list[str] = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                        else:
                            texts.append(json.dumps(block.model_dump()))
                    future.set_result("\n".join(texts) if texts else None)
                except Exception as exc:
                    future.set_exception(exc)

            self._loop.call_soon_threadsafe(asyncio.ensure_future, _call())
            return future.result(timeout=120)

        return handler
