import asyncio
import logging
import time
from enum import Enum
from typing import Optional

import websockets
from websockets import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, InvalidState


class _ConnState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSING = "closing"


class MoexClient:
    """Low-level WebSocket connection manager for MOEX."""

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        reconnect_event: Optional[asyncio.Event] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._url = "ws://127.0.0.1:3366/router/"
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_event = reconnect_event
        self._logger = logger or logging.getLogger(__name__)

        self._ws: Optional[WebSocketClientProtocol] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_ping_ts: Optional[float] = None
        self._state: _ConnState = _ConnState.DISCONNECTED
        self._connect_lock = asyncio.Lock()

    async def connect(self, timeout: float = 10.0) -> None:
        """Establish a WebSocket connection and start reader/heartbeat tasks."""

        async with self._connect_lock:
            if self._state in (_ConnState.CONNECTED, _ConnState.CONNECTING, _ConnState.CLOSING):
                self._log("connect_skip", f"Already {self._state.value}")
                return

            self._state = _ConnState.CONNECTING
            try:
                self._log("connect_start", f"Connecting to {self._url}")
                self._ws = await asyncio.wait_for(
                    websockets.connect(self._url, ping_interval=None, close_timeout=5),
                    timeout=timeout,
                )
                self._state = _ConnState.CONNECTED
                self._log("connect_success", "Connected")
                self._reader_task = asyncio.create_task(self._reader_loop(), name="moex-ws-reader")
                if self._heartbeat_interval:
                    self._heartbeat_task = asyncio.create_task(
                        self._heartbeat_loop(), name="moex-ws-heartbeat"
                    )
            except asyncio.TimeoutError as exc:
                self._log("connect_timeout", "Connect timeout", error=repr(exc))
                await self._cleanup()
                raise
            except Exception as exc:  # broad to ensure cleanup
                self._log("connect_error", "Connect error", error=repr(exc))
                await self._cleanup()
                raise
            finally:
                if self._state == _ConnState.CONNECTING:
                    self._state = _ConnState.DISCONNECTED

    async def disconnect(self) -> None:
        """Close the WebSocket and stop background tasks (idempotent)."""

        if self._state in (_ConnState.DISCONNECTED, _ConnState.CLOSING):
            return
        await self._do_close(cancel_reader=True, reason="manual_disconnect")

    def is_connected(self) -> bool:
        """Report connection state."""

        return bool(self._ws and not self._ws.closed and self._state == _ConnState.CONNECTED)

    async def ping(self, timeout: float = 5.0, disconnect_on_failure: bool = True) -> bool:
        """Send ping and mark disconnected on failure."""

        if not self.is_connected():
            self._log("ping_skip", "Ping skipped: not connected")
            return False
        try:
            pong_waiter = self._ws.ping()
            self._last_ping_ts = time.time()
            await asyncio.wait_for(pong_waiter, timeout=timeout)
            self._log("ping_ok", "Ping successful", last_ping_ts=self._last_ping_ts)
            return True
        except (asyncio.TimeoutError, ConnectionClosed, InvalidState) as exc:
            self._log("ping_fail", "Ping failed", error=repr(exc))
            if disconnect_on_failure:
                await self._do_close(cancel_reader=False, reason="ping_fail")
            return False
        except Exception as exc:
            self._log("ping_error", "Unexpected ping error", error=repr(exc))
            if disconnect_on_failure:
                await self._do_close(cancel_reader=False, reason="ping_error")
            return False

    async def _reader_loop(self) -> None:
        """Continuously read messages; stops on close/errors."""

        try:
            assert self._ws is not None
            async for message in self._ws:
                self._log("recv", "Message received", message_preview=str(message)[:128])
        except asyncio.CancelledError:
            self._log("reader_cancel", "Reader cancelled")
            raise
        except ConnectionClosed as exc:
            self._log("reader_closed", "Connection closed", code=exc.code, reason=exc.reason)
        except Exception as exc:
            self._log("reader_error", "Reader error", error=repr(exc))
        finally:
            if self._state == _ConnState.CONNECTED and self._reconnect_event:
                self._reconnect_event.set()
            if self._state == _ConnState.CONNECTED:
                await self._do_close(cancel_reader=False, reason="reader_exit")

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings while connected."""

        try:
            while self._state == _ConnState.CONNECTED:
                await asyncio.sleep(self._heartbeat_interval)
                if self.is_connected():
                    await self.ping(disconnect_on_failure=False)
        except asyncio.CancelledError:
            self._log("heartbeat_cancel", "Heartbeat cancelled")
            raise
        except Exception as exc:
            self._log("heartbeat_error", "Heartbeat error", error=repr(exc))

    async def _do_close(self, cancel_reader: bool, reason: str) -> None:
        """Centralized close routine to avoid double-cleanup races."""
        if self._state == _ConnState.CLOSING:
            return
        self._state = _ConnState.CLOSING

        await self._cancel_task(self._heartbeat_task)
        if cancel_reader:
            await self._cancel_task(self._reader_task, skip_current=True)

        if self._ws and not self._ws.closed:
            try:
                await asyncio.wait_for(self._ws.close(), timeout=5)
                self._log("disconnect", "WebSocket closed", reason=reason)
            except Exception as exc:
                self._log("disconnect_error", "Error during close", error=repr(exc), reason=reason)

        await self._cleanup()
        self._state = _ConnState.DISCONNECTED

    async def _cleanup(self) -> None:
        """Reset state after close/error."""
        self._last_ping_ts = None
        self._reader_task = None
        self._heartbeat_task = None
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def _cancel_task(self, task: Optional[asyncio.Task], skip_current: bool = False) -> None:
        if task and not task.done() and not (skip_current and task is asyncio.current_task()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _log(self, event: str, message: str, **extra: object) -> None:
        """Structured logging helper."""
        payload = {"event": event, **extra}
        self._logger.info(message, extra={"context": payload})
