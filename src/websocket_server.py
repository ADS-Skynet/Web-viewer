"""
WebSocket Server

Manages WebSocket connections, handles incoming messages (actions,
parameters, toggles), and broadcasts frames + status to all clients.
"""

from __future__ import annotations

import asyncio
import json
import time
from threading import Thread, Lock
from typing import TYPE_CHECKING, Set, Optional, Dict, Any

import websockets

if TYPE_CHECKING:
    from .handlers import MessageHandler


class WebSocketServer:
    """
    Async WebSocket server running in its own thread.

    Delegates incoming messages to a MessageHandler and exposes
    broadcast helpers for the render and status loops.
    """

    def __init__(
        self,
        port: int,
        message_handler: MessageHandler,
        verbose: bool = False,
    ):
        self.port = port
        self.handler = message_handler
        self.verbose = verbose

        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self._lock = Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self._server = None  # websockets server instance
        self._ready = False
        self._send_futures: Dict[int, asyncio.Future] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Launch WebSocket server in a daemon thread."""
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

        # Wait up to 5 s for the server to become ready
        waited = 0.0
        while not self._ready and waited < 5:
            time.sleep(0.1)
            waited += 0.1

        if not self._ready:
            print("[WebSocket] Warning: server may not have started properly")

    def stop(self):
        """Gracefully close all connections, then stop the event loop."""
        if not self._loop or not self._loop.is_running():
            return

        async def _shutdown():
            # Close every client connection and wait for completion
            with self._lock:
                clients = list(self.clients)
                self.clients.clear()
            close_tasks = [asyncio.ensure_future(c.close()) for c in clients]
            if close_tasks:
                await asyncio.wait(close_tasks, timeout=2)

            # Close the server so it stops accepting new connections
            if self._server:
                self._server.close()
                await self._server.wait_closed()

            # Now safe to stop the loop
            self._loop.stop()

        future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        # Wait for shutdown to finish (with timeout so we don't hang forever)
        try:
            future.result(timeout=5)
        except Exception:
            # If shutdown times out, force-stop the loop
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def client_count(self) -> int:
        return len(self.clients)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def broadcast_text(self, message: str):
        """Send a JSON string to every connected client."""
        if not self._loop:
            return

        with self._lock:
            dead = set()
            for client in self.clients:
                try:
                    asyncio.run_coroutine_threadsafe(client.send(message), self._loop)
                except Exception:
                    dead.add(client)
            self.clients -= dead

    def broadcast_binary(self, data: bytes):
        """
        Send binary data to every connected client.

        Uses per-client send tracking: if the previous send hasn't
        completed, skip this frame for that client (prevents queue buildup).
        """
        if not self._loop:
            if self.verbose:
                print("[WebSocket] Warning: loop not ready, skipping binary broadcast")
            return

        with self._lock:
            dead = set()
            for client in self.clients:
                try:
                    cid = id(client)
                    prev = self._send_futures.get(cid)
                    if prev is not None and not prev.done():
                        continue  # drop this frame for this slow client

                    future = asyncio.run_coroutine_threadsafe(
                        client.send(data), self._loop,
                    )
                    self._send_futures[cid] = future
                except Exception:
                    dead.add(client)

            for client in dead:
                self._send_futures.pop(id(client), None)
            self.clients -= dead

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self):
        """Thread entry point — creates an event loop and starts serving."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _start():
            try:
                server = await websockets.serve(
                    self._ws_handler, '0.0.0.0', self.port,
                    ping_interval=20, ping_timeout=20,
                )
                print(f"[WebSocket] Server started on port {self.port}")
                print(f"  Browser should connect to: ws://<your-ip>:{self.port}")
                self._ready = True
                return server
            except Exception as e:
                print(f"[WebSocket] Server startup error: {e}")
                import traceback
                traceback.print_exc()
                self._ready = False
                return None

        try:
            self._server = self._loop.run_until_complete(_start())
            if self._server:
                self._loop.run_forever()
        except Exception as e:
            print(f"[WebSocket] Server error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cancel any remaining tasks before closing the loop
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    async def _ws_handler(self, websocket):
        """Handle a single WebSocket connection lifecycle."""
        with self._lock:
            self.clients.add(websocket)

        addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"[WebSocket] Client connected: {addr}")

        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                    msg_type = data.get('type')

                    if msg_type == 'action':
                        self.handler.handle_action(data.get('action'))

                    elif msg_type == 'parameter':
                        self.handler.handle_parameter(
                            data.get('category'),
                            data.get('parameter'),
                            float(data.get('value')),
                        )

                    elif msg_type == 'toggle':
                        self.handler.handle_toggle(
                            data.get('setting'),
                            data.get('enabled', True),
                        )

                except json.JSONDecodeError:
                    print(f"[WebSocket] Invalid JSON from {addr}")
                except Exception as e:
                    print(f"[WebSocket] Error processing message from {addr}: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            cid = id(websocket)
            with self._lock:
                self.clients.discard(websocket)
            self._send_futures.pop(cid, None)
            print(f"[WebSocket] Client disconnected: {addr}")
