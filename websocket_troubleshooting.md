# WebSocket Frame Queuing - Troubleshooting Guide

## Problem
Browser displays frames with 10+ second delay, while backend logs show no lag.

## Root Cause
WebSocket messages queue up when sent faster than the browser can consume them. The `asyncio.run_coroutine_threadsafe()` queues messages without checking backpressure.

## Solution: Backpressure Control

### Key Changes to `run.py`

1. **Reduce FPS from 80 to 30**
```python
# Line ~106
self.ws_frame_interval = 1.0 / 30.0  # 30 FPS max (reduced from 80)
```

2. **Add pending send tracking**
```python
# Line ~109
self.pending_frame_send = False
```

3. **Skip frames when previous send is pending** (in `_broadcast_frame_ws`)
```python
def _broadcast_frame_ws(self):
    if self.rendered_frame is None:
        return

    # Skip if previous frame send is still pending (backpressure)
    if self.pending_frame_send:
        return

    # ... rest of the method
```

4. **Reduce JPEG quality**
```python
# In _broadcast_frame_ws
success, buffer = cv2.imencode('.jpg', self.rendered_frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 60])  # Reduced from 80
```

5. **Track send completion** (replace `_broadcast_ws_binary`)
```python
def _broadcast_ws_binary(self, data: bytes):
    """Broadcast binary data to all WebSocket clients."""
    if not hasattr(self, 'ws_loop') or self.ws_loop is None:
        return

    with self.ws_lock:
        if not self.ws_clients:
            return

        dead_clients = set()
        futures = []

        for client in self.ws_clients:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    client.send(data),
                    self.ws_loop
                )
                futures.append((client, future))
            except Exception:
                dead_clients.add(client)

        # Mark as pending
        self.pending_frame_send = True

        # Schedule callback to clear pending flag when all sends complete
        def on_sends_complete():
            for client, future in futures:
                try:
                    future.result(timeout=0.1)
                except Exception:
                    dead_clients.add(client)
            self.pending_frame_send = False
            with self.ws_lock:
                self.ws_clients -= dead_clients

        # Run completion check in background thread
        from threading import Thread
        Thread(target=on_sends_complete, daemon=True).start()
```

6. **Match MJPEG stream FPS**
```python
# In do_GET /stream handler
time.sleep(0.033)  # ~30 FPS (changed from 0.01)
```

## Why This Works

- **Frame dropping** instead of **frame queuing**: Old frames are discarded when network/browser can't keep up
- **Backpressure awareness**: Waits for previous send to complete before sending next frame
- **Reduced bandwidth**: Lower FPS + lower JPEG quality = smaller data transfer
- **No indefinite queue buildup**: Messages are dropped rather than accumulated

## Alternative Solutions

1. **Use WebSocket write_limit parameter** (websockets library)
```python
server = await websockets.serve(
    self._ws_handler,
    '0.0.0.0',
    self.ws_port,
    write_limit=2**16,  # Limit write buffer size
)
```

2. **Check WebSocket buffer size before sending**
```python
if client.transport.get_write_buffer_size() < MAX_BUFFER:
    await client.send(data)
```

3. **Use asyncio Queue with maxsize**
```python
frame_queue = asyncio.Queue(maxsize=2)  # Only keep 2 frames max
```
