# Viewer Module

**Real-time WebSocket-powered web interface for remote monitoring and control of autonomous vehicles.**

## Overview

The viewer is a standalone process that provides a rich web interface for monitoring the LKAS system and simulation. It receives data from the LKAS broker via ZMQ and serves an interactive web dashboard with **WebSocket streaming** for ultra-low latency.

## Key Features

- **WebSocket Real-Time Streaming:**
  - Binary frame transmission (no base64 overhead!)
  - Instant parameter updates
  - Live status push notifications
  - ~50-300ms total latency
- **Rich Visualizations:**
  - Lane detection overlays
  - Vehicle telemetry (speed, steering, position)
  - HUD with status indicators
  - Real-time FPS and latency monitoring
- **Interactive Controls:**
  - Pause/Resume simulation
  - Respawn vehicle
  - Adjust detection parameters live
  - Tune PID controller in real-time
- **Performance Optimized:**
  - Rendering offloaded to laptop (vehicle CPU stays free!)
  - Frame rate limiting (30 FPS)
  - Efficient binary WebSocket frames
  - Automatic reconnection

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LKAS Broker                          │
│              (ZMQ Message Hub)                          │
└───────┬─────────────────────┬───────────────────────────┘
        │                     │
   Frames/Status         Actions/Params
     (port 5557)          (port 5558/5559)
        │                     │
┌───────┴─────────────────────┴───────────────────────────┐
│                  Viewer Process                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────────────────┐   │
│  │ ZMQ Layer    │      │  Web Servers             │   │
│  │ - Subscribe  │  ->  │  - HTTP (port 8080)      │   │
│  │ - Publish    │      │  - WebSocket (port 8081) │   │
│  └──────────────┘      └──────────────────────────┘   │
│         │                         │                     │
│         │                         │                     │
│  ┌──────┴─────────────────────────┴─────────────┐     │
│  │          Rendering Engine                     │     │
│  │  - Lane overlays                              │     │
│  │  - HUD rendering                              │     │
│  │  - Metric visualization                       │     │
│  └───────────────────────────────────────────────┘     │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                    WebSocket
                  (Binary Frames)
                          │
              ┌───────────┴───────────┐
              │   Web Browser         │
              │  (localhost:8080)     │
              └───────────────────────┘
```

## Quick Start

### Prerequisites

Ensure LKAS and simulation are running:
```bash
# Terminal 1: CARLA
./CarlaUE4.sh

# Terminal 2: LKAS
lkas --method cv --broadcast

# Terminal 3: Simulation
simulation --broadcast
```

### Start Viewer

```bash
# Default (port 8080)
viewer

# Custom port
viewer --port 9090

# Verbose mode
viewer --verbose
```

### Access Web Interface

Open browser: **http://localhost:8080**

You should see:
- **Connection:** Green "Connected" status
- **FPS:** Real-time frame rate
- **Latency:** Frame-to-frame time
- **Live video:** Lane detection overlay

## Configuration

### Port Configuration

In `config.yaml`:
```yaml
visualization:
  web_port: 8080  # HTTP server port (WebSocket will be port+1)
```

Or via CLI:
```bash
viewer --port 8080  # WebSocket will automatically use 8081
```

### ZMQ Configuration

```yaml
zmq:
  viewer:
    # Data reception
    vehicle_data_url: tcp://localhost:5557    # Receive from LKAS broker

    # Command transmission
    action_url: tcp://localhost:5558          # Send actions to broker
    parameter_url: tcp://localhost:5559       # Send parameters to broker
```

## Web Interface Features

### Video Stream
- **High-quality visualization:**
  - Lane detection overlays (blue lines)
  - Lane fill (green transparent)
  - HUD with telemetry
- **Real-time metrics:**
  - Detection processing time
  - Vehicle speed
  - Steering angle

### Control Panel

**Actions:**
- **🔄 Respawn Vehicle** - Reset vehicle to spawn point
- **⏸ Pause / ▶ Resume** - Control simulation state

**Keyboard Shortcuts:**
- `R` - Respawn vehicle
- `Space` - Toggle pause/resume

### Live Parameter Tuning

#### Detection Parameters
- **Canny Low Threshold** (1-150)
- **Canny High Threshold** (50-255)
- **Hough Threshold** (1-150)
- **Hough Min Line Length** (10-150)
- **Smoothing Factor** (0-1)

#### Decision (PID) Parameters
- **Kp** - Proportional gain (0-2)
- **Kd** - Derivative gain (0-1)
- **Base Throttle** (0-0.5)
- **Min Throttle** (0-0.2)
- **Steer Threshold** (0-0.5)

**All parameter changes apply instantly!**

### Status Indicators

- **Connection Status:**
  - 🟢 Green - Connected
  - 🟠 Orange - Connecting
  - 🔴 Red - Disconnected
- **FPS Display:** Real-time frame rate
- **Latency Display:** Frame-to-frame latency in milliseconds

## WebSocket Protocol

### Communication Flow

1. **Browser connects to WebSocket** (port 8081)
2. **Server pushes binary frames** (JPEG compressed)
3. **Server pushes status updates** (JSON, every 500ms)
4. **Browser sends actions/parameters** (JSON)

### Message Types

#### Server → Browser (Binary)
```
<JPEG bytes>  # Raw JPEG image data
```

#### Server → Browser (JSON)
```json
{
  "type": "status",
  "paused": false,
  "state_received": true,
  "speed_kmh": 8.0,
  "steering": 0.1,
  "detection_time_ms": 12.3
}
```

#### Browser → Server (JSON)
```json
// Action
{
  "type": "action",
  "action": "pause"  // pause, resume, respawn
}

// Parameter update
{
  "type": "parameter",
  "category": "detection",  // detection, decision
  "parameter": "canny_low",
  "value": 50.0
}
```

## Performance

### Latency Breakdown

| Component | Latency |
|-----------|---------|
| ZMQ reception | ~5ms |
| Rendering | ~10-20ms |
| JPEG encoding | ~5-10ms |
| WebSocket transmission | ~5-10ms |
| Browser decode | ~10-20ms |
| **Total** | **~50-100ms** |

### Optimizations

1. **Binary WebSocket Frames:**
   - No base64 encoding (33% overhead eliminated!)
   - Direct JPEG transmission
   - Blob URL creation for instant rendering

2. **Frame Rate Limiting:**
   - Max 30 FPS to prevent flooding
   - Configurable via `ws_frame_interval`

3. **JPEG Quality:**
   - Quality: 80 (balanced)
   - Lower = faster transmission
   - Higher = better quality

4. **Efficient Rendering:**
   - Overlays rendered on laptop CPU
   - Vehicle/simulation CPU stays free
   - Complex HUD doesn't impact vehicle performance

## Module Structure

```
viewer/
├── run.py                        # Main entry point
├── frontend.html                 # Web interface (HTML/CSS/JS)
├── test_websocket.py             # WebSocket testing utility
│
└── README.md                     # This file
```

## Development

### Testing WebSocket Connection

```bash
# In one terminal: start viewer
viewer

# In another terminal: test connection
python3 src/viewer/test_websocket.py

# Should output:
# ✓ Connected to ws://localhost:8081
# ✓ Sent test message
# ✓ Received 10 messages
```

### Adding Custom Overlays

Edit `run.py` in the `_render_frame()` method:

```python
def _render_frame(self):
    # ... existing rendering code ...

    # Add custom overlay
    cv2.putText(
        output,
        f"Custom Info: {self.custom_data}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),  # Yellow text
        2
    )

    self.rendered_frame = output
```

### Custom Frontend Styling

Edit `frontend.html` CSS:

```css
/* Custom theme colors */
body {
    background: #2c2c2c;  /* Darker background */
}

.badge {
    background: #FF5722;  /* Orange badge */
}
```

### Debugging

Enable verbose logging:
```bash
viewer --verbose
```

Check WebSocket connection:
```bash
# Check if ports are open
ss -tlnp | grep '808[01]'

# Monitor WebSocket traffic
wscat -c ws://localhost:8081
```

Browser console (F12):
```javascript
// Check WebSocket status
console.log(ws.readyState);  // 1 = OPEN

// Monitor frame reception
ws.onmessage = (e) => console.log('Frame:', e.data.size);
```

## Troubleshooting

### WebSocket Won't Connect

**Symptoms:**
```
Connection: Disconnected (red)
Browser console: WebSocket connection failed
```

**Fixes:**
1. Check viewer is running:
   ```bash
   ps aux | grep viewer
   ```

2. Verify WebSocket server started:
   ```
   ✓ WebSocket server started on port 8081
   ```

3. Test connection:
   ```bash
   python3 src/viewer/test_websocket.py
   ```

4. Check firewall:
   ```bash
   sudo ufw allow 8081
   ```

### High Latency (>1 second)

**Symptoms:**
```
Latency: 5000+ms
FPS: <10
```

**Fixes:**
1. Reduce JPEG quality:
   ```python
   # In run.py, line ~290
   cv2.imencode('.jpg', ..., [cv2.IMWRITE_JPEG_QUALITY, 70])
   ```

2. Increase frame rate limit:
   ```python
   # In run.py, line ~104
   self.ws_frame_interval = 1.0 / 60.0  # 60 FPS instead of 30
   ```

3. Reduce camera resolution:
   ```yaml
   # In config.yaml
   carla:
     camera:
       width: 480
       height: 360
   ```

### Parameters Not Updating

**Symptoms:**
```
Slider moves but LKAS behavior doesn't change
```

**Fixes:**
1. Check LKAS broker is receiving parameters:
   ```bash
   # In LKAS terminal, should see:
   [Parameter] detection.canny_low = 50.0
   ```

2. Verify parameter port:
   ```bash
   ss -tlnp | grep 5559
   ```

3. Check parameter category name matches:
   ```javascript
   // Should be 'detection' or 'decision'
   updateParam('detection', 'canny_low', 50)
   ```

### Viewer Shows Old/Frozen Frames

**Symptoms:**
```
Video stream frozen or showing old data
```

**Fixes:**
1. Restart viewer:
   ```bash
   # Ctrl+C to stop, then restart
   viewer
   ```

2. Check ZMQ connection:
   ```bash
   # Should show data port listening
   ss -tlnp | grep 5557
   ```

3. Verify LKAS is broadcasting:
   ```bash
   # LKAS should be started with --broadcast flag
   lkas --method cv --broadcast
   ```

## Advanced Usage

### Multiple Viewers

Run multiple viewer instances on different ports:

```bash
# Viewer 1
viewer --port 8080

# Viewer 2 (on same or different machine)
viewer --port 9090

# Viewer 3
viewer --port 7070
```

All viewers receive the same data from LKAS broker!

### Remote Viewing

If viewer is on a different machine than LKAS:

```yaml
# In config.yaml on viewer machine
zmq:
  viewer:
    vehicle_data_url: tcp://192.168.1.100:5557  # LKAS machine IP
    action_url: tcp://192.168.1.100:5558
    parameter_url: tcp://192.168.1.100:5559
```

Then access viewer:
```
http://<viewer-machine-ip>:8080
```

### Custom Frame Rate Limiting

Edit `run.py`:

```python
# Line ~104-106
self.ws_frame_interval = 1.0 / 60.0  # 60 FPS (lower latency)
# or
self.ws_frame_interval = 1.0 / 15.0  # 15 FPS (lower bandwidth)
```

### Recording Sessions

```bash
# TODO: Add recording functionality
viewer --record session.mp4
```

## Integration with Other Modules

### With LKAS
```
LKAS Broker → (port 5557) → Viewer [ZMQ subscription]
Viewer → (port 5558) → LKAS Broker [Actions]
Viewer → (port 5559) → LKAS Broker [Parameters]
```

### With Simulation
```
Simulation → LKAS Broker → Viewer [Data flow]
Viewer → LKAS Broker → Simulation [Control flow]
```

## Performance Benchmarks

Tested on:
- **CPU:** Intel i7-10750H
- **RAM:** 16GB
- **Network:** localhost (loopback)

| Configuration | FPS | Latency | CPU Usage |
|--------------|-----|---------|-----------|
| 640x480, Q=80, 30fps | 30 | 50-100ms | ~15% |
| 640x480, Q=95, 30fps | 30 | 80-150ms | ~18% |
| 800x600, Q=80, 30fps | 30 | 70-120ms | ~20% |
| 640x480, Q=80, 60fps | 60 | 40-80ms | ~25% |

## References

- [WebSocket Protocol](https://datatracker.ietf.org/doc/html/rfc6455)
- [ZMQ Guide](https://zguide.zeromq.org/)
- [LKAS Module](../lkas/README.md)
- [Simulation Module](../simulation/README.md)
