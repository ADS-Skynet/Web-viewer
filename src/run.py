"""
ZMQ-based Web Viewer

Separate process that:
1. Receives data from vehicle via ZMQ (frames, detections, state)
2. Draws overlays on laptop (offloads vehicle CPU)
3. Serves web interface for browser viewing
4. Sends commands back to vehicle via ZMQ

Orchestrates sub-modules: overlay, websocket_server, http_server, handlers.
"""

import json
import time
from threading import Thread, Lock, Event
from typing import Optional, Dict, Any

import cv2
import numpy as np

from common.communication import (
    ViewerSubscriber,
    ActionPublisher,
    DetectionData,
    VehicleState,
    ParameterPublisher,
)
from common.visualization import LKASVisualizer
from common.config import ConfigManager

from viewer.constants import FPS, Streaming, get_target_config
from .overlay import OverlayRenderer
from .websocket_server import WebSocketServer
from .http_server import ViewerHTTPServer
from .handlers import MessageHandler


class ViewerState:
    """
    Shared mutable state accessed by all viewer components.

    Thread safety is handled via render_lock (for rendered_frame) and
    display_lock (for visualization toggles).
    """

    def __init__(self, config, target: str, vehicle_url: str,
                 img_source: str, img_quality: int):
        # Identity / URLs (read-only after init)
        self.target = target
        self.vehicle_url = vehicle_url
        self.img_source = img_source
        self.img_quality = img_quality

        # Detection method from config
        self.detection_method = config.detection_method

        # Latest data from vehicle
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_metadata: Dict[str, Any] = {}
        self.latest_frame_jpeg_bytes: Optional[bytes] = None
        self.latest_detection: Optional[DetectionData] = None
        self.latest_state: Optional[VehicleState] = None

        # Rendered frame with overlays
        self.rendered_frame: Optional[np.ndarray] = None
        self.render_lock = Lock()

        # Visualization layer toggles
        self.show_raw_image = True
        self.show_lanes = True
        self.show_hud = False
        self.show_segmentation = True
        self.display_lock = Lock()

        self.camera_offset_x = config.camera.offset_x

        # Lifecycle flag
        self.running = False

        # ZMQ subscriber reference (set by orchestrator after creation)
        self.subscriber: Optional[ViewerSubscriber] = None


class ZMQWebViewer:
    """
    Web viewer that receives vehicle data via ZMQ and draws overlays.

    Runs on laptop. Vehicle CPU stays free!
    """

    def __init__(
        self,
        vehicle_url: str = "tcp://localhost:5557",
        action_url: str = "tcp://localhost:5558",
        parameter_bind_url: str = "tcp://*:5559",
        web_port: int = 8080,
        verbose: bool = False,
        lkas_mode: bool = True,
        target: str = "simulation",
        img_source: str = "jpeg",
        img_quality: int = 75,
    ):
        self.verbose = verbose
        self.web_port = web_port
        self.ws_port = web_port + 1

        # Load config once
        config = ConfigManager.load()

        # Shared state
        self.state = ViewerState(config, target, vehicle_url, img_source, img_quality)

        # ZMQ communication
        self.subscriber = ViewerSubscriber(vehicle_url)
        self.action_publisher = ActionPublisher(action_url)
        self.parameter_publisher = ParameterPublisher(
            bind_url=parameter_bind_url,
            connect_mode=lkas_mode,
        )
        self.state.subscriber = self.subscriber

        # Sub-components
        self.renderer = OverlayRenderer(LKASVisualizer(), verbose=verbose)

        self.message_handler = MessageHandler(
            state=self.state,
            parameter_publisher=self.parameter_publisher,
            action_publisher=self.action_publisher,
            verbose=verbose,
        )

        self.ws_server = WebSocketServer(
            port=self.ws_port,
            message_handler=self.message_handler,
            verbose=verbose,
        )

        self.http_server = ViewerHTTPServer(
            port=web_port,
            message_handler=self.message_handler,
            state=self.state,
            verbose=verbose,
        )

        # Frame dropping mechanism
        self.frame_ready_event = Event()
        self.frames_received = 0
        self.frames_rendered = 0
        self.frames_dropped = 0

        # Frame rate limiting for WebSocket broadcast
        self.last_ws_frame_time = 0
        self.ws_frame_interval = 1.0 / FPS.WEBSOCKET_MAX
        self.status_broadcast_interval = 1.0 / FPS.STATUS_BROADCAST
        self.zmq_poll_interval = 1.0 / FPS.ZMQ_POLL

        # Banner
        print(f"\n{'='*60}")
        print("ZMQ Web Viewer - Laptop Side (WebSocket Edition)")
        print(f"{'='*60}")
        print(f"  Target: {target.upper()}")
        print(f"  Detection: {self.state.detection_method.upper()}, Segmentation Mask: {'shown' if self.state.show_segmentation else 'hidden'}")
        print(f"  Receiving from: {vehicle_url}")
        print(f"  Sending actions to: {action_url}")
        print(f"  Parameter server: {parameter_bind_url} ({'connect' if lkas_mode else 'bind'} mode)")
        print(f"  Web interface: http://localhost:{web_port}")
        print(f"  WebSocket server: ws://localhost:{self.ws_port}")
        print(f"  Image source: {img_source}")
        print(f"  Image quality: {img_quality}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start all threads and servers."""
        self.state.running = True

        # ZMQ callbacks
        self.subscriber.register_frame_callback(self._on_frame_received)
        self.subscriber.register_detection_callback(self._on_detection_received)
        self.subscriber.register_state_callback(self._on_state_received)

        # Render thread
        Thread(target=self._render_loop, daemon=True).start()

        # Servers
        self.http_server.start()
        self.ws_server.start()

        # Background loops
        Thread(target=self._zmq_poll_loop, daemon=True).start()
        Thread(target=self._status_broadcast_loop, daemon=True).start()

        print("[Viewer] Started (WebSocket Mode)")
        print(f"  Open: http://localhost:{self.web_port}")
        print("  Press Ctrl+C to stop\n")

    def stop(self):
        """Shut down everything."""
        self.state.running = False
        self.ws_server.stop()
        self.http_server.stop()
        self.subscriber.close()
        self.action_publisher.close()
        self.parameter_publisher.close()
        print("[Viewer] Stopped")

    def run(self):
        """Block until Ctrl+C."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping viewer...")
            self.stop()

    # ------------------------------------------------------------------
    # ZMQ Callbacks
    # ------------------------------------------------------------------

    def _on_frame_received(self, image: np.ndarray, metadata: Dict):
        """Store frame and signal render thread."""
        self.state.latest_frame = image
        self.state.latest_frame_metadata = metadata
        self.state.latest_frame_jpeg_bytes = metadata.get('original_jpeg_bytes')
        self.frames_received += 1

        if not self.subscriber.state_received:
            self.subscriber.state_received = True

        if self.verbose:
            frame_id = metadata.get('frame_id', 'N/A')
            frame_timestamp = metadata.get('timestamp', 0)
            latency_ms = (time.time() - frame_timestamp) * 1000 if frame_timestamp > 0 else 0
            if isinstance(frame_id, int) and frame_id % 30 == 0:
                print(f"[Frame] #{frame_id} | Latency: {latency_ms:.1f}ms | Decode: {metadata.get('decode_time_ms', 0):.1f}ms")

        self.frame_ready_event.set()

    def _on_detection_received(self, detection: DetectionData):
        self.state.latest_detection = detection

    def _on_state_received(self, vehicle_state: VehicleState):
        self.state.latest_state = vehicle_state

    # ------------------------------------------------------------------
    # Thread Loops
    # ------------------------------------------------------------------

    def _render_loop(self):
        """Dedicated render thread with automatic frame dropping."""
        last_stats_time = time.time()

        while self.state.running:
            # frame_arrived = self.frame_ready_event.wait(timeout=0.1)

            # if frame_arrived and self.state.latest_frame is not None:
            if self.state.latest_frame is not None:
                self.frame_ready_event.clear()
                frames_before = self.frames_received

                # Render
                output = self.renderer.render(self.state)

                # Store rendered frame
                with self.state.render_lock:
                    self.state.rendered_frame = output

                self.frames_rendered += 1

                # Broadcast to WebSocket clients
                self._broadcast_frame()

                # Track dropped frames
                dropped = self.frames_received - frames_before - 1
                if dropped > 0:
                    self.frames_dropped += dropped
                    if self.verbose:
                        print(f"  [Frame Drop] {dropped} frame(s) | Total: {self.frames_dropped}")

            # Periodic stats
            if self.verbose and time.time() - last_stats_time > 10:
                drop_rate = (self.frames_dropped / max(self.frames_received, 1)) * 100
                print(
                    f"[Render Stats] Received: {self.frames_received} | "
                    f"Rendered: {self.frames_rendered} | "
                    f"Dropped: {self.frames_dropped} ({drop_rate:.1f}%) | "
                    f"WS clients: {self.ws_server.client_count}"
                )
                last_stats_time = time.time()

        print("[Render] Loop stopped")

    def _zmq_poll_loop(self):
        """ZMQ polling loop."""
        print("[ZMQ] Polling loop started")
        while self.state.running:
            self.subscriber.poll()
            time.sleep(self.zmq_poll_interval)
        print("[ZMQ] Polling loop stopped")

    def _status_broadcast_loop(self):
        """Periodically broadcast status to WebSocket clients."""
        print("[Status] Broadcast loop started")
        while self.state.running:
            self._broadcast_status()
            time.sleep(self.status_broadcast_interval)
        print("[Status] Broadcast loop stopped")

    # ------------------------------------------------------------------
    # WebSocket Broadcasting
    # ------------------------------------------------------------------

    def _broadcast_frame(self):
        """Encode and broadcast rendered frame to WebSocket clients."""
        if self.state.rendered_frame is None:
            return

        # Frame rate limiting
        now = time.time()
        if now - self.last_ws_frame_time < self.ws_frame_interval:
            return
        self.last_ws_frame_time = now

        if self.ws_server.client_count == 0:
            return

        # Try to reuse original JPEG when frame is unmodified
        frame_bytes = None
        with self.state.display_lock:
            unmodified = (
                self.state.show_raw_image
                and not self.state.show_lanes
                and not self.state.show_hud
                and not self.state.show_segmentation
            )

        if unmodified and self.state.latest_frame_jpeg_bytes is not None:
            frame_bytes = self.state.latest_frame_jpeg_bytes
        else:
            # Determine encode quality
            fmt = self.state.latest_frame_metadata.get('format', 'jpeg')
            quality = self.state.img_quality if fmt == 'raw_rgb' else 100

            success, buf = cv2.imencode(
                '.jpg', self.state.rendered_frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            if not success:
                return
            frame_bytes = buf.tobytes()

        self.ws_server.broadcast_binary(frame_bytes)

    def _broadcast_status(self):
        """Build and broadcast status JSON to WebSocket clients."""
        status = {
            'type': 'status',
            'paused': self.subscriber.paused,
            'state_received': self.subscriber.state_received,
            'timestamp': time.time(),
        }

        if self.state.latest_state:
            status['speed_kmh'] = self.state.latest_state.speed_kmh
            status['steering'] = self.state.latest_state.steering
            status['throttle'] = self.state.latest_state.throttle
            status['brake'] = self.state.latest_state.brake

        if self.state.latest_detection:
            status['detection_time_ms'] = self.state.latest_detection.processing_time_ms
            status['departure_status'] = self.state.latest_detection.departure_status or 'no_lanes'
            status['lateral_offset_m'] = self.state.latest_detection.lateral_offset_meters
            status['heading_angle_deg'] = self.state.latest_detection.heading_angle_deg

        self.ws_server.broadcast_text(json.dumps(status))


# ==================================================================
# CLI Entry Point
# ==================================================================

def main():
    """Main entry point for ZMQ web viewer."""
    import argparse

    common_config = ConfigManager.load()
    comm = common_config.communication

    parser = argparse.ArgumentParser(description="ZMQ Web Viewer - Laptop Side")
    parser.add_argument('--target', type=str, default='vehicle',
                        choices=['simulation', 'vehicle'],
                        help="Target: 'simulation' (CARLA) or 'vehicle' (Jetracer)")
    parser.add_argument('--vehicle', type=str, default=None,
                        help="ZMQ URL to receive vehicle data (overrides target preset)")
    parser.add_argument('--actions', type=str, default=None,
                        help="ZMQ URL to send actions (overrides target preset)")
    parser.add_argument('--parameters', type=str, default=None,
                        help="ZMQ URL to send parameter updates (overrides target preset)")
    parser.add_argument('--port', type=int, default=common_config.visualization.web_port,
                        help=f"HTTP port (default: {common_config.visualization.web_port})")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose logging")
    parser.add_argument('--simulation-mode', action='store_true',
                        help="Bind as server instead of connecting to LKAS broker")

    args = parser.parse_args()

    # Target presets
    target_config = get_target_config(args.target)
    broadcast_host = target_config.get('broadcast_host', comm.zmq_broadcast_host)

    vehicle_url = args.vehicle or f"tcp://{broadcast_host}:{comm.zmq_broadcast_port}"
    action_url = args.actions or f"tcp://{broadcast_host}:{comm.zmq_action_port}"
    param_url = args.parameters or f"tcp://{broadcast_host}:{comm.zmq_parameter_port}"
    lkas_mode = not args.simulation_mode

    img_source = "jpeg" if not common_config.streaming.raw_rgb else "raw_rgb"
    img_quality = common_config.streaming.jpeg_quality if img_source == "jpeg" else Streaming.JPEG_QUALITY

    print(f"\n[Viewer] Starting with target: {args.target.upper()}")

    viewer = ZMQWebViewer(
        vehicle_url=vehicle_url,
        action_url=action_url,
        parameter_bind_url=param_url,
        web_port=args.port,
        verbose=args.verbose,
        lkas_mode=lkas_mode,
        target=args.target,
        img_source=img_source,
        img_quality=img_quality,
    )

    viewer.start()
    viewer.run()


if __name__ == "__main__":
    main()
