"""
ZMQ-based Web Viewer

Separate process that:
1. Receives data from vehicle via ZMQ (frames, detections, state)
2. Draws overlays on laptop (offloads vehicle CPU)
3. Serves web interface for browser viewing
4. Sends commands back to vehicle via ZMQ

This replaces the threaded web viewer with a proper process-based architecture.
"""

import numpy as np
import cv2
import zmq
import time
import json
import asyncio
import websockets
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread, Lock, Event
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass

# Import ZMQ communication and visualization tools from skynet-common
from common.communication import (
    ViewerSubscriber,
    ActionPublisher,
    DetectionData,
    VehicleState,
    ParameterPublisher,
)
from common.visualization import LKASVisualizer
from common.types import LaneDepartureStatus
from common.config import ConfigManager

# Import module-specific constants
from viewer.constants import FPS, Streaming, get_target_config


class ZMQWebViewer:
    """
    Web viewer that receives vehicle data via ZMQ and draws overlays.

    Runs on laptop. Vehicle CPU stays free!
    """

    def __init__(self,
                 vehicle_url: str = "tcp://localhost:5557",
                 action_url: str = "tcp://localhost:5558",
                 parameter_bind_url: str = "tcp://*:5559",
                 web_port: int = 8080,
                 verbose: bool = False,
                 lkas_mode: bool = True,
                 target: str = "simulation"):
        """
        Initialize ZMQ web viewer.

        Args:
            vehicle_url: ZMQ URL to receive data from vehicle
            action_url: ZMQ URL to send actions to vehicle
            parameter_bind_url: ZMQ URL to bind/connect for parameter updates
            web_port: HTTP port for web interface
            verbose: Enable verbose HTTP request logging
            lkas_mode: If True, connect to LKAS broker (default, new architecture).
                      If False, bind as server for simulation (old architecture).
            target: Target to connect to ("simulation" or "vehicle")
        """
        self.vehicle_url = vehicle_url
        self.action_url = action_url
        self.parameter_bind_url = parameter_bind_url
        self.web_port = web_port
        self.verbose = verbose
        self.lkas_mode = lkas_mode
        self.target = target

        # ZMQ communication
        self.subscriber = ViewerSubscriber(vehicle_url)
        self.action_publisher = ActionPublisher(action_url)
        self.parameter_publisher = ParameterPublisher(
            bind_url=parameter_bind_url,
            connect_mode=lkas_mode  # Connect to LKAS broker in new architecture
        )

        # Visualization
        self.visualizer = LKASVisualizer()

        # Latest data from vehicle
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_metadata: Dict[str, Any] = {}  # Store metadata for latency display
        self.latest_frame_jpeg_bytes: Optional[bytes] = None  # Store original JPEG bytes for optimization
        self.latest_detection: Optional[DetectionData] = None
        self.latest_state: Optional[VehicleState] = None
        self.latest_metrics: Dict[str, Any] = {}

        # Rendered frame with overlays (drawn on laptop!)
        self.rendered_frame: Optional[np.ndarray] = None
        self.render_lock = Lock()

        # Frame dropping mechanism - dedicated render thread
        self.frame_ready_event = Event()
        self.render_thread: Optional[Thread] = None
        self.frames_received = 0
        self.frames_rendered = 0
        self.frames_dropped = 0

        # WebSocket clients
        self.ws_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.ws_lock = Lock()

        # HTTP server
        self.http_server: Optional[HTTPServer] = None
        self.http_thread: Optional[Thread] = None

        # WebSocket server
        self.ws_port = web_port + 1  # WebSocket on port+1 (e.g., 8081 if HTTP is 8080)
        self.ws_server = None
        self.ws_thread: Optional[Thread] = None
        self.ws_loop = None  # Will be set when WebSocket server starts
        self.ws_ready = False  # Flag to indicate WebSocket is ready

        # Frame rate limiting for WebSocket (from constants)
        self.last_ws_frame_time = 0
        self.ws_frame_interval = 1.0 / FPS.WEBSOCKET_MAX
        self.status_broadcast_interval = 1.0 / FPS.STATUS_BROADCAST
        self.zmq_poll_interval = 1.0 / FPS.ZMQ_POLL

        # Streaming quality (from constants)
        self.jpeg_quality = Streaming.JPEG_QUALITY

        # ROI overlay cache (to avoid recomputing on every frame)
        self.roi_vertices_cache: Optional[np.ndarray] = None
        self.roi_cache_frame_size: Optional[tuple] = None

        # Load ROI config parameters once during initialization from common config
        config = ConfigManager.load()
        self.roi_config = {
            'roi_bottom_left_x': config.cv_detector.roi_bottom_left_x,
            'roi_top_left_x': config.cv_detector.roi_top_left_x,
            'roi_top_right_x': config.cv_detector.roi_top_right_x,
            'roi_bottom_right_x': config.cv_detector.roi_bottom_right_x,
            'roi_top_y': config.cv_detector.roi_top_y,
        }

        # Visualization layer toggles (all independent)
        self.show_raw_image = True  # Show raw image as background
        self.show_lanes = True  # Show lane detection (ROI + lanes)
        self.show_canny = False  # Show Canny edges
        self.show_hough = False  # Show Hough lines
        self.show_hud = True  # Show HUD overlay
        self.display_lock = Lock()

        # Cache Canny/Hough parameters from config
        self.canny_low = config.cv_detector.canny_low
        self.canny_high = config.cv_detector.canny_high
        self.hough_threshold = config.cv_detector.hough_threshold
        self.hough_min_line_len = config.cv_detector.hough_min_line_len
        self.hough_max_line_gap = config.cv_detector.hough_max_line_gap
        self.hough_rho = config.cv_detector.hough_rho
        self.hough_theta = config.cv_detector.hough_theta
        self.smoothing_factor = config.cv_detector.smoothing_factor

        self.running = False

        print(f"\n{'='*60}")
        print("ZMQ Web Viewer - Laptop Side (WebSocket Edition)")
        print(f"{'='*60}")
        print(f"  Target: {target.upper()}")
        print(f"  Receiving from: {vehicle_url}")
        print(f"  Sending actions to: {action_url}")
        print(f"  Parameter server: {parameter_bind_url} ({'connect' if lkas_mode else 'bind'} mode)")
        print(f"  Web interface: http://localhost:{web_port}")
        print(f"  WebSocket server: ws://localhost:{self.ws_port}")
        print(f"{'='*60}\n")

    def start(self):
        """Start viewer (ZMQ polling + HTTP server + WebSocket server)."""
        self.running = True

        # Register ZMQ callbacks
        self.subscriber.register_frame_callback(self._on_frame_received)
        self.subscriber.register_detection_callback(self._on_detection_received)
        self.subscriber.register_state_callback(self._on_state_received)

        # Start dedicated render thread (handles frame dropping automatically)
        self.render_thread = Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

        # Start HTTP server
        self._start_http_server()

        # Start WebSocket server
        self.ws_thread = Thread(target=self._run_ws_server, daemon=True)
        self.ws_thread.start()

        # Wait for WebSocket server to be ready (max 5 seconds)
        wait_time = 0
        while not self.ws_ready and wait_time < 5:
            time.sleep(0.1)
            wait_time += 0.1

        if not self.ws_ready:
            print("⚠️  Warning: WebSocket server may not have started properly")

        # Start ZMQ polling thread
        zmq_thread = Thread(target=self._zmq_poll_loop, daemon=True)
        zmq_thread.start()

        # Start status broadcast thread
        status_thread = Thread(target=self._status_broadcast_loop, daemon=True)
        status_thread.start()

        print("✓ ZMQ Web Viewer started (WebSocket Mode)")
        print(f"  Open: http://localhost:{self.web_port}")
        print("  Press Ctrl+C to stop\n")

    def _on_frame_received(self, image: np.ndarray, metadata: Dict):
        """
        Called when new frame received from vehicle.

        Instead of rendering immediately, we signal the dedicated render thread.
        This enables automatic frame dropping when rendering can't keep up!
        """
        self.latest_frame = image
        self.latest_frame_metadata = metadata  # Store for latency calculation
        # Store original JPEG bytes if available (for optimization)
        self.latest_frame_jpeg_bytes = metadata.get('original_jpeg_bytes')
        self.frames_received += 1

        # Mark stream as active when receiving frames (not just state)
        if not self.subscriber.state_received:
            self.subscriber.state_received = True

        # Calculate latency for logging (only if verbose)
        if self.verbose:
            frame_timestamp = metadata.get('timestamp', 0)
            current_time = time.time()
            latency_ms = (current_time - frame_timestamp) * 1000 if frame_timestamp > 0 else 0
            print(f"[Frame] Received frame {metadata.get('frame_id', 'N/A')} | Latency: {latency_ms:.1f}ms | Decode: {metadata.get('decode_time_ms', 0):.1f}ms")

        # Signal render thread that new frame is ready
        # If render is still in progress, this frame will be dropped and we'll render the latest
        self.frame_ready_event.set()

    def _on_detection_received(self, detection: DetectionData):
        """Called when detection results received."""
        self.latest_detection = detection
        # DON'T render here - wait for next frame
        # This prevents duplicate rendering which was causing lag

    def _on_state_received(self, state: VehicleState):
        """Called when vehicle state received."""
        self.latest_state = state
        # Debug: Log when state is received (especially paused status)
        if state.paused is not None and self.verbose:
            print(f"[State] Received: paused={state.paused}, steering={state.steering:.3f}")
        # DON'T render here - wait for next frame
        # This prevents duplicate rendering which was causing lag

    def _process_canny_mode(self, image: np.ndarray) -> np.ndarray:
        """
        Process image for Canny edge visualization.

        All processing happens on viewer side from raw image.

        Args:
            image: Raw RGB image from vehicle

        Returns:
            Canny edge visualization (white edges on black background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # Convert back to RGB for consistent display
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges_rgb

    def _process_hough_mode(self, image: np.ndarray) -> np.ndarray:
        """
        Process image for Hough lines visualization.

        Shows only Hough lines (magenta) on black background.
        All processing happens on viewer side from raw image.

        Args:
            image: Raw RGB image from vehicle

        Returns:
            Hough lines visualization (magenta lines on black background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # Apply ROI mask to edges (same as detection uses)
        height, width = edges.shape
        mask = np.zeros_like(edges)

        # Use cached ROI config
        vertices = np.array([[
            [int(width * self.roi_config['roi_bottom_left_x']), height],
            [int(width * self.roi_config['roi_top_left_x']), int(height * self.roi_config['roi_top_y'])],
            [int(width * self.roi_config['roi_top_right_x']), int(height * self.roi_config['roi_top_y'])],
            [int(width * self.roi_config['roi_bottom_right_x']), height]
        ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap
        )

        # Create black canvas
        output = np.zeros_like(image)

        # Draw Hough lines in magenta
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Magenta color in RGB: (255, 0, 255)
                cv2.line(output, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return output

    def _render_loop(self):
        """
        Dedicated render thread - automatically drops frames when rendering can't keep up.

        How frame dropping works:
        - Waits for frame_ready_event signal from _on_frame_received()
        - Renders the latest frame (stored in self.latest_frame)
        - If multiple frames arrived during render, only the latest is rendered (others dropped)
        - This prevents lag buildup when heavy layers (canny+hough+lanes+hud) are enabled
        """
        last_stats_time = time.time()

        while self.running:
            # Wait for new frame (with timeout to check running flag periodically)
            frame_arrived = self.frame_ready_event.wait(timeout=0.1)

            if frame_arrived and self.latest_frame is not None:
                # Clear the event BEFORE rendering (important!)
                # If new frames arrive during render, event will be set again
                # and we'll immediately render the latest frame after this one finishes
                self.frame_ready_event.clear()

                # Calculate dropped frames before rendering
                frames_before = self.frames_received

                # Render the latest frame
                render_start = time.time()
                self._render_frame()
                render_time_ms = (time.time() - render_start) * 1000

                self.frames_rendered += 1

                # Check if frames arrived during rendering
                frames_after = self.frames_received
                dropped_this_render = frames_after - frames_before - 1  # -1 because we rendered 1 frame

                if dropped_this_render > 0:
                    self.frames_dropped += dropped_this_render
                    if self.verbose:
                        print(f"  [Frame Drop] Dropped {dropped_this_render} frame(s) during {render_time_ms:.1f}ms render | Total dropped: {self.frames_dropped}")

            # Print stats every 10 seconds (only if verbose)
            if self.verbose and time.time() - last_stats_time > 10:
                drop_rate = (self.frames_dropped / max(self.frames_received, 1)) * 100
                print(f"[Render Stats] Received: {self.frames_received} | Rendered: {self.frames_rendered} | Dropped: {self.frames_dropped} ({drop_rate:.1f}%)")
                last_stats_time = time.time()

        print("[Render] Render loop stopped")

    def _calculate_lane_metrics(self, frame_width: int, frame_height: int) -> Dict[str, Any]:
        """
        Extract lane departure metrics from detection data received from LKAS.

        Metrics are calculated in LKAS and sent via ZMQ, ensuring single source of truth.

        Args:
            frame_width: Frame width in pixels (unused, kept for compatibility)
            frame_height: Frame height in pixels (unused, kept for compatibility)

        Returns:
            Dictionary with metrics compatible with visualizer
        """
        # Default metrics if no detection data available
        if not self.latest_detection:
            return {
                'departure_status': LaneDepartureStatus.NO_LANES,
                'lateral_offset_meters': None,
                'heading_angle_deg': None,
                'lane_width_pixels': None
            }

        # Extract metrics from detection data (calculated by LKAS)
        departure_status = LaneDepartureStatus.NO_LANES
        if self.latest_detection.departure_status:
            # Convert string back to enum
            try:
                departure_status = LaneDepartureStatus(self.latest_detection.departure_status)
            except ValueError:
                # If invalid status string, default to NO_LANES
                departure_status = LaneDepartureStatus.NO_LANES

        return {
            'departure_status': departure_status,
            'lateral_offset_meters': self.latest_detection.lateral_offset_meters,
            'heading_angle_deg': self.latest_detection.heading_angle_deg,
            'lane_width_pixels': self.latest_detection.lane_width_pixels
        }

    def _render_frame(self):
        """
        Render frame with overlays.

        THIS RUNS ON LAPTOP, NOT VEHICLE!
        Heavy drawing operations don't impact vehicle performance.
        """
        render_start = time.time()  # MEASURE TOTAL RENDER TIME

        if self.latest_frame is None:
            return

        # Get current display settings
        with self.display_lock:
            show_raw = self.show_raw_image
            show_lanes = self.show_lanes
            show_canny = self.show_canny
            show_hough = self.show_hough
            show_hud = self.show_hud

        # Step 1: Start with raw image or black background
        if show_raw:
            output = self.latest_frame.copy()
        else:
            # Create black canvas with same dimensions
            output = np.zeros_like(self.latest_frame)

        # Step 2: Overlay Canny edges (if enabled)
        if show_canny:
            canny_result = self._process_canny_mode(self.latest_frame)
            # Blend canny edges with current output (60/40)
            output = cv2.addWeighted(output, 0.6, canny_result, 0.4, 0)

        # Step 3: Overlay Hough lines (if enabled)
        if show_hough:
            hough_result = self._process_hough_mode(self.latest_frame)
            # Overlay hough lines (magenta lines on black, so we can just add where lines exist)
            mask = np.any(hough_result > 0, axis=2)
            output[mask] = hough_result[mask]

        # Step 4: Apply lane detection overlays (ROI + detected lanes)
        # This draws on top so actual detection is visible over Canny/Hough
        if show_lanes:
            self._apply_normal_overlays(output)

        # Step 5: Add HUD on top of everything (if enabled)
        if show_hud:
            self._draw_hud_overlay(output)

        # Store rendered frame
        if self.verbose:
            store_start = time.time()
        with self.render_lock:
            self.rendered_frame = output
        if self.verbose:
            store_time_ms = (time.time() - store_start) * 1000

        # Calculate total render time (only measure if verbose)
        if self.verbose:
            total_render_time_ms = (time.time() - render_start) * 1000

            # Log timing breakdown if render is slow (>30ms, below 33 FPS capacity, not intended interval)
            if total_render_time_ms > 30:
                layers = []
                if show_raw: layers.append("raw")
                if show_lanes: layers.append("lanes")
                if show_canny: layers.append("canny")
                if show_hough: layers.append("hough")
                if show_hud: layers.append("hud")
                print(f"  [LAGGING_RENDER_WARN!] Total: {total_render_time_ms:.1f}ms | Layers: {'+'.join(layers)} | Store: {store_time_ms:.1f}ms")

        # Broadcast frame to WebSocket clients
        if self.verbose:
            ws_start = time.time()
        self._broadcast_frame_ws()
        if self.verbose:
            ws_time_ms = (time.time() - ws_start) * 1000

            # Log WebSocket timing if slow (>10ms)
            if ws_time_ms > 10:
                print(f"  [WEBSOCKET] Broadcast took {ws_time_ms:.1f}ms")

    def _apply_normal_overlays(self, output: np.ndarray):
        """
        Apply normal mode overlays (ROI and lanes only, no HUD).

        Modifies output in-place.

        Args:
            output: Image to apply overlays to
        """
        # Draw ROI overlay (green trapezoid showing detection area)
        # Calculate ROI vertices (cached to avoid config loading on every frame)
        try:
            height, width = output.shape[:2]
            current_frame_size = (height, width)

            # Recalculate ROI if frame size changed or not cached
            if self.roi_cache_frame_size != current_frame_size:
                # Use cached ROI config from initialization (no file I/O!)
                self.roi_vertices_cache = np.array([[
                    [int(width * self.roi_config['roi_bottom_left_x']), height],
                    [int(width * self.roi_config['roi_top_left_x']), int(height * self.roi_config['roi_top_y'])],
                    [int(width * self.roi_config['roi_top_right_x']), int(height * self.roi_config['roi_top_y'])],
                    [int(width * self.roi_config['roi_bottom_right_x']), height]
                ]], dtype=np.int32)
                self.roi_cache_frame_size = current_frame_size

            # Draw ROI as green polyline
            if self.roi_vertices_cache is not None:
                cv2.polylines(output, self.roi_vertices_cache, True, (0, 255, 0), 2)
        except Exception as e:
            print(f"[Viewer] Warning: Failed to draw ROI overlay: {e}")
            # If ROI drawing fails, continue without it
            pass

        # Draw lane overlays if detection available
        if self.latest_detection:
            left_lane = None
            right_lane = None

            if self.latest_detection.left_lane:
                ll = self.latest_detection.left_lane
                left_lane = (int(ll['x1']), int(ll['y1']), int(ll['x2']), int(ll['y2']))

            if self.latest_detection.right_lane:
                rl = self.latest_detection.right_lane
                right_lane = (int(rl['x1']), int(rl['y1']), int(rl['x2']), int(rl['y2']))

            # Draw lanes (note: draw_lanes returns a new image)
            modified = self.visualizer.draw_lanes(output, left_lane, right_lane, fill_lane=True)
            # Copy result back to output array
            np.copyto(output, modified)

    def _draw_hud_overlay(self, output: np.ndarray):
        """
        Draw HUD overlay (vehicle state, metrics, performance info).

        Modifies output in-place.

        Args:
            output: Image to draw HUD on
        """
        # Draw vehicle state overlay if available
        if self.latest_state:
            vehicle_telemetry = {
                'speed_kmh': self.latest_state.speed_kmh,
                'throttle': self.latest_state.throttle,
                'position': self.latest_state.position,
                'rotation': self.latest_state.rotation
            }

            # Calculate lane metrics from detection data
            height, width = output.shape[:2]
            metrics = self._calculate_lane_metrics(width, height)

            # Draw HUD with vehicle data (returns new image)
            modified = self.visualizer.draw_hud(
                output,
                metrics,
                show_steering=True,
                steering_value=self.latest_state.steering,
                vehicle_telemetry=vehicle_telemetry
            )
            # Copy result back to output array
            np.copyto(output, modified)

        # Add performance overlay (bottom-left corner) - only in verbose mode
        if self.verbose and self.latest_frame_metadata:
            frame_timestamp = self.latest_frame_metadata.get('timestamp', 0)
            frame_id = self.latest_frame_metadata.get('frame_id', 'N/A')
            decode_time = self.latest_frame_metadata.get('decode_time_ms', 0)

            if frame_timestamp > 0:
                current_time = time.time()
                latency_ms = (current_time - frame_timestamp) * 1000

                # Color-code latency: green (<100ms), yellow (100-500ms), red (>500ms)
                if latency_ms < 100:
                    color = (0, 255, 0)  # Green
                elif latency_ms < 500:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red

                # Display in bottom-left corner
                y_pos = output.shape[0] - 60

                # Frame ID and latency
                cv2.putText(
                    output,
                    f"Frame: {frame_id} | Latency: {latency_ms:.1f}ms",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

                # Decode time if available
                if decode_time > 0:
                    decode_color = (0, 255, 255) if decode_time < 30 else (0, 0, 255)
                    cv2.putText(
                        output,
                        f"Decode: {decode_time:.1f}ms",
                        (10, y_pos + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        decode_color,
                        1
                    )

                # Detection time if available
                if self.latest_detection:
                    cv2.putText(
                        output,
                        f"Detection: {self.latest_detection.processing_time_ms:.1f}ms",
                        (10, y_pos + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

    def _zmq_poll_loop(self):
        """ZMQ polling loop (runs in separate thread)."""
        print("[ZMQ] Polling loop started")

        while self.running:
            # Poll for new messages
            self.subscriber.poll()
            time.sleep(self.zmq_poll_interval)  # Small sleep to prevent busy-wait

        print("[ZMQ] Polling loop stopped")

    def _status_broadcast_loop(self):
        """Periodically broadcast status to WebSocket clients."""
        print("[Status] Broadcast loop started")

        while self.running:
            self._broadcast_status_ws()

            # # Invalidate ROI cache periodically to pick up parameter changes
            # # This allows ROI to update when user adjusts ROI parameters in viewer
            # self.roi_cache_frame_size = None

            time.sleep(self.status_broadcast_interval)  # Broadcast status at configured rate

        print("[Status] Broadcast loop stopped")

    # ============================================================
    # WebSocket Methods
    # ============================================================

    def _broadcast_frame_ws(self):
        """Broadcast rendered frame to all WebSocket clients."""
        if self.rendered_frame is None:
            return

        # Frame rate limiting - only send if enough time has passed
        current_time = time.time()
        if current_time - self.last_ws_frame_time < self.ws_frame_interval:
            return
        self.last_ws_frame_time = current_time

        # Remove disconnected clients
        with self.ws_lock:
            if not self.ws_clients:
                return  # No clients, skip encoding entirely

        # Optimization: Check if we can reuse original JPEG bytes
        # This avoids re-encoding when:
        # 1. Frame was received as JPEG from lkas (raw_rgb=false)
        # 2. No overlays are enabled (frame is unmodified)
        can_reuse_jpeg = False
        with self.display_lock:
            # Check if frame is unmodified (only raw image, no overlays)
            frame_is_unmodified = (
                self.show_raw_image and
                not self.show_lanes and
                not self.show_canny and
                not self.show_hough and
                not self.show_hud
            )

        if frame_is_unmodified and self.latest_frame_jpeg_bytes is not None:
            # Reuse original JPEG bytes (skip re-encoding entirely)
            frame_bytes = self.latest_frame_jpeg_bytes
            can_reuse_jpeg = True
            if self.verbose:
                frame_size_kb = len(frame_bytes) / 1024
                source_quality = self.latest_frame_metadata.get('jpeg_quality', 'unknown')
                print(f"  [WS JPEG] Reusing original JPEG (quality={source_quality}, size={frame_size_kb:.1f}KB) - skipped re-encoding")

        # Encode frame if we can't reuse original JPEG
        if not can_reuse_jpeg:
            # Determine quality based on source format:
            # - raw_rgb: Use viewer's quality setting (for WebSocket bandwidth control)
            # - jpeg: Use quality 100 (minimize re-encoding loss, already compressed by LKAS)
            frame_format = self.latest_frame_metadata.get('format', 'jpeg')
            if frame_format == 'raw_rgb':
                encode_quality = self.jpeg_quality  # Use viewer's setting (e.g., 75)
                quality_reason = "raw_rgb source"
            else:
                encode_quality = 100  # Max quality to minimize re-encoding loss
                quality_reason = "jpeg source (minimize re-encoding loss)"

            if self.verbose:
                encode_start = time.time()
            success, buffer = cv2.imencode('.jpg', self.rendered_frame,
                                            [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
            if self.verbose:
                encode_time_ms = (time.time() - encode_start) * 1000

            if not success:
                return

            # Send binary frame directly (no base64!)
            frame_bytes = buffer.tobytes()
            frame_size_kb = len(frame_bytes) / 1024

            # Log if JPEG encoding is slow (>10ms) or produces large files (only if verbose)
            if self.verbose and (encode_time_ms > 10 or frame_size_kb > 100):
                print(f"  [WS JPEG] Encode: {encode_time_ms:.1f}ms | Size: {frame_size_kb:.1f}KB | Quality: {encode_quality} ({quality_reason}) | Clients: {len(self.ws_clients)}")

        # Broadcast binary to all connected clients
        if self.verbose:
            broadcast_start = time.time()
        self._broadcast_ws_binary(frame_bytes)
        if self.verbose:
            broadcast_time_ms = (time.time() - broadcast_start) * 1000

            # Log if broadcasting is slow
            if broadcast_time_ms > 10:
                print(f"  [WS SEND] Broadcast to {len(self.ws_clients)} clients took {broadcast_time_ms:.1f}ms")

    def _broadcast_status_ws(self):
        """Broadcast status to all WebSocket clients."""
        status_data = {
            'type': 'status',
            'paused': self.subscriber.paused,
            'state_received': self.subscriber.state_received,
            'timestamp': time.time()
        }

        # Add vehicle state if available
        if self.latest_state:
            status_data['speed_kmh'] = self.latest_state.speed_kmh
            status_data['steering'] = self.latest_state.steering
            status_data['throttle'] = self.latest_state.throttle
            status_data['brake'] = self.latest_state.brake

        # Add detection metrics if available
        if self.latest_detection:
            status_data['detection_time_ms'] = self.latest_detection.processing_time_ms

        message = json.dumps(status_data)
        self._broadcast_ws(message)

    def _broadcast_ws(self, message: str):
        """Broadcast JSON message to all WebSocket clients."""
        # Check if WebSocket loop is ready
        if not hasattr(self, 'ws_loop') or self.ws_loop is None:
            return

        with self.ws_lock:
            dead_clients = set()
            for client in self.ws_clients:
                try:
                    # Use asyncio to send message in event loop
                    asyncio.run_coroutine_threadsafe(
                        client.send(message),
                        self.ws_loop
                    )
                except Exception:
                    # Mark client for removal
                    dead_clients.add(client)

            # Remove dead clients
            self.ws_clients -= dead_clients

    def _broadcast_ws_binary(self, data: bytes):
        """
        Broadcast binary data to all WebSocket clients.
        """
        # Check if WebSocket loop is ready
        if not hasattr(self, 'ws_loop') or self.ws_loop is None:
            return

        with self.ws_lock:
            dead_clients = set()
            for client in self.ws_clients:
                try:
                    # Send frame to client
                    asyncio.run_coroutine_threadsafe(
                        client.send(data),
                        self.ws_loop
                    )
                except Exception:
                    # Mark client for removal
                    dead_clients.add(client)

            # Remove dead clients
            self.ws_clients -= dead_clients

    async def _ws_handler(self, websocket):
        """Handle WebSocket connection."""
        # Register client
        with self.ws_lock:
            self.ws_clients.add(websocket)

        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"[WebSocket] Client connected: {client_addr}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')

                    if msg_type == 'action':
                        # Handle action (pause, resume, respawn)
                        action = data.get('action')
                        self.action_publisher.send_action(action)
                        if self.verbose:
                            print(f"[WebSocket] Action from {client_addr}: {action}")

                    elif msg_type == 'parameter':
                        # Handle parameter update
                        category = data.get('category')
                        parameter = data.get('parameter')
                        value = float(data.get('value'))

                        # Update local ROI config if it's a ROI parameter
                        if parameter.startswith('roi_'):
                            self.roi_config[parameter] = value
                            # Invalidate ROI cache to force recalculation with new parameters
                            self.roi_cache_frame_size = None
                            if self.verbose:
                                print(f"[ROI] Updated local ROI config: {parameter} = {value}")

                        # Update local CV detection parameters (used for viewer's own Canny/Hough rendering)
                        if parameter == 'canny_low':
                            self.canny_low = int(value)
                        elif parameter == 'canny_high':
                            self.canny_high = int(value)
                        elif parameter == 'hough_threshold':
                            self.hough_threshold = int(value)
                        elif parameter == 'hough_min_line_len':
                            self.hough_min_line_len = int(value)
                        elif parameter == 'hough_max_line_gap':
                            self.hough_max_line_gap = int(value)
                        elif parameter == 'smoothing_factor':
                            self.smoothing_factor = float(value)

                        if self.verbose and parameter in ['canny_low', 'canny_high', 'hough_threshold',
                                                           'hough_min_line_len', 'hough_max_line_gap',
                                                           'smoothing_factor']:
                            print(f"[CV] Updated local viewer rendering: {parameter} = {value}")

                        self.parameter_publisher.send_parameter(category, parameter, value)
                        if self.verbose:
                            print(f"[WebSocket] Parameter from {client_addr}: {category}.{parameter} = {value}")

                    elif msg_type == 'toggle':
                        # Handle visualization layer toggle (raw_image, lanes, canny, hough, hud)
                        setting = data.get('setting')
                        enabled = data.get('enabled', True)

                        with self.display_lock:
                            if setting == 'raw_image':
                                self.show_raw_image = enabled
                                if self.verbose:
                                    print(f"[WebSocket] Raw image: {'ON' if enabled else 'OFF'}")
                            elif setting == 'lanes':
                                self.show_lanes = enabled
                                if self.verbose:
                                    print(f"[WebSocket] Lanes: {'ON' if enabled else 'OFF'}")
                            elif setting == 'canny':
                                self.show_canny = enabled
                                if self.verbose:
                                    print(f"[WebSocket] Canny: {'ON' if enabled else 'OFF'}")
                            elif setting == 'hough':
                                self.show_hough = enabled
                                if self.verbose:
                                    print(f"[WebSocket] Hough: {'ON' if enabled else 'OFF'}")
                            elif setting == 'hud':
                                self.show_hud = enabled
                                if self.verbose:
                                    print(f"[WebSocket] HUD: {'ON' if enabled else 'OFF'}")

                except json.JSONDecodeError:
                    print(f"[WebSocket] Invalid JSON from {client_addr}")
                except Exception as e:
                    print(f"[WebSocket] Error processing message from {client_addr}: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Unregister client
            with self.ws_lock:
                self.ws_clients.discard(websocket)
            print(f"[WebSocket] Client disconnected: {client_addr}")

    def _run_ws_server(self):
        """Run WebSocket server in separate thread."""
        # Create new event loop for this thread
        self.ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ws_loop)

        async def start_server():
            try:
                server = await websockets.serve(
                    self._ws_handler,
                    '0.0.0.0',
                    self.ws_port,
                    ping_interval=20,
                    ping_timeout=20
                )
                print(f"✓ WebSocket server started on port {self.ws_port}")
                self.ws_ready = True  # Signal that server is ready
                return server
            except Exception as e:
                print(f"[WebSocket] Server startup error: {e}")
                import traceback
                traceback.print_exc()
                self.ws_ready = False
                return None

        try:
            # Start the server
            server = self.ws_loop.run_until_complete(start_server())

            if server:
                # Run the event loop forever
                self.ws_loop.run_forever()
        except Exception as e:
            print(f"[WebSocket] Server error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.ws_loop.close()

    # ============================================================
    # HTTP Server Methods
    # ============================================================

    def _start_http_server(self):
        """Start HTTP server for web interface."""
        viewer_self = self

        class ViewerRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Custom logging based on verbose flag
                if viewer_self.verbose:
                    # Verbose mode: log all requests with details
                    message = format % args
                    print(f"[HTTP] {message}")
                else:
                    # Normal mode: log important messages only
                    message = format % args
                    if "code 404" in message or "code 500" in message or "error" in message.lower():
                        print(f"[HTTP] {message}")
                    # Suppress routine logs (200, 204)

            def log_request(self, code='-', size='-'):
                """Log an HTTP request with detailed information in verbose mode."""
                if viewer_self.verbose:
                    # Extract client info
                    client_ip = self.client_address[0]
                    client_port = self.client_address[1]

                    # Get request details
                    method = self.command
                    path = self.path
                    protocol = self.request_version

                    # Format log message
                    print(f"[HTTP] {method} {path} {protocol} - Client: {client_ip}:{client_port} - Status: {code} - Size: {size}")
                else:
                    # Use default logging (will be filtered by log_message)
                    super().log_request(code, size)

            def do_POST(self):
                """Handle POST requests for actions and parameters."""
                if self.path == '/action':
                    try:
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        data = json.loads(post_data.decode('utf-8'))
                        action = data.get('action')

                        # Send action to vehicle via ZMQ
                        # The viewer will update its footer when it receives the state update from simulation
                        viewer_self.action_publisher.send_action(action)

                        # Send success response
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({'status': 'ok', 'action': action})
                        self.wfile.write(response.encode())

                    except Exception as e:
                        print(f"[Action] Error: {e}")
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({'status': 'error', 'message': str(e)})
                        self.wfile.write(response.encode())

                elif self.path == '/parameter':
                    try:
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        data = json.loads(post_data.decode('utf-8'))

                        category = data.get('category')  # 'detection' or 'decision'
                        parameter = data.get('parameter')
                        value = float(data.get('value'))

                        # Update local ROI config if it's a ROI parameter
                        if parameter.startswith('roi_'):
                            viewer_self.roi_config[parameter] = value
                            # Invalidate ROI cache to force recalculation with new parameters
                            viewer_self.roi_cache_frame_size = None
                            if viewer_self.verbose:
                                print(f"[ROI] Updated local ROI config: {parameter} = {value}")

                        # Update local CV detection parameters (used for viewer's own Canny/Hough rendering)
                        if parameter == 'canny_low':
                            viewer_self.canny_low = int(value)
                        elif parameter == 'canny_high':
                            viewer_self.canny_high = int(value)
                        elif parameter == 'hough_threshold':
                            viewer_self.hough_threshold = int(value)
                        elif parameter == 'hough_min_line_len':
                            viewer_self.hough_min_line_len = int(value)
                        elif parameter == 'hough_max_line_gap':
                            viewer_self.hough_max_line_gap = int(value)
                        elif parameter == 'smoothing_factor':
                            viewer_self.smoothing_factor = float(value)

                        if viewer_self.verbose and parameter in ['canny_low', 'canny_high', 'hough_threshold',
                                                                  'hough_min_line_len', 'hough_max_line_gap',
                                                                  'smoothing_factor']:
                            print(f"[CV] Updated local viewer rendering: {parameter} = {value}")

                        # Send parameter update via ZMQ
                        viewer_self.parameter_publisher.send_parameter(category, parameter, value)

                        # Send success response
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({
                            'status': 'ok',
                            'category': category,
                            'parameter': parameter,
                            'value': value
                        })
                        self.wfile.write(response.encode())

                    except Exception as e:
                        print(f"[Parameter] Error: {e}")
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({'status': 'error', 'message': str(e)})
                        self.wfile.write(response.encode())

                else:
                    self.send_error(404)

            def do_GET(self):
                if self.path == '/':
                    # Serve HTML page
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html = self._get_html()
                    self.wfile.write(html.encode())

                elif self.path == '/stream':
                    # Serve MJPEG stream
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.end_headers()

                    frame_count = 0
                    try:
                        while viewer_self.running:
                            if viewer_self.rendered_frame is not None:
                                frame_count += 1

                                # Encode frame as JPEG with high quality
                                success, buffer = cv2.imencode('.jpg', viewer_self.rendered_frame,
                                                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                                if success:
                                    frame_bytes = buffer.tobytes()

                                    # Send frame
                                    self.wfile.write(b'--jpgboundary\r\n')
                                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                    self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode())
                                    self.wfile.write(frame_bytes)
                                    self.wfile.write(b'\r\n')
                                else:
                                    print(f"[HTTP] Failed to encode frame!")
                            else:
                                if frame_count == 0:
                                    time.sleep(1)
                                    continue

                            time.sleep(0.01)  # ~100 FPS
                    except Exception as e:
                        print(f"[HTTP] Stream ended: {e}")

                elif self.path == '/status':
                    # Status endpoint - returns current pause state
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    status = {
                        'paused': viewer_self.subscriber.paused,
                        'state_received': viewer_self.subscriber.state_received
                    }
                    self.wfile.write(json.dumps(status).encode())

                elif self.path == '/health':
                    # Health check endpoint
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK\n')

                elif self.path == '/favicon.ico':
                    self.send_response(204)
                    self.end_headers()
                else:
                    print(f"[HTTP] 404 - Path not found: {self.path}")
                    self.send_error(404)

            def _get_html(self):
                # Read HTML template from separate file
                template_path = Path(__file__).parent / 'viewer.html'
                with open(template_path, 'r') as f:
                    template = f.read()

                # Substitute dynamic values
                # Hide respawn button for real vehicle (only applicable for simulation)
                respawn_display = "inline-block" if viewer_self.target == "simulation" else "none"
                return template.format(
                    vehicle_url=viewer_self.vehicle_url,
                    target=viewer_self.target.upper(),
                    respawn_display=respawn_display,
                    # ROI parameters
                    roi_bottom_left_x=viewer_self.roi_config['roi_bottom_left_x'],
                    roi_top_left_x=viewer_self.roi_config['roi_top_left_x'],
                    roi_top_right_x=viewer_self.roi_config['roi_top_right_x'],
                    roi_bottom_right_x=viewer_self.roi_config['roi_bottom_right_x'],
                    roi_top_y=viewer_self.roi_config['roi_top_y'],
                    # CV detection parameters
                    canny_low=viewer_self.canny_low,
                    canny_high=viewer_self.canny_high,
                    hough_threshold=viewer_self.hough_threshold,
                    hough_min_line_len=viewer_self.hough_min_line_len,
                    hough_max_line_gap=viewer_self.hough_max_line_gap,
                    smoothing_factor=viewer_self.smoothing_factor,
                    # Throttle policy
                    throttle_base=ConfigManager.load().throttle_policy.base
                )

        # Start HTTP server with error handling wrapper
        def serve_with_error_handling():
            try:
                self.http_server.serve_forever()
            except Exception as e:
                print(f"[HTTP] Server thread crashed: {e}")
                import traceback
                traceback.print_exc()

        try:
            self.http_server = ThreadingHTTPServer(('0.0.0.0', self.web_port), ViewerRequestHandler)
            self.http_thread = Thread(target=serve_with_error_handling, daemon=True)
            self.http_thread.start()

            # Give server a moment to start
            time.sleep(0.2)

            # Verify thread is still running
            if self.http_thread.is_alive():
                print(f"✓ HTTP server started on port {self.web_port}")
                print(f"  Server listening on: http://0.0.0.0:{self.web_port}")
                print(f"  Local access: http://localhost:{self.web_port}")
                print(f"  Remote access (via VSCode): http://localhost:{self.web_port}")
                print(f"")
            else:
                print(f"✗ HTTP server thread died immediately!")
        except Exception as e:
            print(f"✗ Failed to start HTTP server: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop viewer."""
        self.running = False

        # Close WebSocket connections
        with self.ws_lock:
            for client in self.ws_clients:
                try:
                    asyncio.run_coroutine_threadsafe(
                        client.close(),
                        self.ws_loop
                    )
                except:
                    pass
            self.ws_clients.clear()

        # Stop HTTP server
        if self.http_server:
            self.http_server.shutdown()

        # Close ZMQ connections
        self.subscriber.close()
        self.action_publisher.close()
        self.parameter_publisher.close()

        print("✓ ZMQ Web Viewer stopped (WebSocket Mode)")

    def run(self):
        """Run viewer (blocks until Ctrl+C)."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping viewer...")
            self.stop()


def main():
    """Main entry point for ZMQ web viewer."""
    import argparse

    # Load common config for default communication settings
    common_config = ConfigManager.load()
    comm = common_config.communication

    # Build default URLs from common config
    default_vehicle_url = f"tcp://{comm.zmq_broadcast_host}:{comm.zmq_broadcast_port}"
    default_action_url = f"tcp://{comm.zmq_broadcast_host}:{comm.zmq_action_port}"
    default_param_url = f"tcp://*:{comm.zmq_parameter_port}"

    parser = argparse.ArgumentParser(description="ZMQ Web Viewer - Laptop Side")
    parser.add_argument('--target', type=str, default='vehicle',
                       choices=['simulation', 'vehicle'],
                       help="Target to connect to: 'simulation' (CARLA) or 'vehicle' (Jetracer). Default: simulation")
    parser.add_argument('--vehicle', type=str, default=None,
                       help=f"ZMQ URL to receive vehicle data (overrides target preset)")
    parser.add_argument('--actions', type=str, default=None,
                       help=f"ZMQ URL to send actions (overrides target preset)")
    parser.add_argument('--parameters', type=str, default=None,
                       help=f"ZMQ URL to send parameter updates (overrides target preset)")
    parser.add_argument('--port', type=int, default=common_config.visualization.web_port,
                       help=f"HTTP port for web interface (default: {common_config.visualization.web_port})")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose HTTP request logging")
    parser.add_argument('--simulation-mode', action='store_true',
                       help="Use simulation mode (bind as server). Default is LKAS mode (connect to broker).")

    args = parser.parse_args()

    # Apply target presets from constants
    target_config = get_target_config(args.target)

    # Use preset values or fall back to defaults
    broadcast_host = target_config.get('broadcast_host', comm.zmq_broadcast_host)
    broadcast_port = comm.zmq_broadcast_port
    action_port = comm.zmq_action_port
    parameter_port = comm.zmq_parameter_port

    # Build URLs from preset (or override with explicit args)
    vehicle_url = args.vehicle or f"tcp://{broadcast_host}:{broadcast_port}"
    action_url = args.actions or f"tcp://{broadcast_host}:{action_port}"
    param_url = args.parameters or f"tcp://*:{parameter_port}"

    # Determine mode: LKAS mode (connect to broker) is default
    lkas_mode = not args.simulation_mode

    print(f"\n[Viewer] Starting with target: {args.target.upper()}")

    # Create and run viewer
    viewer = ZMQWebViewer(
        vehicle_url=vehicle_url,
        action_url=action_url,
        parameter_bind_url=param_url,
        web_port=args.port,
        verbose=args.verbose,
        lkas_mode=lkas_mode,
        target=args.target
    )

    viewer.start()
    viewer.run()


# Main entry point
if __name__ == "__main__":
    main()
