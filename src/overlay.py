"""
Overlay Renderer

Composites visualization layers onto camera frames.
Orchestrates LKASVisualizer drawing primitives and manages
viewer-specific concerns (base64 decode, state extraction, timing).
"""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Dict, Any

import cv2
import numpy as np

from common.types import LaneDepartureStatus
from common.visualization import LKASVisualizer

if TYPE_CHECKING:
    from common.communication import DetectionData
    from .run import ViewerState


class OverlayRenderer:
    """
    Renders a composited frame from the current ViewerState.

    All drawing primitives are delegated to LKASVisualizer.
    This class handles layer composition, state extraction,
    and viewer-specific logic (base64 decode, performance timing).
    """

    def __init__(self, visualizer: LKASVisualizer, verbose: bool = False):
        self.visualizer = visualizer
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, state: ViewerState) -> np.ndarray:
        """
        Composite all enabled layers into a single output frame.

        Args:
            state: Shared viewer state with frame data, detection, and toggles.

        Returns:
            Rendered RGB image with overlays applied.
        """
        if self.verbose:
            render_start = time.time()

        with state.display_lock:
            show_raw = state.show_raw_image
            show_lanes = state.show_lanes
            show_hud = state.show_hud
            show_segmentation = state.show_segmentation

        # Step 1: Base layer
        if show_raw:
            output = state.latest_frame.copy()
        else:
            output = np.zeros_like(state.latest_frame)

        # Step 2: DL segmentation mask
        if show_segmentation and state.latest_detection:
            self._apply_segmentation(output, state.latest_detection)

        # Step 3: Lane polynomial overlays
        if show_lanes:
            try:
                self._apply_lane_overlays(output, state)
            except Exception as e:
                if not hasattr(self, '_overlay_error_logged'):
                    print(f"[Overlay] Warning: Lane overlay failed: {e}")
                    self._overlay_error_logged = True

        # Step 4: HUD on top
        if show_hud:
            self._draw_hud(output, state)

        if self.verbose:
            total_ms = (time.time() - render_start) * 1000
            if total_ms > 30:
                layers = []
                if show_raw: layers.append("raw")
                if show_segmentation: layers.append("seg")
                if show_lanes: layers.append("lanes")
                if show_hud: layers.append("hud")
                print(f"  [Render Warn] Total: {total_ms:.1f}ms | Layers: {'+'.join(layers)}")

        return output

    # ------------------------------------------------------------------
    # Segmentation (viewer-specific: base64 decode from ZMQ transport)
    # ------------------------------------------------------------------

    def _apply_segmentation(self, output: np.ndarray, detection: DetectionData):
        """Decode base64 segmentation mask and delegate drawing."""
        try:
            mask_bytes = base64.b64decode(detection.segmentation_mask_base64)
            mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
            seg_mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            if seg_mask is not None:
                self.visualizer.draw_segmentation(output, seg_mask, alpha=0.35)
        except Exception as e:
            print(f"[Overlay] Warning: Failed to decode segmentation mask: {e}")

    # ------------------------------------------------------------------
    # Lane Overlays
    # ------------------------------------------------------------------

    def _apply_lane_overlays(self, output: np.ndarray, state: ViewerState):
        """Draw polynomial lane boundaries from DL detection."""
        if state.latest_detection:
            self.visualizer.draw_polynomials(
                output,
                left_poly=getattr(state.latest_detection, 'left_poly', None),
                right_poly=getattr(state.latest_detection, 'right_poly', None),
                center_poly=getattr(state.latest_detection, 'center_poly', None),
                left_confidence=getattr(state.latest_detection, 'left_confidence', 0.0),
                right_confidence=getattr(state.latest_detection, 'right_confidence', 0.0),
                camera_offset_x=state.camera_offset_x,
            )

    # ------------------------------------------------------------------
    # HUD (state extraction + delegate)
    # ------------------------------------------------------------------

    def _draw_hud(self, output: np.ndarray, state: ViewerState):
        """Extract telemetry from state and delegate to visualizer."""
        if state.latest_state:
            vehicle_telemetry = {
                'speed_kmh': state.latest_state.speed_kmh,
                'throttle': state.latest_state.throttle,
                'position': state.latest_state.position,
                'rotation': state.latest_state.rotation,
            }
            metrics = self._calculate_lane_metrics(state)
            modified = self.visualizer.draw_hud(
                output, metrics,
                show_steering=True,
                steering_value=state.latest_state.steering,
                vehicle_telemetry=vehicle_telemetry,
            )
            np.copyto(output, modified)

        if self.verbose and state.latest_frame_metadata:
            self._draw_performance_overlay(output, state)

    def _draw_performance_overlay(self, output: np.ndarray, state: ViewerState):
        """Draw frame latency / detection timing in bottom-left corner."""
        frame_timestamp = state.latest_frame_metadata.get('timestamp', 0)
        frame_id = state.latest_frame_metadata.get('frame_id', 'N/A')
        decode_time = state.latest_frame_metadata.get('decode_time_ms', 0)

        if frame_timestamp <= 0:
            return

        current_time = time.time()
        latency_ms = (current_time - frame_timestamp) * 1000

        if latency_ms < 100:
            color = (0, 255, 0)
        elif latency_ms < 500:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        y_pos = output.shape[0] - 60
        cv2.putText(output, f"Frame: {frame_id} | Latency: {latency_ms:.1f}ms",
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if decode_time > 0:
            decode_color = (0, 255, 255) if decode_time < 30 else (0, 0, 255)
            cv2.putText(output, f"Decode: {decode_time:.1f}ms",
                        (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, decode_color, 1)

        if state.latest_detection:
            cv2.putText(output, f"Detection: {state.latest_detection.processing_time_ms:.1f}ms",
                        (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_lane_metrics(state: ViewerState) -> Dict[str, Any]:
        """Extract lane departure metrics from detection data."""
        if not state.latest_detection:
            return {
                'departure_status': LaneDepartureStatus.NO_LANES,
                'lateral_offset_meters': None,
                'heading_angle_deg': None,
                'lane_width_pixels': None,
            }

        departure_status = LaneDepartureStatus.NO_LANES
        if state.latest_detection.departure_status:
            try:
                departure_status = LaneDepartureStatus(state.latest_detection.departure_status)
            except ValueError:
                departure_status = LaneDepartureStatus.NO_LANES

        return {
            'departure_status': departure_status,
            'lateral_offset_meters': state.latest_detection.lateral_offset_meters,
            'heading_angle_deg': state.latest_detection.heading_angle_deg,
            'lane_width_pixels': state.latest_detection.lane_width_pixels,
        }
