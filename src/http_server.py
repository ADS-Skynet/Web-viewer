"""
HTTP Server

Serves the web interface (HTML page), MJPEG stream, status endpoint,
and handles POST requests for actions and parameters.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Callable, Optional

import cv2

from common.config import ConfigManager

if TYPE_CHECKING:
    from .handlers import MessageHandler
    from .run import ViewerState


class ViewerHTTPServer:
    """
    Threaded HTTP server for the viewer web interface.

    Delegates POST logic to a shared MessageHandler so it stays
    in sync with the WebSocket handler.
    """

    def __init__(
        self,
        port: int,
        message_handler: MessageHandler,
        state: ViewerState,
        verbose: bool = False,
    ):
        self.port = port
        self.handler = message_handler
        self.state = state
        self.verbose = verbose

        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the HTTP server in a daemon thread."""
        handler_ref = self.handler
        state_ref = self.state
        verbose_ref = self.verbose

        class RequestHandler(BaseHTTPRequestHandler):
            """Inner request handler with closure over viewer refs."""

            # ---- Logging ----

            def log_message(self, fmt, *args):
                if verbose_ref:
                    print(f"[HTTP] {fmt % args}")
                else:
                    msg = fmt % args
                    if "code 404" in msg or "code 500" in msg or "error" in msg.lower():
                        print(f"[HTTP] {msg}")

            def log_request(self, code='-', size='-'):
                if verbose_ref:
                    print(
                        f"[HTTP] {self.command} {self.path} {self.request_version} "
                        f"- {self.client_address[0]}:{self.client_address[1]} "
                        f"- Status: {code} - Size: {size}"
                    )
                else:
                    super().log_request(code, size)

            # ---- POST ----

            def do_POST(self):
                if self.path == '/action':
                    self._handle_action_post()
                elif self.path == '/parameter':
                    self._handle_parameter_post()
                else:
                    self.send_error(404)

            def _handle_action_post(self):
                try:
                    data = self._read_json()
                    handler_ref.handle_action(data.get('action'))
                    self._json_response(200, {'status': 'ok', 'action': data.get('action')})
                except Exception as e:
                    print(f"[Action] Error: {e}")
                    self._json_response(500, {'status': 'error', 'message': str(e)})

            def _handle_parameter_post(self):
                try:
                    data = self._read_json()
                    result = handler_ref.handle_parameter(
                        data.get('category'),
                        data.get('parameter'),
                        float(data.get('value')),
                    )
                    self._json_response(200, result)
                except Exception as e:
                    print(f"[Parameter] Error: {e}")
                    self._json_response(500, {'status': 'error', 'message': str(e)})

            # ---- GET ----

            def do_GET(self):
                if self.path == '/':
                    self._serve_html()
                elif self.path == '/stream':
                    self._serve_mjpeg()
                elif self.path == '/status':
                    self._serve_status()
                elif self.path == '/health':
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

            # ---- GET helpers ----

            def _serve_html(self):
                try:
                    html = _build_html(state_ref)
                    payload = html.encode()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as e:
                    import traceback
                    error_msg = f"Failed to render viewer page: {e}\n{traceback.format_exc()}"
                    print(f"[HTTP] ERROR: {error_msg}")
                    self.send_response(500)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(error_msg.encode())

            def _serve_mjpeg(self):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.end_headers()

                frame_count = 0
                try:
                    while state_ref.running:
                        if state_ref.rendered_frame is not None:
                            frame_count += 1
                            success, buf = cv2.imencode(
                                '.jpg', state_ref.rendered_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95],
                            )
                            if success:
                                frame_bytes = buf.tobytes()
                                self.wfile.write(b'--jpgboundary\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode())
                                self.wfile.write(frame_bytes)
                                self.wfile.write(b'\r\n')
                            else:
                                print("[HTTP] Failed to encode frame!")
                        else:
                            if frame_count == 0:
                                time.sleep(1)
                                continue
                        time.sleep(0.01)
                except Exception as e:
                    print(f"[HTTP] Stream ended: {e}")

            def _serve_status(self):
                status = {
                    'paused': state_ref.subscriber.paused,
                    'state_received': state_ref.subscriber.state_received,
                }
                self._json_response(200, status)

            # ---- Utilities ----

            def _read_json(self) -> dict:
                length = int(self.headers['Content-Length'])
                return json.loads(self.rfile.read(length).decode('utf-8'))

            def _json_response(self, code: int, data: dict):
                payload = json.dumps(data).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payload)

        # Launch server
        def _serve():
            try:
                self._server.serve_forever()
            except Exception as e:
                print(f"[HTTP] Server thread crashed: {e}")
                import traceback
                traceback.print_exc()

        try:
            self._server = ThreadingHTTPServer(('0.0.0.0', self.port), RequestHandler)
            self._thread = Thread(target=_serve, daemon=True)
            self._thread.start()
            time.sleep(0.2)

            if self._thread.is_alive():
                print(f"[HTTP] Server started on port {self.port}")
                print(f"  Local access: http://localhost:{self.port}")
            else:
                print("[HTTP] Server thread died immediately!")
        except Exception as e:
            print(f"[HTTP] Failed to start server: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()


# ------------------------------------------------------------------
# HTML Template Builder
# ------------------------------------------------------------------

def _build_html(state: ViewerState) -> str:
    """Read viewer.html and substitute dynamic template variables."""
    template_path = Path(__file__).parent / 'viewer.html'
    with open(template_path, 'r') as f:
        template = f.read()

    config = ConfigManager.load()
    ctrl_method = config.controller.method.lower()

    ctrl_labels = {
        'pid': ('PID Control', 'Kp (Proportional Gain)', 'Kd (Derivative Gain)'),
        'pd': ('PD Control', 'Kp (Proportional Gain)', 'Kd (Derivative Gain)'),
        'pure_pursuit': ('Pure Pursuit', 'Gain (Steering)', 'Heading Gain'),
        'mpc': ('MPC Control', 'Q Lateral', 'Q Heading'),
    }
    controller_label, kp_label, kd_label = ctrl_labels.get(
        ctrl_method, ('PID Control', 'Kp (Proportional Gain)', 'Kd (Derivative Gain)'),
    )

    respawn_display = "inline-block" if state.target == "simulation" else "none"
    cv_display = "block" if state.detection_method == "cv" else "none"

    return template.format(
        vehicle_url=state.vehicle_url,
        target=state.target.upper(),
        detection_method=state.detection_method.upper(),
        respawn_display=respawn_display,
        cv_display=cv_display,
        # ROI
        roi_bottom_left_x=state.roi_config['roi_bottom_left_x'],
        roi_top_left_x=state.roi_config['roi_top_left_x'],
        roi_top_right_x=state.roi_config['roi_top_right_x'],
        roi_bottom_right_x=state.roi_config['roi_bottom_right_x'],
        roi_top_y=state.roi_config['roi_top_y'],
        # CV detection
        canny_low=state.canny_low,
        canny_high=state.canny_high,
        hough_threshold=state.hough_threshold,
        hough_min_line_len=state.hough_min_line_len,
        hough_max_line_gap=state.hough_max_line_gap,
        smoothing_factor=state.smoothing_factor,
        # Throttle
        throttle_base=config.throttle_policy.base,
        # Controller
        controller_label=controller_label,
        kp_label=kp_label,
        kd_label=kd_label,
        kp_value=config.controller.kp,
        ki_value=config.controller.ki,
        kd_value=config.controller.kd,
        ki_display="block" if ctrl_method == "pid" else "none",
        lookahead_display="block" if ctrl_method == "pure_pursuit" else "none",
        lookahead_ratio=config.controller.lookahead_ratio,
        camera_offset_x=config.camera.offset_x,
    )
