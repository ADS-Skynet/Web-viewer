"""
Message Handlers

Shared logic for processing WebSocket and HTTP messages.
Eliminates duplication between WS handler and HTTP POST handler.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run import ViewerState
    from common.communication import ActionPublisher, ParameterPublisher


class MessageHandler:
    """
    Processes actions, parameter updates, and toggle commands.

    Used by both WebSocketServer and ViewerHTTPServer so the logic
    lives in exactly one place.
    """

    def __init__(
        self,
        state: ViewerState,
        parameter_publisher: ParameterPublisher,
        action_publisher: ActionPublisher,
        verbose: bool = False,
    ):
        self.state = state
        self.parameter_publisher = parameter_publisher
        self.action_publisher = action_publisher
        self.verbose = verbose

    def handle_action(self, action: str):
        """Send an action command (pause, resume, respawn) to vehicle."""
        self.action_publisher.send_action(action)
        if self.verbose:
            print(f"[Handler] Action: {action}")

    def handle_parameter(self, category: str, parameter: str, value: float) -> dict:
        """
        Apply a parameter update locally and forward via ZMQ.

        Returns a response dict suitable for JSON serialization.
        """
        if parameter == 'camera_offset_x':
            self.state.camera_offset_x = int(value)

        # Forward to remote servers
        self.parameter_publisher.send_parameter(category, parameter, value)

        if self.verbose:
            print(f"[Handler] Parameter: {category}.{parameter} = {value}")

        return {
            'status': 'ok',
            'category': category,
            'parameter': parameter,
            'value': value,
        }

    def handle_toggle(self, setting: str, enabled: bool):
        """Toggle a visualization layer on/off."""
        toggle_map = {
            'raw_image': 'show_raw_image',
            'lanes': 'show_lanes',
            'hud': 'show_hud',
            'segmentation': 'show_segmentation',
        }

        attr = toggle_map.get(setting)
        if attr is None:
            return

        with self.state.display_lock:
            setattr(self.state, attr, enabled)

        if self.verbose:
            label = setting.replace('_', ' ').title()
            print(f"[Handler] {label}: {'ON' if enabled else 'OFF'}")
