"""
Viewer Module Constants

Module-specific constants for the web viewer.
Configurable values that control viewer behavior, performance, and connectivity.
"""

from typing import Dict, Any


class FPS:
    """Frame rate and polling frequency constants."""

    WEBSOCKET_MAX = 60          # Max FPS for WebSocket frame broadcast
    STATUS_BROADCAST = 2        # Status updates per second (Hz)
    ZMQ_POLL = 1000            # ZMQ polling rate (Hz)


class Streaming:
    """Video streaming quality constants."""

    JPEG_QUALITY = 75          # JPEG compression quality for WebSocket (0-100, lower = faster)


class ROI:
    """Region of Interest overlay configuration.

    These values control the green trapezoid overlay shown on the viewer.
    They should match the values used by the detection module.
    """

    BOTTOM_LEFT_X = 0.05       # Fraction of width (bottom-left corner)
    TOP_LEFT_X = 0.35          # Fraction of width (top-left corner)
    TOP_RIGHT_X = 0.65         # Fraction of width (top-right corner)
    BOTTOM_RIGHT_X = 0.95      # Fraction of width (bottom-right corner)
    TOP_Y = 0.5                # Fraction of height (look at top 50% of image)


class Targets:
    """Target connection presets for simulation and vehicle."""

    # Target configurations: host addresses for different deployment scenarios
    SIMULATION = {
        'broadcast_host': 'localhost'
    }

    VEHICLE = {
        'broadcast_host': 'localhost'  # Will be Jetson's IP in production
    }


# Convenience function to get target config
def get_target_config(target: str) -> Dict[str, Any]:
    """
    Get configuration for specified target.

    Args:
        target: Target name ('simulation' or 'vehicle')

    Returns:
        Dictionary with target configuration
    """
    if target == 'simulation':
        return Targets.SIMULATION.copy()
    elif target == 'vehicle':
        return Targets.VEHICLE.copy()
    else:
        return Targets.SIMULATION.copy()


# Export all constants
__all__ = [
    'FPS',
    'Streaming',
    'ROI',
    'Targets',
    'get_target_config',
]
