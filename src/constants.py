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

    # Only the RAW_RGB HAS SET AS TRUE, it would be needed
    # to adjust the quality for websocket overloads.

    # Otherwise, the jpeg quality would apply in advance from the lkas module.
    # So, in the viewer, it doesn't be needed to adjust it again.


class Targets:
    """Target connection presets for simulation and vehicle."""

    # Target configurations: host addresses for different deployment scenarios
    SIMULATION = {
        'broadcast_host': 'jetracer.local'
    }

    VEHICLE = {
        'broadcast_host': 'jetracer.local'  # Will be Jetson's IP in production
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
    'Targets',
    'get_target_config',
]
