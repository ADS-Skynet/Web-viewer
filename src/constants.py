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

    # Quality policy based on source format:
    # - raw_rgb=true:  Viewer encodes with this quality (controls WebSocket bandwidth)
    # - raw_rgb=false: Viewer uses quality 100 (minimize re-encoding loss)
    #                  Original quality is already applied by LKAS from config.yaml


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
