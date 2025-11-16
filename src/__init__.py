"""
Viewer Module

Remote web-based viewer for monitoring vehicle/simulation.
Runs on laptop, receives data via ZMQ, draws overlays, serves web interface.
"""

from .run import ZMQWebViewer

__all__ = ['ZMQWebViewer']
