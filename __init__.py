# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth Map Visualization Plugin for LichtFeld Studio.

Provides Jet colormap and Grayscale depth visualization for Gaussian Splats.
"""

import lichtfeld as lf

from .panels.depthmap_panel import DepthmapPanel
from .core.colormaps import jet_colormap, grayscale_colormap
from .core.depthmap import apply_depthmap_colors

_classes = [DepthmapPanel]


def on_load():
    """Called when plugin loads."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("Depth Map Visualization plugin loaded")


def on_unload():
    """Called when plugin unloads."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("Depth Map Visualization plugin unloaded")


__all__ = [
    "DepthmapPanel",
    "jet_colormap",
    "grayscale_colormap",
    "apply_depthmap_colors",
]
