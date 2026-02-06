# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point picker tool definition for toolbar integration."""

import os

# Get the plugin path for custom icon loading
_PLUGIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGIN_NAME = "depthmap_viz"


def _poll_has_gaussians(context) -> bool:
    """Check if scene has gaussians to pick from."""
    return (
        getattr(context, "has_scene", False)
        and getattr(context, "num_gaussians", 0) > 0
    )


def _create_tool_def():
    """Create the tool definition (deferred to avoid import issues)."""
    from lfs_plugins.tool_defs.definition import ToolDef
    
    return ToolDef(
        id="depthmap.point_picker",
        label="Depth Point Picker",
        icon="point-picker",
        group="utility",
        order=80,
        description="Pick points on the model for depth range",
        shortcut="",
        operator="depthmap_viz.operators.point_picker.PointPickerOperator",
        poll=_poll_has_gaussians,
        plugin_name=_PLUGIN_NAME,
        plugin_path=_PLUGIN_PATH,
    )


def register_picker_tool():
    """Register the point picker tool with the toolbar."""
    import lichtfeld as lf
    from lfs_plugins.tools import ToolRegistry
    
    tool = _create_tool_def()
    ToolRegistry.register_tool(tool)
    lf.log.info("Registered point picker tool")


def unregister_picker_tool():
    """Unregister the point picker tool."""
    import lichtfeld as lf
    from lfs_plugins.tools import ToolRegistry
    
    ToolRegistry.unregister_tool("depthmap.point_picker")
    lf.log.info("Unregistered point picker tool")
