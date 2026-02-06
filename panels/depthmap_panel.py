# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth Map Visualization Panel with live preview."""

from typing import Optional
import numpy as np
import lichtfeld as lf
import lichtfeld.selection as sel
from lfs_plugins.types import Panel

from ..core.depthmap import apply_depthmap_colors


class DepthmapPanel(Panel):
    """Panel for depth map visualization with live preview."""
    
    label = "Depth Map"
    space = "MAIN_PANEL_TAB"
    order = 25
    
    COLORMAP_ITEMS = [
        ("jet", "Jet (Rainbow)"),
        ("grayscale", "Grayscale"),
        ("turbo", "Turbo"),
        ("viridis", "Viridis"),
    ]
    
    AXIS_ITEMS = [
        ("z", "Z-Axis (Height)"),
        ("y", "Y-Axis"),
        ("x", "X-Axis"),
        ("camera", "Camera Distance"),
    ]
    
    def __init__(self):
        # Enable toggle for non-destructive preview
        self._enabled = False
        self._live_preview = True  # Always on by default
        self._depth_map_active = False
        
        # Colormap settings
        self._colormap_idx = 0
        self._axis_idx = 0
        self._invert = False
        
        # Depth range
        self._use_custom_range = False
        self._min_depth = 0.0
        self._max_depth = 10.0
        self._range_only = False  # False = flood all, True = only within range
        
        # Point markers for depth range
        self._point1_pos = None  # (x, y, z) world position
        self._point2_pos = None
        self._captured_pos = None
        self._captured_depth = None
        self._draw_handler_id = "depthmap_markers"
        
        # Saved colors
        self._saved_colors: dict = {}  # SH0
        self._saved_shN: dict = {}  # Higher-order SH (for grayscale restore)
        self._current_target: Optional[str] = None
        
        # Status
        self._status_msg = ""
        self._status_is_error = False
        
        # Register marker draw handler
        self._register_marker_handler()
    
    def _register_marker_handler(self):
        """Register viewport draw handler for point markers."""
        panel = self  # Capture reference
        
        def draw_markers(ctx):
            # Draw Point 1 marker (green sphere)
            if panel._point1_pos is not None:
                ctx.draw_point_3d(panel._point1_pos, (0.0, 1.0, 0.0, 1.0), 20.0)
                screen = ctx.world_to_screen(panel._point1_pos)
                if screen:
                    ctx.draw_text_2d((screen[0] + 15, screen[1] - 8), f"P1 Min: {panel._min_depth:.1f}", (0.0, 1.0, 0.0, 1.0))
                    ctx.draw_circle_2d(screen, 15.0, (0.0, 1.0, 0.0, 1.0), 2.0)
            
            # Draw Point 2 marker (orange sphere)
            if panel._point2_pos is not None:
                ctx.draw_point_3d(panel._point2_pos, (1.0, 0.5, 0.0, 1.0), 20.0)
                screen = ctx.world_to_screen(panel._point2_pos)
                if screen:
                    ctx.draw_text_2d((screen[0] + 15, screen[1] - 8), f"P2 Max: {panel._max_depth:.1f}", (1.0, 0.5, 0.0, 1.0))
                    ctx.draw_circle_2d(screen, 15.0, (1.0, 0.5, 0.0, 1.0), 2.0)
            
            # Draw captured position marker (cyan)
            if panel._captured_pos is not None:
                ctx.draw_point_3d(panel._captured_pos, (0.0, 1.0, 1.0, 1.0), 18.0)
                screen = ctx.world_to_screen(panel._captured_pos)
                if screen:
                    ctx.draw_text_2d((screen[0] + 15, screen[1]), f"Captured: {panel._captured_depth:.2f}", (0.0, 1.0, 1.0, 1.0))
                    ctx.draw_circle_2d(screen, 12.0, (0.0, 1.0, 1.0, 1.0), 2.0)
        
        try:
            lf.remove_draw_handler(self._draw_handler_id)
        except:
            pass
        lf.add_draw_handler(self._draw_handler_id, draw_markers, "POST_VIEW")
    
    def _capture_from_selection(self) -> bool:
        """Capture position from selected gaussians."""
        try:
            scene = lf.get_scene()
            if not scene:
                self._status_msg = "No scene"
                return False
            
            # Check if there's a gaussian selection
            if not scene.has_selection():
                self._status_msg = "No gaussian selection (use S key + click)"
                return False
            
            mask = scene.selection_mask
            if mask is None:
                self._status_msg = "Selection mask is None"
                return False
            
            mask_np = mask.numpy().astype(bool)
            if not np.any(mask_np):
                self._status_msg = "Selection mask is empty"
                return False
            
            # Use combined_model which matches the selection mask
            combined = scene.combined_model()
            if not combined:
                self._status_msg = "No combined model"
                return False
            
            means = combined.get_means().numpy()
            
            # Make sure mask and means have compatible sizes
            min_len = min(len(mask_np), len(means))
            selected_positions = means[:min_len][mask_np[:min_len]]
            
            if len(selected_positions) == 0:
                self._status_msg = "No points in selection"
                return False
            
            # Use the center of selected points
            center = np.mean(selected_positions, axis=0)
            self._captured_pos = tuple(center)
            self._captured_depth = self._get_depth_from_position(center)
            self._status_msg = f"Captured from {len(selected_positions)} points: {self._captured_depth:.2f}"
            self._status_is_error = False
            return True
        except Exception as e:
            self._status_msg = f"Error: {e}"
            self._status_is_error = True
        return False
    
    def _set_point_from_captured(self, point_num: int) -> bool:
        """Set point 1 or 2 from captured position."""
        if self._captured_pos is None:
            self._status_msg = "Click on the model first to capture a point!"
            self._status_is_error = True
            return False
        
        if point_num == 1:
            self._point1_pos = self._captured_pos
            self._min_depth = self._captured_depth
            self._status_msg = f"Point 1 set: {self._min_depth:.2f}"
        else:
            self._point2_pos = self._captured_pos
            self._max_depth = self._captured_depth
            self._status_msg = f"Point 2 set: {self._max_depth:.2f}"
        
        self._use_custom_range = True
        self._status_is_error = False
        lf.ui.request_redraw()
        return True
    
    def _get_depth_from_position(self, pos) -> float:
        """Get depth value from a 3D position based on current axis setting."""
        axis = self.AXIS_ITEMS[self._axis_idx][0]
        if axis == "x":
            return pos[0]
        elif axis == "y":
            return pos[1]
        elif axis == "z":
            return pos[2]
        elif axis == "camera":
            view = lf.get_current_view()
            if view:
                cam_pos = np.array(view.translation.numpy()).flatten()
                return float(np.linalg.norm(np.array(pos) - cam_pos))
        return pos[2]
    
    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()
    
    def _get_selected_splat_name(self) -> Optional[str]:
        """Get the selected splat name, or auto-select first available splat."""
        scene = lf.get_scene()
        if not scene:
            return None
        
        # First check if a splat is already selected
        selected = lf.get_selected_node_names()
        for name in selected:
            node = scene.get_node(name)
            if node and (node.splat_data() is not None):
                return name
        
        # Auto-select first available splat
        for node in scene.get_nodes():
            if node.splat_data() is not None:
                return node.name
        
        return None
    
    def _save_original_colors(self, node_name: str, force: bool = False) -> bool:
        """Save original colors for a node. If force=True, overwrite existing save."""
        if not force and node_name in self._saved_colors:
            return True
        
        scene = lf.get_scene()
        if not scene:
            return False
        
        node = scene.get_node(node_name)
        if not node:
            return False
        
        splat = node.splat_data()
        if splat:
            # Clone to CPU to ensure we have a stable copy
            self._saved_colors[node_name] = splat.sh0_raw.clone().cpu()
            # Also save higher-order SH for grayscale restore
            shN = splat.shN_raw
            if shN is not None and shN.shape[0] > 0:
                self._saved_shN[node_name] = shN.clone().cpu()
            self._current_target = node_name
            return True
        return False
    
    def _restore_original_colors(self, silent: bool = False):
        """Restore original colors for the current target node."""
        # Use current target if set, otherwise try selected
        node_name = self._current_target or self._get_selected_splat_name()
        
        if not node_name or node_name not in self._saved_colors:
            if not silent:
                self._status_msg = "No saved colors to restore"
                self._status_is_error = True
            return False
        
        scene = lf.get_scene()
        if not scene:
            return False
        
        node = scene.get_node(node_name)
        if not node:
            return False
        
        splat = node.splat_data()
        if splat:
            try:
                # Restore SH0
                saved = self._saved_colors[node_name].cuda()
                sh0_tensor = splat.sh0_raw
                sh0_tensor[:] = saved
                
                # Restore higher-order SH if saved (for grayscale)
                if node_name in self._saved_shN:
                    saved_shN = self._saved_shN[node_name].cuda()
                    shN_tensor = splat.shN_raw
                    if shN_tensor is not None:
                        shN_tensor[:] = saved_shN
                
                scene.notify_changed()
                lf.ui.request_redraw()
                self._depth_map_active = False
                if not silent:
                    self._status_msg = "Restored original colors"
                    self._status_is_error = False
                return True
            except Exception as e:
                if not silent:
                    self._status_msg = f"Restore failed: {e}"
                    self._status_is_error = True
        return False
    
    def _apply_depthmap(self, silent: bool = False):
        node_name = self._get_selected_splat_name()
        if not node_name:
            if not silent:
                self._status_msg = "No splat selected"
                self._status_is_error = True
            return
        
        # Save original colors
        self._save_original_colors(node_name)
        self._current_target = node_name
        
        colormap = self.COLORMAP_ITEMS[self._colormap_idx][0]
        axis = self.AXIS_ITEMS[self._axis_idx][0]
        
        min_depth = self._min_depth if self._use_custom_range else None
        max_depth = self._max_depth if self._use_custom_range else None
        range_only = self._range_only if self._use_custom_range else False
        
        # Get saved original colors for range_only mode
        original_sh0 = None
        if range_only and node_name in self._saved_colors:
            original_sh0 = self._saved_colors[node_name].numpy()
        
        success, msg = apply_depthmap_colors(
            node_name=node_name,
            colormap=colormap,
            axis=axis,
            min_depth=min_depth,
            max_depth=max_depth,
            range_only=range_only,
            invert=self._invert,
            original_sh0=original_sh0,
        )
        
        self._depth_map_active = success
        if not silent:
            self._status_msg = msg
            self._status_is_error = not success
    
    def draw(self, layout):
        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()
        
        node_name = self._get_selected_splat_name()
        if not node_name:
            layout.text_colored("Select a splat or point cloud", theme.palette.text_dim)
            return
        
        # Header with target
        layout.label(f"Target: {node_name}")
        
        # Enable toggle - controls whether depth map is active
        changed, self._enabled = layout.checkbox("Enable Depth Map", self._enabled)
        if changed:
            if self._enabled:
                # Save colors and apply
                self._save_original_colors(node_name, force=True)
                self._apply_depthmap(silent=True)
            else:
                # Restore original colors
                self._restore_original_colors(silent=True)
        
        if self._enabled:
            _, self._live_preview = layout.checkbox("Live Preview", self._live_preview)
            if layout.is_item_hovered():
                layout.set_tooltip("Automatically update depth map when settings change")
        
        layout.separator()
        
        # Track if any setting changed
        settings_changed = False
        
        # === Colormap ===
        if layout.collapsing_header("Colormap", default_open=True):
            colormap_labels = [item[1] for item in self.COLORMAP_ITEMS]
            changed, self._colormap_idx = layout.combo("##cmap", self._colormap_idx, colormap_labels)
            settings_changed |= changed
            
            changed, self._invert = layout.checkbox("Invert##cmap", self._invert)
            settings_changed |= changed
        
        # === Depth Axis ===
        if layout.collapsing_header("Depth Axis", default_open=True):
            axis_labels = [item[1] for item in self.AXIS_ITEMS]
            changed, self._axis_idx = layout.combo("##axis", self._axis_idx, axis_labels)
            settings_changed |= changed
        
        # === Depth Range ===
        if layout.collapsing_header("Depth Range", default_open=True):
            
            # Instructions
            layout.text_colored("1. Switch to Select mode (toolbar)", theme.palette.text_dim)
            layout.text_colored("2. Click on model to select points", theme.palette.text_dim)
            layout.text_colored("3. Click Set Point button below", theme.palette.text_dim)
            layout.spacing()
            
            # Point 1 section
            if self._point1_pos:
                layout.text_colored(f"Point 1: {self._min_depth:.2f}", (0.0, 1.0, 0.0, 1.0))
            else:
                layout.label("Point 1: Not set")
            
            if layout.button("Set Point 1 from Selection##setp1", (-1, 32 * scale)):
                if self._capture_from_selection():
                    self._point1_pos = self._captured_pos
                    self._min_depth = self._captured_depth
                    self._use_custom_range = True
                    self._status_msg = f"Point 1 set: {self._min_depth:.2f}"
                    self._status_is_error = False
                    settings_changed = True
                    # Save colors if not enabled, then apply
                    if not self._enabled:
                        self._save_original_colors(node_name, force=True)
                        self._enabled = True
                    self._apply_depthmap(silent=False)
                else:
                    self._status_is_error = True
            
            layout.spacing()
            
            # Point 2 section
            if self._point2_pos:
                layout.text_colored(f"Point 2: {self._max_depth:.2f}", (1.0, 0.5, 0.0, 1.0))
            else:
                layout.label("Point 2: Not set")
            
            if layout.button("Set Point 2 from Selection##setp2", (-1, 32 * scale)):
                if self._capture_from_selection():
                    self._point2_pos = self._captured_pos
                    self._max_depth = self._captured_depth
                    self._use_custom_range = True
                    self._status_msg = f"Point 2 set: {self._max_depth:.2f}"
                    self._status_is_error = False
                    settings_changed = True
                    # Save colors if not enabled, then apply
                    if not self._enabled:
                        self._save_original_colors(node_name, force=True)
                        self._enabled = True
                    self._apply_depthmap(silent=False)
                else:
                    self._status_is_error = True
            
            layout.spacing()
            
            # Clear points button
            if self._point1_pos or self._point2_pos:
                if layout.button("Clear Points##clearpts", (-1, 0)):
                    self._point1_pos = None
                    self._point2_pos = None
                    self._use_custom_range = False
                    self._status_msg = "Points cleared"
                    self._status_is_error = False
                    settings_changed = True
                    lf.ui.request_redraw()
            
            layout.separator()
            
            # Manual adjustment
            if self._use_custom_range:
                layout.label("Fine-tune:")
                btn_w = 32 * scale
                btn_w_small = 38 * scale
                
                # Min row
                layout.push_item_width(80 * scale)
                changed, self._min_depth = layout.input_float("Min##minval", self._min_depth, 0.0, 0.0, "%.2f")
                if changed:
                    settings_changed = True
                layout.pop_item_width()
                layout.same_line()
                if layout.button("-##minsub", (btn_w, 0)):
                    self._min_depth -= 1.0
                    settings_changed = True
                layout.same_line()
                if layout.button("+##minadd", (btn_w, 0)):
                    self._min_depth += 1.0
                    settings_changed = True
                layout.same_line()
                if layout.button("-.1##minsub01", (btn_w_small, 0)):
                    self._min_depth -= 0.1
                    settings_changed = True
                layout.same_line()
                if layout.button("+.1##minadd01", (btn_w_small, 0)):
                    self._min_depth += 0.1
                    settings_changed = True
                
                # Max row
                layout.push_item_width(80 * scale)
                changed, self._max_depth = layout.input_float("Max##maxval", self._max_depth, 0.0, 0.0, "%.2f")
                if changed:
                    settings_changed = True
                layout.pop_item_width()
                layout.same_line()
                if layout.button("-##maxsub", (btn_w, 0)):
                    self._max_depth -= 1.0
                    settings_changed = True
                layout.same_line()
                if layout.button("+##maxadd", (btn_w, 0)):
                    self._max_depth += 1.0
                    settings_changed = True
                layout.same_line()
                if layout.button("-.1##maxsub01", (btn_w_small, 0)):
                    self._max_depth -= 0.1
                    settings_changed = True
                layout.same_line()
                if layout.button("+.1##maxadd01", (btn_w_small, 0)):
                    self._max_depth += 0.1
                    settings_changed = True
                
                # Swap if needed
                if self._min_depth > self._max_depth:
                    layout.text_colored("Min > Max!", (1.0, 0.5, 0.0, 1.0))
                    layout.same_line()
                    if layout.button("Swap##swaprange", (60 * scale, 0)):
                        self._min_depth, self._max_depth = self._max_depth, self._min_depth
                        settings_changed = True
                
                layout.spacing()
                changed, self._range_only = layout.checkbox("Only Color Within Range##rangeonly", self._range_only)
                settings_changed |= changed
                if layout.is_item_hovered():
                    layout.set_tooltip("Checked: Only color within range (keep original outside)\nUnchecked: Flood all splats with depth colors")
            else:
                layout.text_colored("Range: Auto (from scene)", theme.palette.text_dim)
        
        # === Live preview update ===
        if self._enabled and self._live_preview and settings_changed:
            self._apply_depthmap(silent=True)
        
        # === Apply / Restore buttons ===
        layout.separator()
        
        if not self._enabled:
            if layout.button("Apply Depth Map", (-1, 28 * scale)):
                self._save_original_colors(node_name, force=True)
                self._apply_depthmap()
                self._enabled = True
        else:
            if layout.button("Update Depth Map", (-1, 28 * scale)):
                self._apply_depthmap()
            
            if layout.button("Restore Original", (-1, 0)):
                self._restore_original_colors()
                self._enabled = False
        
        # Status
        if self._status_msg:
            layout.spacing()
            color = (1.0, 0.4, 0.4, 1.0) if self._status_is_error else (0.4, 1.0, 0.4, 1.0)
            layout.text_colored(self._status_msg, color)
        
        # === Quick presets ===
        layout.separator()
        if layout.collapsing_header("Presets", default_open=False):
            avail_w, _ = layout.get_content_region_avail()
            half = avail_w * 0.5 - 2
            
            def apply_preset(cmap_idx, axis_idx):
                self._colormap_idx = cmap_idx
                self._axis_idx = axis_idx
                if not self._enabled:
                    self._save_original_colors(node_name, force=True)
                    self._enabled = True
                self._apply_depthmap()
            
            if layout.button("Jet Z", (half, 0)):
                apply_preset(0, 0)
            layout.same_line()
            if layout.button("Gray Z", (half, 0)):
                apply_preset(1, 0)
            
            if layout.button("Turbo Z", (half, 0)):
                apply_preset(2, 0)
            layout.same_line()
            if layout.button("Camera", (half, 0)):
                apply_preset(0, 3)
