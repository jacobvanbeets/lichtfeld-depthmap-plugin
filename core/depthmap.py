# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth map computation and color application for Gaussian Splats."""

import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass

import lichtfeld as lf

from .colormaps import get_colormap, jet_colormap, grayscale_colormap


@dataclass
class BoundingBox:
    """3D axis-aligned bounding box for region-of-interest selection."""
    min_x: float = -float('inf')
    max_x: float = float('inf')
    min_y: float = -float('inf')
    max_y: float = float('inf')
    min_z: float = -float('inf')
    max_z: float = float('inf')
    
    def contains(self, positions: np.ndarray) -> np.ndarray:
        """Check which positions are inside the bounding box.
        
        Args:
            positions: [N, 3] array of XYZ positions
            
        Returns:
            Boolean mask [N] indicating which points are inside
        """
        inside = (
            (positions[:, 0] >= self.min_x) & (positions[:, 0] <= self.max_x) &
            (positions[:, 1] >= self.min_y) & (positions[:, 1] <= self.max_y) &
            (positions[:, 2] >= self.min_z) & (positions[:, 2] <= self.max_z)
        )
        return inside
    
    def is_valid(self) -> bool:
        """Check if bounding box has valid finite bounds."""
        return (
            self.min_x < self.max_x and
            self.min_y < self.max_y and
            self.min_z < self.max_z and
            np.isfinite(self.min_x) and np.isfinite(self.max_x) and
            np.isfinite(self.min_y) and np.isfinite(self.max_y) and
            np.isfinite(self.min_z) and np.isfinite(self.max_z)
        )


def compute_depth_values(
    positions: np.ndarray,
    axis: Literal["x", "y", "z", "camera"] = "z",
    camera_pos: Optional[np.ndarray] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    bbox: Optional[BoundingBox] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute normalized depth values from positions.
    
    Args:
        positions: [N, 3] array of XYZ positions
        axis: Which axis to use for depth ('x', 'y', 'z', or 'camera')
        camera_pos: Camera position [3] for camera-relative depth
        min_depth: Manual minimum depth (auto-computed if None)
        max_depth: Manual maximum depth (auto-computed if None)
        bbox: Optional bounding box to filter positions
        
    Returns:
        Tuple of:
        - normalized_depths: [N] normalized depth values in [0, 1]
        - mask: [N] boolean mask of valid points (inside bbox if specified)
        - actual_min: The actual minimum depth used
        - actual_max: The actual maximum depth used
    """
    positions = np.asarray(positions)
    n_points = positions.shape[0]
    
    # Create mask for points inside bounding box
    if bbox is not None and bbox.is_valid():
        mask = bbox.contains(positions)
    else:
        mask = np.ones(n_points, dtype=bool)
    
    # Extract depth based on axis
    if axis == "x":
        depths = positions[:, 0]
    elif axis == "y":
        depths = positions[:, 1]
    elif axis == "z":
        depths = positions[:, 2]
    elif axis == "camera":
        if camera_pos is None:
            # Default to origin if no camera position
            camera_pos = np.array([0.0, 0.0, 0.0])
        # Euclidean distance from camera
        depths = np.linalg.norm(positions - camera_pos, axis=1)
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'x', 'y', 'z', or 'camera'")
    
    # Compute min/max from masked points if not provided
    masked_depths = depths[mask] if mask.any() else depths
    
    actual_min = min_depth if min_depth is not None else float(np.min(masked_depths))
    actual_max = max_depth if max_depth is not None else float(np.max(masked_depths))
    
    # Ensure min <= max (user may pick points in any order, including with negative values)
    if actual_min > actual_max:
        actual_min, actual_max = actual_max, actual_min
    
    # Avoid division by zero
    depth_range = actual_max - actual_min
    if depth_range < 1e-6:
        depth_range = 1.0
    
    # Normalize depths
    normalized = (depths - actual_min) / depth_range
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized, mask, actual_min, actual_max


def apply_depthmap_colors(
    node_name: str,
    colormap: str = "jet",
    axis: Literal["x", "y", "z", "camera"] = "z",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    range_only: bool = False,
    invert: bool = False,
    original_sh0: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    """Apply depth-based colors to a splat node.
    
    Args:
        node_name: Name of the splat node to colorize
        colormap: Colormap to use ('jet', 'grayscale', 'turbo', 'viridis')
        axis: Which axis/method for depth ('x', 'y', 'z', 'camera')
        min_depth: Manual minimum depth (auto if None)
        max_depth: Manual maximum depth (auto if None)
        range_only: If True, only colorize points within min/max range
        invert: Invert the depth (far=low value, near=high value)
        original_sh0: Saved original SH0 colors to use as base when range_only is enabled
        
    Returns:
        Tuple of (success, message)
    """
    scene = lf.get_scene()
    if scene is None:
        return False, "No scene loaded"
    
    node = scene.get_node(node_name)
    if node is None:
        return False, f"Node '{node_name}' not found"
    
    # Get splat data
    splat = node.splat_data()
    if splat is None:
        # Try point cloud
        pc = node.point_cloud()
        if pc is None:
            return False, f"Node '{node_name}' is not a splat or point cloud"
        
        # Point cloud path
        positions = pc.means.numpy()
        n_points = positions.shape[0]
        
        # Get camera position if needed
        camera_pos = None
        if axis == "camera":
            view = lf.get_current_view()
            if view is not None:
                camera_pos = np.array(view.translation.numpy()).flatten()
        
        # Compute depths
        normalized, mask, d_min, d_max = compute_depth_values(
            positions, axis, camera_pos, min_depth, max_depth
        )
        
        if invert:
            normalized = 1.0 - normalized
        
        # Apply colormap
        cmap_fn = get_colormap(colormap)
        if colormap == "grayscale":
            colors = grayscale_colormap(normalized, invert=False)  # already inverted above
        else:
            colors = cmap_fn(normalized)
        
        # Update point cloud colors
        colors_tensor = lf.Tensor.from_numpy(colors.astype(np.float32))
        positions_tensor = lf.Tensor.from_numpy(positions.astype(np.float32))
        pc.set_data(positions_tensor, colors_tensor)
        
        return True, f"Applied {colormap} depth map (depth range: {d_min:.2f} - {d_max:.2f})"
    
    # Splat path
    # Use combined_model positions for depth calculation - this matches the coordinate
    # space used by pick_at_screen (world space). Colors are still applied to splat.sh0_raw.
    combined = scene.combined_model()
    if combined is not None:
        positions = combined.get_means().numpy()
    else:
        positions = splat.get_means().numpy()
    n_points = positions.shape[0]
    
    # Verify we have the same number of points as the splat's SH0 tensor
    sh0_count = splat.sh0_raw.shape[0]
    if n_points != sh0_count:
        # Fall back to splat positions if counts don't match
        positions = splat.get_means().numpy()
        n_points = positions.shape[0]
    
    # Get camera position if needed
    camera_pos = None
    if axis == "camera":
        view = lf.get_current_view()
        if view is not None:
            camera_pos = np.array(view.translation.numpy()).flatten()
    
    # Compute depths
    normalized, mask, d_min, d_max = compute_depth_values(
        positions, axis, camera_pos, min_depth, max_depth
    )
    
    if invert:
        normalized = 1.0 - normalized
    
    # Apply colormap
    cmap_fn = get_colormap(colormap)
    if colormap == "grayscale":
        colors = grayscale_colormap(normalized, invert=False)
    else:
        colors = cmap_fn(normalized)
    
    # Convert colors to SH0 format
    C0 = 0.28209479177387814
    
    # Handle range-only coloring - only modify points within depth range
    if range_only and min_depth is not None and max_depth is not None:
        # Get raw depths to check which are in range
        if axis == "x":
            depths = positions[:, 0]
        elif axis == "y":
            depths = positions[:, 1]
        elif axis == "z":
            depths = positions[:, 2]
        elif axis == "camera":
            if camera_pos is None:
                camera_pos = np.array([0.0, 0.0, 0.0])
            depths = np.linalg.norm(positions - camera_pos, axis=1)
        
        # Handle either order (min can be > max if user selected in reverse)
        range_lo = min(min_depth, max_depth)
        range_hi = max(min_depth, max_depth)
        
        # Find points within the depth range
        in_range = (depths >= range_lo) & (depths <= range_hi)
        
        # Use saved original colors as base, or current if not provided
        sh0_tensor = splat.sh0_raw
        if original_sh0 is not None:
            base_sh0 = original_sh0.copy()
        else:
            base_sh0 = sh0_tensor.numpy().copy()
        
        # Convert new colors to SH0 format
        new_sh0_colors = (colors - 0.5) / C0
        new_sh0_colors = new_sh0_colors.reshape(-1, 1, 3).astype(np.float32)
        
        # Only update in-range points, keep original for out-of-range
        base_sh0[in_range] = new_sh0_colors[in_range]
        
        # Write back
        new_sh0 = lf.Tensor.from_numpy(base_sh0).cuda()
        sh0_tensor[:] = new_sh0
    else:
        # Apply to all points
        in_range = None  # All points affected
        sh0_colors = (colors - 0.5) / C0
        sh0_colors = sh0_colors.reshape(-1, 1, 3).astype(np.float32)
        
        new_sh0 = lf.Tensor.from_numpy(sh0_colors).cuda()
        sh0_tensor = splat.sh0_raw
        sh0_tensor[:] = new_sh0
    
    # For grayscale mode, zero out higher-order SH to remove view-dependent color
    if colormap == "grayscale":
        shN_tensor = splat.shN_raw
        if shN_tensor is not None and shN_tensor.shape[0] > 0:
            shN_np = shN_tensor.numpy().copy()
            if in_range is not None:
                # Only zero out in-range points
                shN_np[in_range] = 0.0
            else:
                # Zero out all
                shN_np[:] = 0.0
            new_shN = lf.Tensor.from_numpy(shN_np.astype(np.float32)).cuda()
            shN_tensor[:] = new_shN
    
    # Force a scene update
    scene = lf.get_scene()
    if scene:
        scene.notify_changed()
    lf.ui.request_redraw()
    
    return True, f"Applied {colormap} depth map (depth range: {d_min:.2f} - {d_max:.2f})"


def get_scene_bounds(node_name: Optional[str] = None) -> Optional[BoundingBox]:
    """Get the bounding box of a node or the entire scene.
    
    Args:
        node_name: Optional node name. If None, returns scene bounds.
        
    Returns:
        BoundingBox or None if no data available
    """
    scene = lf.get_scene()
    if scene is None:
        return None
    
    all_positions = []
    
    if node_name:
        node = scene.get_node(node_name)
        if node is None:
            return None
        nodes = [node]
    else:
        nodes = list(scene.get_nodes())
    
    for node in nodes:
        splat = node.splat_data()
        if splat is not None:
            positions = splat.get_means().numpy()
            all_positions.append(positions)
            continue
        
        pc = node.point_cloud()
        if pc is not None:
            positions = pc.means.numpy()
            all_positions.append(positions)
    
    if not all_positions:
        return None
    
    positions = np.concatenate(all_positions, axis=0)
    
    return BoundingBox(
        min_x=float(np.min(positions[:, 0])),
        max_x=float(np.max(positions[:, 0])),
        min_y=float(np.min(positions[:, 1])),
        max_y=float(np.max(positions[:, 1])),
        min_z=float(np.min(positions[:, 2])),
        max_z=float(np.max(positions[:, 2])),
    )
