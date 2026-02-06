# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point picker - finds the gaussian splat at a screen position."""

import numpy as np
from typing import Optional, Tuple

import lichtfeld as lf


def pick_point_at_screen(screen_x: float, screen_y: float) -> Optional[Tuple[float, float, float]]:
    """
    Find the frontmost gaussian splat at the given screen position.
    
    Uses splat scales to compute approximate screen-space radius for each splat,
    then picks the closest-to-camera splat that contains the click point.
    
    Args:
        screen_x: X pixel coordinate
        screen_y: Y pixel coordinate
        
    Returns:
        3D position (x, y, z) of the picked splat, or None if no splat found
    """
    # Get viewport render with screen positions
    vp_render = lf.get_viewport_render()
    if vp_render is None:
        return None
    
    screen_positions = vp_render.screen_positions
    if screen_positions is None:
        return None
    
    # Get scene and view
    scene = lf.get_scene()
    if not scene:
        return None
    
    view = lf.get_current_view()
    if not view:
        return None
    
    combined = scene.combined_model()
    if not combined:
        return None
    
    means = combined.get_means().numpy()  # [N, 3]
    screen_pos = screen_positions.numpy()  # [N, 2]
    
    if len(means) != len(screen_pos):
        return None
    
    # Get splat scales to compute screen-space radius
    scales = combined.get_scaling().numpy()  # [N, 3] - world-space scales
    
    # Camera position for depth calculation
    cam_pos = np.array(view.translation.numpy()).flatten()
    
    # Compute distance from camera to each splat
    to_splats = means - cam_pos
    depths = np.linalg.norm(to_splats, axis=1)
    
    # Compute approximate screen-space radius for each splat
    # Use the max scale dimension as the splat "radius" in world space
    # Then project to screen space: screen_radius ≈ world_radius * focal / depth
    world_radii = np.max(scales, axis=1)  # [N]
    
    # Approximate focal length from FOV and viewport size
    fov_rad = np.radians(view.fov_y)
    focal = view.height / (2.0 * np.tan(fov_rad / 2.0))
    
    # Screen-space radius (with minimum to ensure pickability)
    screen_radii = np.maximum(world_radii * focal / np.maximum(depths, 0.01), 5.0)
    
    # Distance from click to each splat center in screen space
    dx = screen_pos[:, 0] - screen_x
    dy = screen_pos[:, 1] - screen_y
    screen_distances = np.sqrt(dx*dx + dy*dy)
    
    # Find splats where click is within their screen radius
    # Also filter out points behind camera (negative depth conceptually, but we use distance)
    # and points with invalid screen positions
    valid_x = (screen_pos[:, 0] >= 0) & (screen_pos[:, 0] < view.width)
    valid_y = (screen_pos[:, 1] >= 0) & (screen_pos[:, 1] < view.height)
    within_radius = screen_distances < screen_radii
    
    mask = valid_x & valid_y & within_radius
    
    if not np.any(mask):
        # Fallback: use a larger fixed radius
        mask = screen_distances < 30.0
        if not np.any(mask):
            return None
    
    # Among valid splats, pick the one closest to camera (frontmost)
    valid_indices = np.where(mask)[0]
    valid_depths = depths[mask]
    closest_idx = valid_indices[np.argmin(valid_depths)]
    
    return tuple(means[closest_idx])


