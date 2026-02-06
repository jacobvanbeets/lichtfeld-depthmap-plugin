# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Oriented Bounding Box for depth map region selection."""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import math


@dataclass
class OrientedBoundingBox:
    """3D oriented bounding box with position, rotation, and size."""
    
    # Center position
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    
    # Rotation in degrees (Euler angles XYZ)
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    
    # Half-extents (size/2)
    size_x: float = 5.0
    size_y: float = 5.0
    size_z: float = 5.0
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from Euler angles (degrees)."""
        rx = math.radians(self.rotation_x)
        ry = math.radians(self.rotation_y)
        rz = math.radians(self.rotation_z)
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def get_inverse_rotation_matrix(self) -> np.ndarray:
        """Get inverse (transpose) of rotation matrix."""
        return self.get_rotation_matrix().T
    
    def get_center(self) -> np.ndarray:
        """Get center position as array."""
        return np.array([self.center_x, self.center_y, self.center_z])
    
    def get_half_extents(self) -> np.ndarray:
        """Get half-extents (size/2) as array."""
        return np.array([self.size_x, self.size_y, self.size_z])
    
    def contains(self, positions: np.ndarray) -> np.ndarray:
        """Check which positions are inside the oriented bounding box.
        
        Args:
            positions: [N, 3] array of XYZ positions
            
        Returns:
            Boolean mask [N] indicating which points are inside
        """
        # Transform points to box-local coordinates
        # 1. Translate to box center
        centered = positions - self.get_center()
        
        # 2. Rotate by inverse rotation to align with box axes
        R_inv = self.get_inverse_rotation_matrix()
        local_coords = centered @ R_inv.T
        
        # 3. Check if within half-extents
        half_extents = self.get_half_extents()
        inside = (
            (np.abs(local_coords[:, 0]) <= half_extents[0]) &
            (np.abs(local_coords[:, 1]) <= half_extents[1]) &
            (np.abs(local_coords[:, 2]) <= half_extents[2])
        )
        
        return inside
    
    def get_corners(self) -> np.ndarray:
        """Get the 8 corners of the OBB in world coordinates.
        
        Returns:
            [8, 3] array of corner positions
        """
        hx, hy, hz = self.size_x, self.size_y, self.size_z
        
        # Local corners (relative to center)
        local_corners = np.array([
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [+hx, +hy, -hz],
            [-hx, +hy, -hz],
            [-hx, -hy, +hz],
            [+hx, -hy, +hz],
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
        ])
        
        # Rotate and translate to world
        R = self.get_rotation_matrix()
        world_corners = local_corners @ R.T + self.get_center()
        
        return world_corners
    
    def get_edges(self) -> list:
        """Get the 12 edges as pairs of corner indices.
        
        Returns:
            List of (i, j) tuples for edge connections
        """
        return [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
    
    def set_from_aabb(self, min_pt: np.ndarray, max_pt: np.ndarray):
        """Set OBB from axis-aligned bounding box (no rotation)."""
        center = (min_pt + max_pt) / 2
        half_size = (max_pt - min_pt) / 2
        
        self.center_x, self.center_y, self.center_z = center
        self.size_x, self.size_y, self.size_z = half_size
        self.rotation_x = self.rotation_y = self.rotation_z = 0.0
    
    def get_transform_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix (for visualization)."""
        R = self.get_rotation_matrix()
        t = self.get_center()
        
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t
        
        return mat
