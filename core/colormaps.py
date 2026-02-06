# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Colormap functions for depth visualization."""

import numpy as np
from typing import Tuple


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """Apply jet colormap to normalized values [0, 1].
    
    The jet colormap transitions: blue -> cyan -> green -> yellow -> red
    
    Args:
        values: Normalized depth values in range [0, 1], shape [N] or [N, 1]
        
    Returns:
        RGB colors in range [0, 1], shape [N, 3]
    """
    values = np.asarray(values).flatten()
    values = np.clip(values, 0.0, 1.0)
    
    # Jet colormap implementation
    # 4 segments: blue->cyan, cyan->green, green->yellow, yellow->red
    r = np.zeros_like(values)
    g = np.zeros_like(values)
    b = np.zeros_like(values)
    
    # Segment 1: 0.0 - 0.25 (blue to cyan)
    mask = values <= 0.25
    r[mask] = 0.0
    g[mask] = 4.0 * values[mask]
    b[mask] = 1.0
    
    # Segment 2: 0.25 - 0.5 (cyan to green)
    mask = (values > 0.25) & (values <= 0.5)
    r[mask] = 0.0
    g[mask] = 1.0
    b[mask] = 1.0 - 4.0 * (values[mask] - 0.25)
    
    # Segment 3: 0.5 - 0.75 (green to yellow)
    mask = (values > 0.5) & (values <= 0.75)
    r[mask] = 4.0 * (values[mask] - 0.5)
    g[mask] = 1.0
    b[mask] = 0.0
    
    # Segment 4: 0.75 - 1.0 (yellow to red)
    mask = values > 0.75
    r[mask] = 1.0
    g[mask] = 1.0 - 4.0 * (values[mask] - 0.75)
    b[mask] = 0.0
    
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def grayscale_colormap(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """Apply grayscale colormap to normalized values [0, 1].
    
    Args:
        values: Normalized depth values in range [0, 1], shape [N] or [N, 1]
        invert: If True, closer points are brighter (white), farther are darker
        
    Returns:
        RGB colors (grayscale) in range [0, 1], shape [N, 3]
    """
    values = np.asarray(values).flatten()
    values = np.clip(values, 0.0, 1.0)
    
    if invert:
        values = 1.0 - values
    
    # Stack same value for R, G, B
    return np.stack([values, values, values], axis=-1).astype(np.float32)


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Apply turbo colormap (improved rainbow) to normalized values [0, 1].
    
    Turbo is perceptually uniform and doesn't have the issues of jet colormap.
    
    Args:
        values: Normalized depth values in range [0, 1], shape [N] or [N, 1]
        
    Returns:
        RGB colors in range [0, 1], shape [N, 3]
    """
    values = np.asarray(values).flatten()
    values = np.clip(values, 0.0, 1.0)
    
    # Turbo colormap coefficients (simplified polynomial approximation)
    r = np.clip(
        0.13572138 + values * (4.6153926 + values * (-42.66032258 + 
        values * (132.13108234 + values * (-152.94239396 + values * 59.28637943)))),
        0.0, 1.0
    )
    g = np.clip(
        0.09140261 + values * (2.19418839 + values * (4.84296658 + 
        values * (-14.18503333 + values * (4.27729857 + values * 2.82956604)))),
        0.0, 1.0
    )
    b = np.clip(
        0.1066733 + values * (12.64194608 + values * (-60.58204836 + 
        values * (110.36276771 + values * (-89.90310912 + values * 27.34824973)))),
        0.0, 1.0
    )
    
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def viridis_colormap(values: np.ndarray) -> np.ndarray:
    """Apply viridis colormap to normalized values [0, 1].
    
    Viridis is perceptually uniform and colorblind friendly.
    
    Args:
        values: Normalized depth values in range [0, 1], shape [N] or [N, 1]
        
    Returns:
        RGB colors in range [0, 1], shape [N, 3]
    """
    values = np.asarray(values).flatten()
    values = np.clip(values, 0.0, 1.0)
    
    # Simplified viridis approximation
    r = np.clip(
        0.267004 + values * (-0.004874 + values * (1.404613 + values * (-1.839716 + values * 1.171995))),
        0.0, 1.0
    )
    g = np.clip(
        0.004874 + values * (1.015861 + values * (-0.194222 + values * (0.307382 + values * -0.126894))),
        0.0, 1.0
    )
    b = np.clip(
        0.329415 + values * (1.242458 + values * (-4.903168 + values * (7.172552 + values * -3.843771))),
        0.0, 1.0
    )
    
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# Available colormaps
COLORMAPS = {
    "jet": jet_colormap,
    "grayscale": grayscale_colormap,
    "turbo": turbo_colormap,
    "viridis": viridis_colormap,
}


def get_colormap(name: str):
    """Get a colormap function by name.
    
    Args:
        name: Colormap name ('jet', 'grayscale', 'turbo', 'viridis')
        
    Returns:
        Colormap function
        
    Raises:
        ValueError: If colormap name is unknown
    """
    if name not in COLORMAPS:
        raise ValueError(f"Unknown colormap: {name}. Available: {list(COLORMAPS.keys())}")
    return COLORMAPS[name]
