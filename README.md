# Depth Map Visualization Plugin for LichtFeld Studio

A plugin that creates a colored depth map for Gaussian Splats and point clouds, with live preview, adjustable depth ranges from selected points, and an optional oriented region box.

## Installation (LichtFeld Studio v0.5+)

In LichtFeld Studio:
1. Open the Plugins panel.
2. Enter: https://github.com/jacobvanbeets/lichtfeld-depthmap-plugin
3. Click Install.

Manual:
`ash
git clone https://github.com/jacobvanbeets/lichtfeld-depthmap-plugin ~/.lichtfeld/plugins/lichtfeld-depthmap-plugin
`

## Usage
- Open the "Depth Map" panel in the side panel.
- Pick colormap (Jet, Grayscale, Turbo, Viridis) and axis (Z/Y/X/Camera).
- Optionally set the depth range from two selected points (S + click).
- Optionally enable the Region Box to limit coloring to an OBB; show a non-destructive box visualization.
- Apply/Update/Restore from the panel.

## Project
- __init__.py: plugin registration and exports
- core/: depth computation and colormap helpers
- panels/: UI panel
- pyproject.toml: plugin metadata and dependencies

## License
GPL-3.0-or-later (same as LichtFeld Studio)
