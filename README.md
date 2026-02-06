# Depth Map Visualization Plugin for LichtFeld Studio

A plugin that creates colored depth maps for Gaussian Splats and point clouds, with live preview and adjustable depth ranges from selected points.
<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/8bbc8afd-d9ca-4f98-a05e-c8befa96c8ab" />

## Installation (LichtFeld Studio v0.5+)

In LichtFeld Studio:
1. Open the Plugins panel.
2. Enter: https://github.com/jacobvanbeets/lichtfeld-depthmap-plugin
3. Click Install.

Manual:
```bash
git clone https://github.com/jacobvanbeets/lichtfeld-depthmap-plugin ~/.lichtfeld/plugins/lichtfeld-depthmap-plugin
```

## Usage
- Open the "Depth Map" panel in the side panel.
- Pick colormap (Jet, Grayscale, Turbo, Viridis) and axis (Z/Y/X/Camera).
- Optionally set the depth range by selecting points (switch to Select mode in the toolbar, then click on the model).
- Apply/Update/Restore from the panel.

## Project
- __init__.py: plugin registration and exports
- core/: depth computation and colormap helpers
- panels/: UI panel
- pyproject.toml: plugin metadata and dependencies

## License
GPL-3.0-or-later (same as LichtFeld Studio)
