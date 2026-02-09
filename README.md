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
1. Open the "Depth Map" panel in the side panel.
2. Pick colormap (Jet, Grayscale, Turbo, Viridis) and axis (Z/Y/X/Camera).
3. Set custom depth range using one of two methods:

### Method 1: Click to Pick (Default)
- Click "Pick Point 1" or "Pick Point 2", then click directly on the model to set min/max depth values.
- You can click multiple times to adjust the point - each click updates the depth value.
- Click the red "Stop Picking" button or press ESC to exit picking mode.

### Method 2: Selection-Based (Old Method)
Useful when XYZ picking coordinates are unreliable:
1. Check "Use Selection Method (old)" in the Depth Range section.
2. Use the Splat Select tool to select gaussians at your desired min depth location.
3. Click "Set Point 1 from Selection".
4. Select gaussians at your desired max depth location.
5. Click "Set Point 2 from Selection".

4. Fine-tune the depth values with the +/- buttons or input fields.
5. Enable "Live Preview" to see changes in real-time.
6. Use "Restore Original" to revert to original colors.

## Project
- __init__.py: plugin registration and exports
- core/: depth computation and colormap helpers
- panels/: UI panel
- pyproject.toml: plugin metadata and dependencies

## License
GPL-3.0-or-later (same as LichtFeld Studio)
