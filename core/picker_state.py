# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared state for point picker tool."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PickerState:
    """Shared state for tracking which point is being picked."""
    
    # Which point we're picking (1 or 2), or None if not picking
    picking_point: Optional[int] = None
    
    # Node name for pending pick
    _pending_node_name: Optional[str] = None
    
    def is_picking(self) -> bool:
        """Check if we're currently waiting for a selection."""
        return self.picking_point is not None
    
    def clear(self):
        """Clear picking state."""
        self.picking_point = None
        self._pending_node_name = None


# Global singleton
_state = PickerState()


def get_picker_state() -> PickerState:
    """Get the global picker state."""
    return _state
