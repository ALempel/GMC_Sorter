import numpy as np
from matplotlib.path import Path


class Bound:
    """Represents a bound (polygon) for unit definition"""
    def __init__(self, property_indices, vertices, unit_label):
        """
        Args:
            property_indices: tuple (x_idx, y_idx) indicating which properties define the bound
            vertices: list of (x, y) tuples defining the polygon vertices
            unit_label: int, label for this unit
        """
        self.property_indices = property_indices
        self.vertices = vertices
        self.path = Path(vertices)
        self.unit_label = unit_label


class Unit:
    """Represents a spike sorting unit. Spike membership is determined by the unit assignment vector (unit_id), not stored here."""
    def __init__(self, unit_variables, mean_y_pos):
        """
        Args:
            unit_variables: for bounded units, list of Bound objects; for gaussian, list of dicts with model params
            mean_y_pos: float, mean y-position of spikes in this unit (for sorting)
        """
        self.unit_variables = unit_variables
        self.mean_y_pos = mean_y_pos

    def is_gaussian(self):
        """True if this unit is a Gaussian model (unit_variables are dicts with 'gaussian' key)."""
        if hasattr(self, 'unit_type'):
            return self.unit_type == 'gaussian'
        return bool(self.unit_variables) and isinstance(self.unit_variables[0], dict) and self.unit_variables[0].get('gaussian')

    def is_bounded(self):
        """True if this unit is bounded (unit_variables are Bound objects or dicts with 'vertices')."""
        if hasattr(self, 'unit_type'):
            return self.unit_type == 'bounded'
        if not self.unit_variables:
            return False
        first = self.unit_variables[0]
        if isinstance(first, Bound):
            return True
        return isinstance(first, dict) and 'vertices' in first
