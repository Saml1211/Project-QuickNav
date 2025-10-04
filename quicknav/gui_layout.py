#!/usr/bin/env python3
"""
GUI Layout Manager Module

Handles responsive layout management, DPI scaling, and window sizing
for the Project QuickNav GUI application.
"""

import tkinter as tk
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GUIConstants:
    """Constants used throughout the GUI application."""
    # Window dimensions
    DEFAULT_MIN_WIDTH = 420
    DEFAULT_MIN_HEIGHT = 720
    COMPILED_ENV_CHECK_DELAY = 200  # ms
    DPI_SCALE_THRESHOLD = 1.1
    THEME_REFRESH_DELAY = 50  # ms

    # UI Layout
    SECTION_PADDING = 16
    ELEMENT_PADDING = 8
    BUTTON_WIDTH = 16
    ENTRY_WIDTH = 8

    # Animation and timing
    RESIZE_DEBOUNCE_DELAY = 100  # ms
    AUTOCOMPLETE_DEBOUNCE_DELAY = 300  # ms
    WINDOW_AUTO_HIDE_DELAY = 1500  # ms

    # Display scaling
    DPI_SCALE_MIN = 1.0
    DPI_SCALE_MAX = 3.0
    STANDARD_DPI = 96

    # Keyboard shortcuts
    HOTKEY_RESET_WINDOW = '<Control-Shift-R>'
    HOTKEY_TOGGLE_THEME = '<Control-d>'
    HOTKEY_RESET_FORM = '<Control-r>'


class LayoutManager:
    """
    Manages layout, DPI scaling, and responsive design for the GUI.

    Responsibilities:
    - DPI scaling calculations and application
    - Responsive layout adjustments based on window size
    - Minimum/maximum window size enforcement
    - Geometry calculations and validation
    """

    def __init__(self, root: tk.Tk, settings):
        """
        Initialize the layout manager.

        Args:
            root: The Tkinter root window
            settings: SettingsManager instance for persistent settings
        """
        self.root = root
        self.settings = settings
        self.dpi_scale = self._get_dpi_scale()
        self.min_width = GUIConstants.DEFAULT_MIN_WIDTH
        self.min_height = GUIConstants.DEFAULT_MIN_HEIGHT

        # Apply initial DPI scaling
        self._apply_dpi_scaling()

        logger.info(f"LayoutManager initialized with DPI scale: {self.dpi_scale}")

    def _get_dpi_scale(self) -> float:
        """
        Get DPI scaling factor for the display.

        Returns:
            DPI scale factor (1.0 = 96 DPI, 2.0 = 192 DPI, etc.)
        """
        try:
            dpi = self.root.winfo_fpixels('1i')
            scale = dpi / GUIConstants.STANDARD_DPI
            # Clamp scale between min and max values
            return max(GUIConstants.DPI_SCALE_MIN, min(GUIConstants.DPI_SCALE_MAX, scale))
        except Exception as e:
            logger.warning(f"Failed to get DPI scale: {e}")
            return GUIConstants.DPI_SCALE_MIN

    def _apply_dpi_scaling(self):
        """Apply DPI scaling to fonts and window size."""
        if self.dpi_scale > GUIConstants.DPI_SCALE_THRESHOLD:
            try:
                logger.info(f"Applying DPI scaling factor: {self.dpi_scale}")

                # Scale fonts
                import tkinter.font as tkFont
                default_font = tkFont.nametofont("TkDefaultFont")
                original_size = default_font['size']
                new_size = int(original_size * self.dpi_scale)
                default_font.configure(size=new_size)
                logger.info(f"Font scaled from {original_size} to {new_size}")

                # Adjust minimum window size for DPI
                original_min_width = self.min_width
                original_min_height = self.min_height
                self.min_width = int(self.min_width * self.dpi_scale)
                self.min_height = int(self.min_height * self.dpi_scale)
                logger.info(f"Min window size scaled from {original_min_width}x{original_min_height} to {self.min_width}x{self.min_height}")
            except Exception as e:
                logger.warning(f"Failed to apply DPI scaling: {e}")

    def update_dpi_scaling_for_resize(self, width: int, height: int):
        """
        Update DPI scaling calculations during resize for better consistency.

        Args:
            width: Current window width
            height: Current window height
        """
        try:
            old_dpi_scale = self.dpi_scale
            self.dpi_scale = self._get_dpi_scale()

            # Only reapply scaling if there's a significant change (>5%)
            if abs(self.dpi_scale - old_dpi_scale) > 0.05:
                logger.info(f"DPI scale changed during resize: {old_dpi_scale} -> {self.dpi_scale}")

                if self.dpi_scale > GUIConstants.DPI_SCALE_THRESHOLD:
                    import tkinter.font as tkFont
                    default_font = tkFont.nametofont("TkDefaultFont")
                    original_size = default_font['size']
                    new_size = int(original_size * self.dpi_scale)
                    default_font.configure(size=new_size)

                    # Update minimum window size for new DPI scale
                    base_min_width = GUIConstants.DEFAULT_MIN_WIDTH
                    base_min_height = GUIConstants.DEFAULT_MIN_HEIGHT
                    self.min_width = int(base_min_width * self.dpi_scale)
                    self.min_height = int(base_min_height * self.dpi_scale)

                    self.root.minsize(self.min_width, self.min_height)
                    logger.info(f"DPI scaling updated for resize - Min size: {self.min_width}x{self.min_height}")
        except Exception as e:
            logger.warning(f"Failed to update DPI scaling during resize: {e}")

    def get_consistent_padding(self) -> int:
        """
        Get consistent padding value based on DPI scaling and window size.

        Returns:
            Padding value in pixels
        """
        base_padding = max(8, int(12 * self.dpi_scale))
        current_width = self.root.winfo_width()

        # Adjust padding for very small windows to maximize content area
        if current_width > 0 and current_width < 450:
            return max(4, int(base_padding * 0.7))
        return base_padding

    def get_consistent_spacing(self) -> int:
        """
        Get consistent spacing value for internal component spacing.

        Returns:
            Spacing value in pixels
        """
        return max(6, int(8 * self.dpi_scale))

    def get_responsive_geometry(self) -> str:
        """
        Calculate responsive geometry based on screen size and DPI.

        Returns:
            Geometry string in format "WIDTHxHEIGHT+X+Y"
        """
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Calculate window dimensions (30% of screen, but within reasonable limits)
            width = max(self.min_width, min(int(screen_width * 0.3), 600))
            height = max(self.min_height, min(int(screen_height * 0.6), 900))

            # Center the window
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

            geometry = f"{width}x{height}+{x}+{y}"
            logger.info(f"Calculated responsive geometry: {geometry}")
            return geometry
        except Exception as e:
            logger.error(f"Error calculating responsive geometry: {e}")
            return f"{self.min_width}x{self.min_height}+100+100"

    def is_geometry_on_screen(self, geometry: str) -> bool:
        """
        Check if a geometry string represents a window that would be visible.

        Args:
            geometry: Geometry string in format "WIDTHxHEIGHT+X+Y"

        Returns:
            True if geometry is valid and on screen, False otherwise
        """
        try:
            parts = geometry.replace('+', ' ').replace('x', ' ').split()
            if len(parts) != 4:
                return False

            width, height, x, y = map(int, parts)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Check if at least some part of the window would be visible
            if x >= screen_width or y >= screen_height:
                return False
            if x + width < 0 or y + height < 0:
                return False

            return True
        except Exception:
            return False

    def enforce_minimum_size(self):
        """Ensure the window meets minimum size requirements."""
        try:
            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()

            if current_width < self.min_width or current_height < self.min_height:
                new_width = max(current_width, self.min_width)
                new_height = max(current_height, self.min_height)

                # Get current position
                x = self.root.winfo_x()
                y = self.root.winfo_y()

                # Apply new geometry
                self.root.geometry(f"{new_width}x{new_height}+{x}+{y}")
                logger.info(f"Enforced minimum window size: {new_width}x{new_height}")
        except Exception as e:
            logger.error(f"Error enforcing minimum size: {e}")

    def calculate_minimum_width(self) -> int:
        """
        Calculate the minimum width required for the current content.

        Returns:
            Minimum width in pixels
        """
        base_width = GUIConstants.DEFAULT_MIN_WIDTH
        padding = self.get_consistent_padding()

        # Account for DPI scaling and padding
        min_width = int(base_width * self.dpi_scale)
        min_width += padding * 2  # Left and right padding

        return min_width

    def calculate_minimum_height(self) -> int:
        """
        Calculate the minimum height required for the current content.

        Returns:
            Minimum height in pixels
        """
        base_height = GUIConstants.DEFAULT_MIN_HEIGHT
        padding = self.get_consistent_padding()

        # Account for DPI scaling and padding
        min_height = int(base_height * self.dpi_scale)
        min_height += padding * 2  # Top and bottom padding

        return min_height

    def get_scale_adjusted_value(self, base_value: int) -> int:
        """
        Get a DPI-scaled value.

        Args:
            base_value: Base value at 96 DPI

        Returns:
            Scaled value based on current DPI
        """
        return max(1, int(base_value * self.dpi_scale))

    def reset_window_to_default(self):
        """Reset window size and position to defaults."""
        try:
            geometry = self.get_responsive_geometry()
            self.root.geometry(geometry)
            self.settings.set_window_geometry(geometry)
            logger.info(f"Window reset to default geometry: {geometry}")
        except Exception as e:
            logger.error(f"Error resetting window: {e}")
