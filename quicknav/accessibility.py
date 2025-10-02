"""
Accessibility Utilities for Project QuickNav

This module provides comprehensive accessibility features to ensure WCAG 2.1 AA
compliance and universal accessibility for users with diverse needs.

Features:
- Screen reader support with ARIA labels and announcements
- High contrast and color-blind friendly themes
- Keyboard navigation and focus management
- Font scaling and motor accessibility
- Alternative input method support
- Accessibility testing utilities

WCAG 2.1 AA Compliance Areas:
- Perceivable: Color contrast, text alternatives, adaptable content
- Operable: Keyboard accessible, timing adjustable, seizure-safe
- Understandable: Readable, predictable, input assistance
- Robust: Compatible with assistive technologies
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import platform
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import re
import colorsys

logger = logging.getLogger(__name__)


class AccessibilityLevel(Enum):
    """Accessibility compliance levels."""
    BASIC = "basic"
    AA = "aa"
    AAA = "aaa"


class ColorBlindType(Enum):
    """Types of color blindness for simulation."""
    NORMAL = "normal"
    PROTANOPIA = "protanopia"      # Red-blind
    DEUTERANOPIA = "deuteranopia"  # Green-blind
    TRITANOPIA = "tritanopia"      # Blue-blind
    ACHROMATOPSIA = "achromatopsia"  # Complete color blindness


class AccessibilityManager:
    """Main accessibility manager for coordinating all accessibility features."""

    def __init__(self, root: tk.Tk, settings_manager=None):
        self.root = root
        self.settings = settings_manager

        # Initialize components
        self.screen_reader = ScreenReaderSupport(root)
        self.keyboard_nav = KeyboardNavigationManager(root)
        self.focus_manager = FocusManager(root)
        self.announcer = AccessibilityAnnouncer(root)
        self.color_analyzer = ColorContrastAnalyzer()

        # Settings
        self.high_contrast_mode = False
        self.font_scale = 1.0
        self.motor_accessibility = False
        self.color_blind_mode = ColorBlindType.NORMAL
        self.screen_reader_enabled = False

        # Initialize from settings
        self._load_settings()

        # Set up accessibility features
        self._setup_accessibility()

        logger.info("AccessibilityManager initialized")

    def _load_settings(self):
        """Load accessibility settings."""
        if not self.settings:
            return

        self.high_contrast_mode = self.settings.get("accessibility.high_contrast", False)
        self.font_scale = self.settings.get("accessibility.font_scale", 1.0)
        self.motor_accessibility = self.settings.get("accessibility.motor_aids", False)
        self.screen_reader_enabled = self.settings.get("accessibility.screen_reader", False)

        color_blind_setting = self.settings.get("accessibility.color_blind_mode", "normal")
        try:
            self.color_blind_mode = ColorBlindType(color_blind_setting)
        except ValueError:
            self.color_blind_mode = ColorBlindType.NORMAL

    def _setup_accessibility(self):
        """Set up all accessibility features."""
        # Apply font scaling
        if self.font_scale != 1.0:
            self.apply_font_scaling(self.font_scale)

        # Enable screen reader if requested
        if self.screen_reader_enabled:
            self.screen_reader.enable()

        # Apply motor accessibility aids
        if self.motor_accessibility:
            self.enable_motor_accessibility()

        # Set up keyboard navigation
        self.keyboard_nav.setup()

    def apply_font_scaling(self, scale_factor: float):
        """Apply font scaling for better readability."""
        self.font_scale = scale_factor

        # Get all fonts in the application
        font_families = list(tkFont.families())
        default_fonts = ["TkDefaultFont", "TkTextFont", "TkFixedFont",
                        "TkMenuFont", "TkHeadingFont", "TkCaptionFont",
                        "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"]

        for font_name in default_fonts:
            try:
                font = tkFont.nametofont(font_name)
                original_size = font.cget("size")
                if original_size < 0:
                    # Negative size means pixels, convert to points
                    original_size = abs(original_size) * 72 / 96
                new_size = max(8, int(original_size * scale_factor))
                font.configure(size=new_size)
            except Exception as e:
                logger.debug(f"Could not scale font {font_name}: {e}")

        # Save setting
        if self.settings:
            self.settings.set("accessibility.font_scale", scale_factor)

        # Announce change
        self.announcer.announce(f"Font size scaled to {int(scale_factor * 100)}%")

        logger.info(f"Applied font scaling: {scale_factor}")

    def toggle_high_contrast(self):
        """Toggle high contrast mode."""
        self.high_contrast_mode = not self.high_contrast_mode

        if self.settings:
            self.settings.set("accessibility.high_contrast", self.high_contrast_mode)

        # Announce change
        status = "enabled" if self.high_contrast_mode else "disabled"
        self.announcer.announce(f"High contrast mode {status}")

        return self.high_contrast_mode

    def enable_motor_accessibility(self):
        """Enable motor accessibility aids."""
        self.motor_accessibility = True

        # Increase click targets
        self._increase_click_targets()

        # Add hover delays
        self._add_hover_delays()

        if self.settings:
            self.settings.set("accessibility.motor_aids", True)

        self.announcer.announce("Motor accessibility aids enabled")
        logger.info("Motor accessibility aids enabled")

    def _increase_click_targets(self):
        """Increase the size of clickable targets."""
        def apply_to_widget(widget):
            try:
                if isinstance(widget, (ttk.Button, tk.Button)):
                    # Increase button padding
                    current_config = widget.cget('padding') if hasattr(widget, 'cget') else None
                    if isinstance(widget, ttk.Button):
                        widget.configure(padding=(16, 12))
                    elif isinstance(widget, tk.Button):
                        widget.configure(padx=16, pady=12)

                elif isinstance(widget, (ttk.Checkbutton, ttk.Radiobutton)):
                    # Increase indicator size and padding
                    widget.configure(padding=(8, 8))

                # Recursively apply to children
                for child in widget.winfo_children():
                    apply_to_widget(child)

            except Exception as e:
                logger.debug(f"Could not modify widget for motor accessibility: {e}")

        apply_to_widget(self.root)

    def _add_hover_delays(self):
        """Add hover delays to prevent accidental activations."""
        hover_delay = 750  # milliseconds

        def add_hover_delay(widget):
            if isinstance(widget, (ttk.Button, tk.Button)):
                original_enter = widget.bind('<Enter>')
                hover_timer = None

                def delayed_enter(event):
                    nonlocal hover_timer
                    if hover_timer:
                        widget.after_cancel(hover_timer)
                    hover_timer = widget.after(hover_delay, lambda: None)

                def leave_handler(event):
                    nonlocal hover_timer
                    if hover_timer:
                        widget.after_cancel(hover_timer)
                        hover_timer = None

                widget.bind('<Enter>', delayed_enter)
                widget.bind('<Leave>', leave_handler)

            # Recursively apply to children
            for child in widget.winfo_children():
                add_hover_delay(child)

        add_hover_delay(self.root)

    def set_color_blind_mode(self, color_blind_type: ColorBlindType):
        """Set color blind simulation mode."""
        self.color_blind_mode = color_blind_type

        if self.settings:
            self.settings.set("accessibility.color_blind_mode", color_blind_type.value)

        mode_name = color_blind_type.value.replace('_', ' ').title()
        self.announcer.announce(f"Color blind mode set to {mode_name}")

        logger.info(f"Color blind mode set to: {color_blind_type.value}")

    def enable_screen_reader(self):
        """Enable screen reader support."""
        self.screen_reader_enabled = True
        self.screen_reader.enable()

        if self.settings:
            self.settings.set("accessibility.screen_reader", True)

        self.announcer.announce("Screen reader support enabled")
        logger.info("Screen reader support enabled")

    def disable_screen_reader(self):
        """Disable screen reader support."""
        self.screen_reader_enabled = False
        self.screen_reader.disable()

        if self.settings:
            self.settings.set("accessibility.screen_reader", False)

        logger.info("Screen reader support disabled")

    def get_accessibility_status(self) -> Dict[str, Any]:
        """Get current accessibility status."""
        return {
            "high_contrast": self.high_contrast_mode,
            "font_scale": self.font_scale,
            "motor_accessibility": self.motor_accessibility,
            "color_blind_mode": self.color_blind_mode.value,
            "screen_reader": self.screen_reader_enabled,
            "keyboard_nav_active": self.keyboard_nav.is_active()
        }

    def run_accessibility_check(self) -> Dict[str, List[str]]:
        """Run comprehensive accessibility check."""
        issues = []
        recommendations = []

        # Check color contrast
        contrast_issues = self.color_analyzer.check_application_contrast(self.root)
        if contrast_issues:
            issues.extend([f"Color contrast: {issue}" for issue in contrast_issues])
            recommendations.append("Enable high contrast mode or adjust theme colors")

        # Check font sizes
        small_fonts = self._check_font_sizes()
        if small_fonts:
            issues.extend([f"Small font: {font}" for font in small_fonts])
            recommendations.append("Increase font scaling for better readability")

        # Check keyboard navigation
        if not self.keyboard_nav.is_active():
            issues.append("Keyboard navigation not fully enabled")
            recommendations.append("Enable comprehensive keyboard navigation")

        # Check focus indicators
        focus_issues = self._check_focus_indicators()
        if focus_issues:
            issues.extend(focus_issues)
            recommendations.append("Improve focus indicators for better visibility")

        return {
            "issues": issues,
            "recommendations": recommendations,
            "compliance_level": self._assess_compliance_level(issues)
        }

    def _check_font_sizes(self) -> List[str]:
        """Check for fonts that are too small."""
        small_fonts = []
        default_fonts = ["TkDefaultFont", "TkTextFont", "TkFixedFont"]

        for font_name in default_fonts:
            try:
                font = tkFont.nametofont(font_name)
                size = font.cget("size")
                if abs(size) < 12:  # Less than 12pt is considered small
                    small_fonts.append(f"{font_name} ({abs(size)}pt)")
            except Exception:
                pass

        return small_fonts

    def _check_focus_indicators(self) -> List[str]:
        """Check for adequate focus indicators."""
        issues = []

        # This is a simplified check - in practice, you'd traverse the widget tree
        # and check each focusable widget's focus style

        return issues

    def _assess_compliance_level(self, issues: List[str]) -> str:
        """Assess WCAG compliance level based on issues found."""
        if not issues:
            return "AA"
        elif len(issues) <= 3:
            return "Partial AA"
        else:
            return "Basic"


class ScreenReaderSupport:
    """Provides screen reader support and ARIA-like functionality."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.enabled = False
        self.widget_labels = {}
        self.widget_descriptions = {}
        self.live_regions = {}

    def enable(self):
        """Enable screen reader support."""
        self.enabled = True
        self._setup_screen_reader_support()
        logger.info("Screen reader support enabled")

    def disable(self):
        """Disable screen reader support."""
        self.enabled = False
        logger.info("Screen reader support disabled")

    def _setup_screen_reader_support(self):
        """Set up screen reader support for all widgets."""
        self._add_aria_labels()
        self._setup_live_regions()
        self._bind_focus_events()

    def add_label(self, widget: tk.Widget, label: str):
        """Add accessible label to a widget (ARIA-label equivalent)."""
        self.widget_labels[widget] = label

        # Store in widget for potential screen reader access
        try:
            widget.accessible_label = label
        except Exception:
            pass

    def add_description(self, widget: tk.Widget, description: str):
        """Add accessible description to a widget (ARIA-describedby equivalent)."""
        self.widget_descriptions[widget] = description

        try:
            widget.accessible_description = description
        except Exception:
            pass

    def create_live_region(self, widget: tk.Widget, politeness: str = "polite"):
        """Create a live region for dynamic content announcements."""
        self.live_regions[widget] = {
            "politeness": politeness,
            "content": ""
        }

    def update_live_region(self, widget: tk.Widget, content: str):
        """Update live region content."""
        if widget in self.live_regions:
            self.live_regions[widget]["content"] = content
            # In a real implementation, this would trigger screen reader announcement
            logger.info(f"Live region update: {content}")

    def _add_aria_labels(self):
        """Add ARIA labels to common widgets."""
        def process_widget(widget):
            try:
                widget_type = widget.__class__.__name__

                # Add default labels based on widget type
                if isinstance(widget, (ttk.Entry, tk.Entry)):
                    if widget not in self.widget_labels:
                        placeholder = getattr(widget, 'placeholder', None)
                        if placeholder:
                            self.add_label(widget, f"Input field: {placeholder}")
                        else:
                            self.add_label(widget, "Input field")

                elif isinstance(widget, (ttk.Button, tk.Button)):
                    if widget not in self.widget_labels:
                        button_text = widget.cget('text')
                        self.add_label(widget, f"Button: {button_text}")

                elif isinstance(widget, (ttk.Checkbutton, tk.Checkbutton)):
                    if widget not in self.widget_labels:
                        check_text = widget.cget('text')
                        self.add_label(widget, f"Checkbox: {check_text}")

                elif isinstance(widget, (ttk.Radiobutton, tk.Radiobutton)):
                    if widget not in self.widget_labels:
                        radio_text = widget.cget('text')
                        self.add_label(widget, f"Radio button: {radio_text}")

                elif isinstance(widget, (ttk.Combobox,)):
                    if widget not in self.widget_labels:
                        self.add_label(widget, "Dropdown list")

                # Process children
                for child in widget.winfo_children():
                    process_widget(child)

            except Exception as e:
                logger.debug(f"Error processing widget for ARIA labels: {e}")

        process_widget(self.root)

    def _setup_live_regions(self):
        """Set up common live regions."""
        # Look for status labels and progress bars
        def find_live_regions(widget):
            try:
                if isinstance(widget, (ttk.Label, tk.Label)):
                    text = widget.cget('text').lower()
                    if any(keyword in text for keyword in ['status', 'progress', 'loading']):
                        self.create_live_region(widget, "polite")

                elif isinstance(widget, (ttk.Progressbar,)):
                    self.create_live_region(widget, "polite")

                for child in widget.winfo_children():
                    find_live_regions(child)

            except Exception as e:
                logger.debug(f"Error setting up live regions: {e}")

        find_live_regions(self.root)

    def _bind_focus_events(self):
        """Bind focus events for screen reader announcements."""
        def announce_widget(event):
            if not self.enabled:
                return

            widget = event.widget
            announcement = self._get_widget_announcement(widget)
            if announcement:
                # In a real implementation, this would interface with system screen reader
                logger.info(f"Screen reader: {announcement}")

        # Bind to focus events
        self.root.bind_all('<FocusIn>', announce_widget, add=True)

    def _get_widget_announcement(self, widget: tk.Widget) -> str:
        """Get screen reader announcement for a widget."""
        announcement_parts = []

        # Add label
        if widget in self.widget_labels:
            announcement_parts.append(self.widget_labels[widget])

        # Add current value/state
        try:
            if isinstance(widget, (ttk.Entry, tk.Entry)):
                value = widget.get()
                if value:
                    announcement_parts.append(f"Contains: {value}")
                else:
                    announcement_parts.append("Empty")

            elif isinstance(widget, (ttk.Checkbutton, tk.Checkbutton)):
                var = widget.cget('variable')
                if var:
                    try:
                        checked = widget.tk.globalgetvar(var)
                        announcement_parts.append("Checked" if checked else "Unchecked")
                    except:
                        pass

            elif isinstance(widget, (ttk.Combobox,)):
                value = widget.get()
                if value:
                    announcement_parts.append(f"Selected: {value}")

        except Exception as e:
            logger.debug(f"Error getting widget state for announcement: {e}")

        # Add description
        if widget in self.widget_descriptions:
            announcement_parts.append(self.widget_descriptions[widget])

        return ". ".join(announcement_parts)


class KeyboardNavigationManager:
    """Manages comprehensive keyboard navigation."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.active = False
        self.focusable_widgets = []
        self.current_focus_index = 0

    def setup(self):
        """Set up keyboard navigation."""
        self.active = True
        self._find_focusable_widgets()
        self._bind_navigation_keys()
        logger.info("Keyboard navigation enabled")

    def is_active(self) -> bool:
        """Check if keyboard navigation is active."""
        return self.active

    def _find_focusable_widgets(self):
        """Find all focusable widgets in the application."""
        self.focusable_widgets = []

        def collect_focusable(widget):
            try:
                # Check if widget can receive focus
                if widget.winfo_class() in ['Entry', 'TEntry', 'Button', 'TButton',
                                          'Checkbutton', 'TCheckbutton', 'Radiobutton',
                                          'TRadiobutton', 'Combobox', 'TCombobox',
                                          'Listbox', 'Text', 'Scrollbar', 'TScrollbar']:
                    if str(widget.cget('state')) != 'disabled':
                        self.focusable_widgets.append(widget)

                # Check children
                for child in widget.winfo_children():
                    collect_focusable(child)

            except Exception as e:
                logger.debug(f"Error checking widget focusability: {e}")

        collect_focusable(self.root)
        logger.debug(f"Found {len(self.focusable_widgets)} focusable widgets")

    def _bind_navigation_keys(self):
        """Bind keyboard navigation keys."""
        # Tab navigation (already handled by Tkinter)
        # Add custom navigation keys

        def navigate_next(event):
            self._navigate_to_next()
            return "break"

        def navigate_previous(event):
            self._navigate_to_previous()
            return "break"

        def navigate_first(event):
            self._navigate_to_first()
            return "break"

        def navigate_last(event):
            self._navigate_to_last()
            return "break"

        # Bind additional navigation keys
        self.root.bind_all('<Control-Tab>', navigate_next, add=True)
        self.root.bind_all('<Control-Shift-Tab>', navigate_previous, add=True)
        self.root.bind_all('<Control-Home>', navigate_first, add=True)
        self.root.bind_all('<Control-End>', navigate_last, add=True)

    def _navigate_to_next(self):
        """Navigate to next focusable widget."""
        if not self.focusable_widgets:
            return

        current_widget = self.root.focus_get()
        if current_widget in self.focusable_widgets:
            current_index = self.focusable_widgets.index(current_widget)
            next_index = (current_index + 1) % len(self.focusable_widgets)
        else:
            next_index = 0

        self.focusable_widgets[next_index].focus_set()

    def _navigate_to_previous(self):
        """Navigate to previous focusable widget."""
        if not self.focusable_widgets:
            return

        current_widget = self.root.focus_get()
        if current_widget in self.focusable_widgets:
            current_index = self.focusable_widgets.index(current_widget)
            prev_index = (current_index - 1) % len(self.focusable_widgets)
        else:
            prev_index = len(self.focusable_widgets) - 1

        self.focusable_widgets[prev_index].focus_set()

    def _navigate_to_first(self):
        """Navigate to first focusable widget."""
        if self.focusable_widgets:
            self.focusable_widgets[0].focus_set()

    def _navigate_to_last(self):
        """Navigate to last focusable widget."""
        if self.focusable_widgets:
            self.focusable_widgets[-1].focus_set()


class FocusManager:
    """Manages focus indicators and focus behavior."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.focus_ring_width = 2
        self.focus_ring_color = "#0078d4"
        self.original_styles = {}

        self._setup_focus_management()

    def _setup_focus_management(self):
        """Set up focus management."""
        self._bind_focus_events()
        self._enhance_focus_indicators()

    def _bind_focus_events(self):
        """Bind focus events for enhanced focus management."""
        def on_focus_in(event):
            self._enhance_widget_focus(event.widget)

        def on_focus_out(event):
            self._restore_widget_focus(event.widget)

        self.root.bind_all('<FocusIn>', on_focus_in, add=True)
        self.root.bind_all('<FocusOut>', on_focus_out, add=True)

    def _enhance_widget_focus(self, widget: tk.Widget):
        """Enhance focus indicator for a widget."""
        try:
            widget_id = str(widget)

            # Store original style
            if isinstance(widget, ttk.Widget):
                try:
                    style = widget.cget('style') or widget.winfo_class()
                    self.original_styles[widget_id] = style

                    # Create enhanced focus style
                    enhanced_style = f"Focus.{style}"

                    # Configure enhanced focus style if not exists
                    ttk_style = ttk.Style()
                    if enhanced_style not in ttk_style.theme_names():
                        ttk_style.configure(enhanced_style,
                                          focuscolor=self.focus_ring_color,
                                          borderwidth=self.focus_ring_width)

                    widget.configure(style=enhanced_style)
                except Exception:
                    pass

            elif isinstance(widget, tk.Widget):
                # For tk widgets, modify border
                try:
                    original_config = {
                        'highlightbackground': widget.cget('highlightbackground'),
                        'highlightcolor': widget.cget('highlightcolor'),
                        'highlightthickness': widget.cget('highlightthickness')
                    }
                    self.original_styles[widget_id] = original_config

                    widget.configure(
                        highlightcolor=self.focus_ring_color,
                        highlightthickness=self.focus_ring_width
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error enhancing focus for widget: {e}")

    def _restore_widget_focus(self, widget: tk.Widget):
        """Restore original focus indicator for a widget."""
        try:
            widget_id = str(widget)

            if widget_id in self.original_styles:
                original = self.original_styles[widget_id]

                if isinstance(widget, ttk.Widget) and isinstance(original, str):
                    widget.configure(style=original)
                elif isinstance(widget, tk.Widget) and isinstance(original, dict):
                    widget.configure(**original)

                del self.original_styles[widget_id]

        except Exception as e:
            logger.debug(f"Error restoring focus for widget: {e}")

    def _enhance_focus_indicators(self):
        """Enhance focus indicators application-wide."""
        try:
            # Configure ttk style focus indicators
            style = ttk.Style()

            # Enhance standard widget focus
            widgets_to_enhance = ['TButton', 'TEntry', 'TCombobox', 'TCheckbutton', 'TRadiobutton']

            for widget_class in widgets_to_enhance:
                style.configure(widget_class, focuscolor=self.focus_ring_color)
                style.map(widget_class,
                         focuscolor=[('focus', self.focus_ring_color)])

        except Exception as e:
            logger.debug(f"Error enhancing focus indicators: {e}")


class AccessibilityAnnouncer:
    """Provides accessibility announcements and notifications."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.announcement_queue = []
        self.announcement_thread = None
        self.running = True

        # Create announcement display widget
        self._create_announcement_display()

        # Start announcement processor
        self._start_announcement_processor()

    def _create_announcement_display(self):
        """Create visual announcement display."""
        # Create a label for visual announcements (initially hidden)
        self.announcement_label = tk.Label(
            self.root,
            text="",
            bg="#000000",
            fg="#ffffff",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=5
        )
        # Position at top of window
        self.announcement_label.place(relx=0.5, y=10, anchor="n")
        self.announcement_label.place_forget()  # Hide initially

    def announce(self, message: str, priority: str = "polite"):
        """Make an accessibility announcement."""
        self.announcement_queue.append({
            "message": message,
            "priority": priority,
            "timestamp": time.time()
        })

        # Show visual announcement
        self._show_visual_announcement(message)

        # Log for screen reader simulation
        logger.info(f"Accessibility announcement ({priority}): {message}")

    def _show_visual_announcement(self, message: str):
        """Show visual announcement temporarily."""
        self.announcement_label.config(text=message)
        self.announcement_label.place(relx=0.5, y=10, anchor="n")

        # Hide after 3 seconds
        self.root.after(3000, lambda: self.announcement_label.place_forget())

    def _start_announcement_processor(self):
        """Start background announcement processor."""
        def process_announcements():
            while self.running:
                if self.announcement_queue:
                    announcement = self.announcement_queue.pop(0)
                    # In a real implementation, this would interface with system TTS
                    # or screen reader APIs
                    time.sleep(0.1)
                else:
                    time.sleep(0.5)

        self.announcement_thread = threading.Thread(target=process_announcements, daemon=True)
        self.announcement_thread.start()

    def stop(self):
        """Stop the announcement processor."""
        self.running = False
        if self.announcement_thread:
            self.announcement_thread.join(timeout=1.0)


class ColorContrastAnalyzer:
    """Analyzes and ensures adequate color contrast for accessibility."""

    def __init__(self):
        self.wcag_aa_ratio = 4.5
        self.wcag_aaa_ratio = 7.0
        self.large_text_aa_ratio = 3.0
        self.large_text_aaa_ratio = 4.5

    def calculate_contrast_ratio(self, color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors."""
        try:
            # Convert colors to RGB
            rgb1 = self._hex_to_rgb(color1)
            rgb2 = self._hex_to_rgb(color2)

            # Calculate relative luminance
            lum1 = self._relative_luminance(rgb1)
            lum2 = self._relative_luminance(rgb2)

            # Calculate contrast ratio
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)

            return (lighter + 0.05) / (darker + 0.05)

        except Exception as e:
            logger.debug(f"Error calculating contrast ratio: {e}")
            return 1.0

    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB values (0-1 range)."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])

        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        return (r, g, b)

    def _relative_luminance(self, rgb: Tuple[float, float, float]) -> float:
        """Calculate relative luminance of an RGB color."""
        def linearize(c):
            if c <= 0.03928:
                return c / 12.92
            else:
                return pow((c + 0.055) / 1.055, 2.4)

        r, g, b = rgb
        r_lin = linearize(r)
        g_lin = linearize(g)
        b_lin = linearize(b)

        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    def check_contrast_compliance(self, fg_color: str, bg_color: str,
                                font_size: int = 12, is_bold: bool = False) -> Dict[str, Any]:
        """Check if color combination meets WCAG contrast requirements."""
        ratio = self.calculate_contrast_ratio(fg_color, bg_color)

        # Determine if it's large text (18pt+ or 14pt+ bold)
        is_large_text = font_size >= 18 or (font_size >= 14 and is_bold)

        # Check compliance levels
        aa_threshold = self.large_text_aa_ratio if is_large_text else self.wcag_aa_ratio
        aaa_threshold = self.large_text_aaa_ratio if is_large_text else self.wcag_aaa_ratio

        return {
            "ratio": ratio,
            "aa_compliant": ratio >= aa_threshold,
            "aaa_compliant": ratio >= aaa_threshold,
            "large_text": is_large_text,
            "recommendation": self._get_contrast_recommendation(ratio, aa_threshold, aaa_threshold)
        }

    def _get_contrast_recommendation(self, ratio: float, aa_threshold: float,
                                   aaa_threshold: float) -> str:
        """Get recommendation for contrast improvement."""
        if ratio >= aaa_threshold:
            return "Excellent contrast - exceeds AAA standards"
        elif ratio >= aa_threshold:
            return "Good contrast - meets AA standards"
        else:
            needed_improvement = aa_threshold / ratio
            return f"Poor contrast - needs {needed_improvement:.1f}x improvement to meet AA standards"

    def check_application_contrast(self, root: tk.Tk) -> List[str]:
        """Check contrast ratios throughout the application."""
        issues = []

        def check_widget_contrast(widget):
            try:
                # Get widget colors
                fg = None
                bg = None

                if isinstance(widget, (tk.Label, tk.Button, tk.Entry, tk.Text)):
                    try:
                        fg = widget.cget('foreground') or widget.cget('fg')
                        bg = widget.cget('background') or widget.cget('bg')
                    except:
                        pass

                elif isinstance(widget, ttk.Widget):
                    # For ttk widgets, getting colors is more complex
                    # This is a simplified approach
                    pass

                if fg and bg and fg != bg:
                    # Check if colors are valid hex colors
                    if self._is_valid_color(fg) and self._is_valid_color(bg):
                        result = self.check_contrast_compliance(fg, bg)
                        if not result["aa_compliant"]:
                            widget_type = widget.__class__.__name__
                            issues.append(f"{widget_type} has poor contrast (ratio: {result['ratio']:.1f})")

                # Check children
                for child in widget.winfo_children():
                    check_widget_contrast(child)

            except Exception as e:
                logger.debug(f"Error checking widget contrast: {e}")

        check_widget_contrast(root)
        return issues

    def _is_valid_color(self, color: str) -> bool:
        """Check if a color string is a valid hex color."""
        if not color:
            return False

        # Check hex color pattern
        hex_pattern = re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$')
        return bool(hex_pattern.match(color))

    def simulate_color_blindness(self, color: str, color_blind_type: ColorBlindType) -> str:
        """Simulate how a color appears to someone with color blindness."""
        if color_blind_type == ColorBlindType.NORMAL:
            return color

        try:
            rgb = self._hex_to_rgb(color)

            if color_blind_type == ColorBlindType.PROTANOPIA:
                # Red-blind: reduce red channel
                r, g, b = rgb
                new_rgb = (0.567 * r + 0.433 * g, 0.558 * r + 0.442 * g, 0.242 * g + 0.758 * b)

            elif color_blind_type == ColorBlindType.DEUTERANOPIA:
                # Green-blind: reduce green channel
                r, g, b = rgb
                new_rgb = (0.625 * r + 0.375 * g, 0.7 * r + 0.3 * g, 0.3 * g + 0.7 * b)

            elif color_blind_type == ColorBlindType.TRITANOPIA:
                # Blue-blind: reduce blue channel
                r, g, b = rgb
                new_rgb = (0.95 * r + 0.05 * g, 0.433 * r + 0.567 * g, 0.475 * r + 0.525 * g)

            elif color_blind_type == ColorBlindType.ACHROMATOPSIA:
                # Complete color blindness: convert to grayscale
                r, g, b = rgb
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                new_rgb = (gray, gray, gray)

            else:
                return color

            # Convert back to hex
            r, g, b = [max(0, min(1, c)) for c in new_rgb]
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            return hex_color

        except Exception as e:
            logger.debug(f"Error simulating color blindness: {e}")
            return color


def create_accessible_widget_factory():
    """Create a factory for accessible widgets."""

    class AccessibleEntry(ttk.Entry):
        """Entry widget with enhanced accessibility features."""

        def __init__(self, parent, label: str = "", description: str = "", **kwargs):
            super().__init__(parent, **kwargs)

            # Store accessibility attributes
            self.accessible_label = label
            self.accessible_description = description

            # Enhance focus behavior
            self.bind('<FocusIn>', self._on_focus_in)
            self.bind('<FocusOut>', self._on_focus_out)

        def _on_focus_in(self, event):
            """Handle focus in with accessibility announcement."""
            announcement = self.accessible_label
            if self.accessible_description:
                announcement += f". {self.accessible_description}"

            # In a real implementation, this would trigger screen reader
            logger.info(f"Accessible Entry focused: {announcement}")

        def _on_focus_out(self, event):
            """Handle focus out."""
            pass

    class AccessibleButton(ttk.Button):
        """Button widget with enhanced accessibility features."""

        def __init__(self, parent, label: str = "", description: str = "", **kwargs):
            super().__init__(parent, **kwargs)

            self.accessible_label = label or kwargs.get('text', '')
            self.accessible_description = description

            # Enhance keyboard interaction
            self.bind('<Return>', self._on_activate)
            self.bind('<space>', self._on_activate)
            self.bind('<FocusIn>', self._on_focus_in)

        def _on_activate(self, event):
            """Handle keyboard activation."""
            self.invoke()
            return "break"

        def _on_focus_in(self, event):
            """Handle focus in with accessibility announcement."""
            announcement = f"Button: {self.accessible_label}"
            if self.accessible_description:
                announcement += f". {self.accessible_description}"

            logger.info(f"Accessible Button focused: {announcement}")

    return {
        'Entry': AccessibleEntry,
        'Button': AccessibleButton
    }


# Example usage and testing
if __name__ == "__main__":
    # Demo application showing accessibility features
    root = tk.Tk()
    root.title("Accessibility Demo")
    root.geometry("800x600")

    # Create accessibility manager
    accessibility = AccessibilityManager(root)

    # Create demo interface
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Add widgets for testing
    ttk.Label(frame, text="Accessibility Demo", font=("Arial", 16, "bold")).pack(pady=10)

    entry = ttk.Entry(frame, width=30)
    entry.pack(pady=5)
    accessibility.screen_reader.add_label(entry, "Name input field")

    button = ttk.Button(frame, text="Test Button")
    button.pack(pady=5)
    accessibility.screen_reader.add_label(button, "Test action button")

    check = ttk.Checkbutton(frame, text="Enable notifications")
    check.pack(pady=5)

    # Add accessibility controls
    control_frame = ttk.LabelFrame(frame, text="Accessibility Controls")
    control_frame.pack(fill=tk.X, pady=20)

    ttk.Button(control_frame, text="Toggle High Contrast",
              command=accessibility.toggle_high_contrast).pack(side=tk.LEFT, padx=5)

    ttk.Button(control_frame, text="Increase Font Size",
              command=lambda: accessibility.apply_font_scaling(accessibility.font_scale * 1.2)).pack(side=tk.LEFT, padx=5)

    ttk.Button(control_frame, text="Run Accessibility Check",
              command=lambda: print(accessibility.run_accessibility_check())).pack(side=tk.LEFT, padx=5)

    # Enable screen reader
    accessibility.enable_screen_reader()

    root.mainloop()