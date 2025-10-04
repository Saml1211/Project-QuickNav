#!/usr/bin/env python3
"""
AI Assistant Section Module

Handles the AI assistant toolbar UI section.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class AIAssistantSection:
    """
    AI Assistant toolbar UI component.

    Responsibilities:
    - AI enable/disable toggle
    - AI chat panel toggle
    - AI settings access
    """

    def __init__(self, parent, state, layout, tooltip_callback: Optional[Callable] = None):
        """
        Initialize the AI assistant section.

        Args:
            parent: Parent widget (sidebar_frame)
            state: GUIState instance
            layout: LayoutManager instance
            tooltip_callback: Optional callback for adding tooltips
        """
        self.parent = parent
        self.state = state
        self.layout = layout
        self.add_tooltip = tooltip_callback

        # UI elements
        self.frame = None
        self.ai_toggle_button = None
        self.ai_chat_button = None
        self.ai_settings_button = None

        # Callbacks
        self.on_toggle_ai = None
        self.on_toggle_chat = None
        self.on_show_settings = None

        # Create the UI
        self._create_ui()

        logger.info("AIAssistantSection initialized")

    def _create_ui(self):
        """Create the AI assistant section UI."""
        # AI Toolbar frame with improved styling
        self.frame = ttk.LabelFrame(self.parent, text="AI Assistant")
        self.frame.grid(
            row=0, column=1, sticky="ew",
            padx=(self.layout.get_consistent_spacing() // 2, 0)
        )
        self.frame.columnconfigure(0, weight=1)

        # AI controls container with grid layout
        ai_container = ttk.Frame(self.frame)
        ai_container.grid(
            row=0, column=0, sticky="ew",
            padx=self.layout.get_consistent_padding(),
            pady=self.layout.get_consistent_padding()
        )
        ai_container.columnconfigure(0, weight=1)

        # AI toggle button with primary styling and icon
        self.ai_toggle_button = ttk.Button(
            ai_container,
            text="🤖 Enable AI",
            command=self._on_toggle_ai,
            width=14
        )
        self.ai_toggle_button.grid(
            row=0, column=0, sticky="w",
            padx=(0, self.layout.get_consistent_spacing() // 2)
        )

        if self.add_tooltip:
            self.add_tooltip(self.ai_toggle_button, "Enable/disable AI assistant features")

        # AI chat panel button (hidden by default)
        self.ai_chat_button = ttk.Button(
            ai_container,
            text="💬 Chat",
            command=self._on_toggle_chat,
            width=10
        )
        self.ai_chat_button.grid(row=0, column=1, sticky="w")
        self.ai_chat_button.grid_remove()  # Hidden until AI is enabled

        if self.add_tooltip:
            self.add_tooltip(self.ai_chat_button, "Open AI chat assistant panel")

    def set_on_toggle_ai(self, callback: Callable):
        """Set callback for AI toggle."""
        self.on_toggle_ai = callback

    def set_on_toggle_chat(self, callback: Callable):
        """Set callback for chat panel toggle."""
        self.on_toggle_chat = callback

    def set_on_show_settings(self, callback: Callable):
        """Set callback for showing AI settings."""
        self.on_show_settings = callback

    def _on_toggle_ai(self):
        """Handle AI toggle button click."""
        if self.on_toggle_ai:
            try:
                self.on_toggle_ai()
            except Exception as e:
                logger.error(f"Error in AI toggle callback: {e}")

    def _on_toggle_chat(self):
        """Handle chat toggle button click."""
        if self.on_toggle_chat:
            try:
                self.on_toggle_chat()
            except Exception as e:
                logger.error(f"Error in chat toggle callback: {e}")

    def update_ui(self, ai_enabled: bool):
        """
        Update UI based on AI enabled state.

        Args:
            ai_enabled: Whether AI is enabled
        """
        if ai_enabled:
            self.ai_toggle_button.config(text="🤖 Disable AI")
            self.ai_chat_button.grid()
        else:
            self.ai_toggle_button.config(text="🤖 Enable AI")
            self.ai_chat_button.grid_remove()

    def is_ai_enabled(self) -> bool:
        """Check if AI is enabled."""
        return self.state.is_ai_enabled()

    def get_widget(self):
        """Get the main frame widget."""
        return self.frame
