"""
AI Chat Widget for Project QuickNav

This module provides a comprehensive chat interface widget that integrates
with the AI client to provide intelligent assistance within the GUI.

Features:
- Modern chat interface with message bubbles
- Syntax highlighting for code and file paths
- Tool execution visualization
- Message history and search
- Export/import conversations
- Typing indicators and status
- Auto-suggestions and quick actions
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import asyncio
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import re
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MessageBubble(ttk.Frame):
    """A message bubble widget for displaying chat messages."""

    def __init__(self, parent, message: Dict[str, Any], theme_manager=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.message = message
        self.theme_manager = theme_manager
        self.role = message.get("role", "user")
        self.content = message.get("content", "")
        self.timestamp = message.get("timestamp", datetime.now().isoformat())
        self.tool_calls = message.get("tool_calls", [])

        self._create_bubble()

    def _create_bubble(self):
        """Create the message bubble."""
        # Configure grid weights
        self.columnconfigure(1, weight=1)

        # Determine bubble side and colors with safe defaults and theme support
        if self.role == "user":
            bubble_side = "right"
            bubble_color = self._get_safe_color("frame", "#0078d4")
            text_color = self._get_safe_color("text_fg", "#ffffff")
        elif self.role == "assistant":
            bubble_side = "left"
            bubble_color = self._get_safe_color("frame", "#f0f0f0")
            text_color = self._get_safe_color("text_fg", "#000000")
        elif self.role == "tool":
            bubble_side = "left"
            bubble_color = self._get_safe_color("frame", "#e8f5e8")
            text_color = self._get_safe_color("text_fg", "#2d5a2d")
        else:
            bubble_side = "left"
            bubble_color = self._get_safe_color("frame", "#fff4e6")
            text_color = self._get_safe_color("text_fg", "#8b4513")

        # Create bubble frame
        bubble_frame = ttk.Frame(self)

        if bubble_side == "right":
            bubble_frame.grid(row=0, column=1, sticky="e", padx=(50, 5), pady=2)
        else:
            bubble_frame.grid(row=0, column=0, sticky="w", padx=(5, 50), pady=2)

        # Create bubble content
        self._create_bubble_content(bubble_frame, bubble_color, text_color)

    def _get_safe_color(self, element: str, default: str) -> str:
        """Get a safe color string, always returning a valid hex color."""
        # First try to get color from theme
        theme_color = self._get_theme_color(element, default)
        
        # Validate that it's a proper hex color
        if self._is_valid_hex_color(theme_color):
            return theme_color
        
        # Fallback to default if theme color is invalid
        return default

    def _is_valid_hex_color(self, color: str) -> bool:
        """Validate if a string is a valid hex color."""
        if not isinstance(color, str):
            return False
        
        # Check if it's a valid hex color pattern
        if re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color):
            return True
        
        # Check if it's a valid named color (basic ones)
        named_colors = {
            'black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
            'gray', 'grey', 'darkgray', 'darkgrey', 'lightgray', 'lightgrey'
        }
        return color.lower() in named_colors

    def _get_theme_color(self, element: str, default: str) -> str:
        """Get color from theme manager or use default."""
        try:
            if self.theme_manager and hasattr(self.theme_manager, 'get_current_theme'):
                theme = self.theme_manager.get_current_theme()
                if theme and hasattr(theme, 'get_color'):
                    color = theme.get_color(element)
                    # Handle dict color responses
                    if isinstance(color, dict):
                        # Try different keys to extract the color
                        if "bg" in color:
                            return str(color["bg"])
                        elif "normal" in color and isinstance(color["normal"], dict):
                            return str(color["normal"].get("bg", default))
                        elif "normal" in color:
                            return str(color["normal"])
                        else:
                            # Return first value if available
                            for key, value in color.items():
                                if isinstance(value, str) and value.startswith("#"):
                                    return value
                    elif isinstance(color, str) and color.startswith("#"):
                        return color
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # Log specific theme access errors for debugging
            logger.debug(f"Theme color access failed for element '{element}': {e}")
            pass
        return default

    def _create_bubble_content(self, parent, bg_color: str, text_color: str):
        """Create the content of the message bubble with modern styling."""
        # Main content frame with improved styling
        content_frame = tk.Frame(parent, bg=bg_color, relief=tk.FLAT, bd=0)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Add subtle border radius effect with padding
        content_frame.configure(highlightbackground="#e1dfdd", highlightthickness=1)

        # Role indicator with icons
        role_icons = {
            "assistant": "🤖",
            "tool": "🔧",
            "system": "ℹ️"
        }

        if self.role != "user":
            icon = role_icons.get(self.role, "")
            role_text = f"{icon} {self.role.title()}" if icon else self.role.title()

            role_label = tk.Label(
                content_frame,
                text=role_text,
                bg=bg_color,
                fg=text_color,
                font=("Arial", 8, "italic"),
                anchor="w"
            )
            role_label.pack(fill=tk.X, padx=5, pady=(2, 0))

        # Message content
        if self.content:
            self._create_message_content(content_frame, bg_color, text_color)

        # Tool calls
        if self.tool_calls:
            self._create_tool_calls(content_frame, bg_color, text_color)

        # Timestamp
        time_str = self._format_timestamp()
        time_label = tk.Label(
            content_frame,
            text=time_str,
            bg=bg_color,
            fg=text_color,
            font=("Arial", 7),
            anchor="e"
        )
        time_label.pack(fill=tk.X, padx=5, pady=(2, 2))

    def _create_message_content(self, parent, bg_color: str, text_color: str):
        """Create the main message content."""
        # Detect if content contains code, file paths, or structured data
        if self._is_structured_content():
            self._create_structured_content(parent, bg_color, text_color)
        else:
            self._create_text_content(parent, bg_color, text_color)

    def _is_structured_content(self) -> bool:
        """Check if content contains structured data."""
        content = self.content.lower()
        return (
            "```" in self.content or
            self.content.startswith("{") or
            self.content.startswith("[") or
            "error:" in content or
            "success:" in content or
            re.search(r'[a-z]:\\', self.content, re.IGNORECASE) or  # Windows paths
            re.search(r'/[a-z]', self.content)  # Unix paths
        )

    def _create_text_content(self, parent, bg_color: str, text_color: str):
        """Create simple text content."""
        text_widget = tk.Text(
            parent,
            bg=bg_color,
            fg=text_color,
            font=("Arial", 10),
            wrap=tk.WORD,
            height=self._calculate_text_height(),
            state=tk.DISABLED,
            relief=tk.FLAT,
            cursor="arrow"
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Insert content
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, self.content)
        text_widget.config(state=tk.DISABLED)

        # Make links clickable
        self._make_links_clickable(text_widget)

    def _create_structured_content(self, parent, bg_color: str, text_color: str):
        """Create structured content with syntax highlighting."""
        text_widget = tk.Text(
            parent,
            bg=bg_color,
            fg=text_color,
            font=("Consolas", 9),
            wrap=tk.WORD,
            height=self._calculate_text_height(),
            state=tk.DISABLED,
            relief=tk.FLAT,
            cursor="arrow"
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Configure tags for syntax highlighting
        text_widget.tag_configure("code", font=("Consolas", 9), background="#f8f8f8")
        text_widget.tag_configure("error", foreground="#d32f2f")
        text_widget.tag_configure("success", foreground="#388e3c")
        text_widget.tag_configure("path", foreground="#1976d2", underline=True)
        text_widget.tag_configure("json", foreground="#795548")

        # Insert and highlight content
        text_widget.config(state=tk.NORMAL)
        self._insert_highlighted_content(text_widget)
        text_widget.config(state=tk.DISABLED)

        # Make paths clickable
        self._make_paths_clickable(text_widget)

    def _insert_highlighted_content(self, text_widget):
        """Insert content with syntax highlighting."""
        content = self.content

        # Handle code blocks
        if "```" in content:
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Regular text
                    text_widget.insert(tk.END, part)
                else:
                    # Code block
                    text_widget.insert(tk.END, part, "code")
        else:
            # Highlight specific patterns
            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()

                if line_lower.startswith('error:'):
                    text_widget.insert(tk.END, line + '\n', "error")
                elif line_lower.startswith('success:'):
                    text_widget.insert(tk.END, line + '\n', "success")
                elif re.match(r'^[{[]', line.strip()):
                    text_widget.insert(tk.END, line + '\n', "json")
                elif re.search(r'[a-z]:\\|/[a-z]', line, re.IGNORECASE):
                    # Line contains file paths
                    self._insert_line_with_paths(text_widget, line + '\n')
                else:
                    text_widget.insert(tk.END, line + '\n')

    def _insert_line_with_paths(self, text_widget, line):
        """Insert a line with file path highlighting."""
        # Find file paths in the line
        path_pattern = r'([a-zA-Z]:\\[^\s]+|/[^\s]+)'
        parts = re.split(path_pattern, line)

        for part in parts:
            if re.match(path_pattern, part):
                text_widget.insert(tk.END, part, "path")
            else:
                text_widget.insert(tk.END, part)

    def _calculate_text_height(self) -> int:
        """Calculate appropriate height for text widget."""
        lines = self.content.count('\n') + 1
        # Minimum 1 line, maximum 10 lines
        return min(max(lines, 1), 10)

    def _make_links_clickable(self, text_widget):
        """Make URLs in text clickable."""
        url_pattern = r'https?://[^\s]+'
        content = text_widget.get(1.0, tk.END)

        for match in re.finditer(url_pattern, content):
            start_idx = f"1.0+{match.start()}c"
            end_idx = f"1.0+{match.end()}c"

            text_widget.tag_add("url", start_idx, end_idx)
            text_widget.tag_configure("url", foreground="blue", underline=True)
            text_widget.tag_bind("url", "<Button-1>",
                               lambda e, url=match.group(): self._open_url(url))

    def _make_paths_clickable(self, text_widget):
        """Make file paths clickable."""
        def open_path(path):
            """Open file path in explorer."""
            try:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.startfile(path)
                    else:
                        os.startfile(path)
                else:
                    messagebox.showwarning("Path Not Found", f"Path does not exist:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open path:\n{e}")

        text_widget.tag_bind("path", "<Button-1>",
                           lambda e: self._handle_path_click(text_widget, e))

    def _handle_path_click(self, text_widget, event):
        """Handle clicking on a file path."""
        # Get the clicked position
        index = text_widget.index(f"@{event.x},{event.y}")

        # Find the path at this position
        tags = text_widget.tag_names(index)
        if "path" in tags:
            # Get the range of the path tag
            range_start = text_widget.tag_prevrange("path", index + "+1c")
            if range_start:
                start_idx, end_idx = range_start
                path = text_widget.get(start_idx, end_idx)
                self._open_path(path.strip())

    def _open_url(self, url: str):
        """Open URL in browser."""
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open URL:\n{e}")

    def _open_path(self, path: str):
        """Open file path."""
        try:
            if os.path.exists(path):
                if os.name == 'nt':  # Windows
                    os.startfile(path)
                elif os.name == 'posix':  # macOS and Linux
                    if os.uname().sysname == 'Darwin':  # macOS
                        os.system(f'open "{path}"')
                    else:  # Linux
                        os.system(f'xdg-open "{path}"')
            else:
                messagebox.showwarning("Path Not Found", f"Path does not exist:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open path:\n{e}")

    def _create_tool_calls(self, parent, bg_color: str, text_color: str):
        """Create tool call visualization."""
        tool_frame = tk.Frame(parent, bg=bg_color)
        tool_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(
            tool_frame,
            text="🔧 Tool Calls:",
            bg=bg_color,
            fg=text_color,
            font=("Arial", 9, "bold")
        ).pack(anchor="w")

        for tool_call in self.tool_calls:
            function_name = tool_call.get("function", {}).get("name", "Unknown")
            arguments = tool_call.get("function", {}).get("arguments", "{}")

            tool_info = tk.Label(
                tool_frame,
                text=f"• {function_name}({arguments})",
                bg=bg_color,
                fg=text_color,
                font=("Consolas", 8),
                anchor="w",
                justify="left"
            )
            tool_info.pack(fill=tk.X, padx=10)

    def _format_timestamp(self) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%H:%M")
        except:
            return ""


class TypingIndicator(ttk.Frame):
    """Enhanced typing indicator widget with modern animation."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.is_typing = False
        self.animation_job = None

        # Create enhanced typing indicator with icon
        self.label = ttk.Label(self, text="🤖 AI is thinking", font=("Segoe UI", 9))
        self.label.pack(side=tk.LEFT, padx=8)

        self.dots = ttk.Label(self, text="", font=("Segoe UI", 12))
        self.dots.pack(side=tk.LEFT)

        self.dot_count = 0

    def start_typing(self):
        """Start the typing animation."""
        self.is_typing = True
        self._animate_dots()
        self.pack(fill=tk.X, padx=5, pady=2)

    def stop_typing(self):
        """Stop the typing animation."""
        self.is_typing = False
        if self.animation_job:
            self.after_cancel(self.animation_job)
        self.pack_forget()

    def _animate_dots(self):
        """Animate the typing dots with modern animation."""
        if not self.is_typing:
            return

        # Use different animation styles
        animation_frames = ["", "•", "••", "•••"]
        self.dot_count = (self.dot_count + 1) % len(animation_frames)
        self.dots.config(text=animation_frames[self.dot_count])

        self.animation_job = self.after(400, self._animate_dots)


class ChatWidget(ttk.Frame):
    """Main chat widget for AI interaction."""

    def __init__(self, parent, ai_client=None, theme_manager=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.ai_client = ai_client
        self.theme_manager = theme_manager
        self.messages = []
        self.is_processing = False

        self._create_widget()

    def _create_widget(self):
        """Create the chat widget with enhanced modern styling."""
        # Configure grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create main container with improved styling
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Create header with modern design
        self._create_header(main_frame)

        # Create chat area with better styling
        self._create_chat_area(main_frame)

        # Create input area with enhanced UX
        self._create_input_area(main_frame)

        # Create status bar with improved design
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create the chat header."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        header_frame.columnconfigure(1, weight=1)

        # AI status indicator
        self.status_indicator = tk.Label(
            header_frame,
            text="🤖",
            font=("Arial", 16),
            fg="green" if self.ai_client and self.ai_client.is_enabled() else "gray"
        )
        self.status_indicator.grid(row=0, column=0, padx=(0, 5))

        # Title
        title_label = ttk.Label(
            header_frame,
            text="QuickNav AI Assistant",
            font=("Arial", 12, "bold")
        )
        title_label.grid(row=0, column=1, sticky="w")

        # Control buttons
        btn_frame = ttk.Frame(header_frame)
        btn_frame.grid(row=0, column=2, sticky="e")

        ttk.Button(
            btn_frame,
            text="Clear",
            width=8,
            command=self.clear_chat
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame,
            text="Export",
            width=8,
            command=self.export_chat
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame,
            text="Settings",
            width=8,
            command=self.show_ai_settings
        ).pack(side=tk.LEFT, padx=2)

    def _create_chat_area(self, parent):
        """Create the main chat area."""
        chat_frame = ttk.Frame(parent)
        chat_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        # Create scrollable chat area
        self.chat_canvas = tk.Canvas(chat_frame, bg="white")
        self.chat_scrollbar = ttk.Scrollbar(
            chat_frame,
            orient="vertical",
            command=self.chat_canvas.yview
        )
        self.chat_scrollable_frame = ttk.Frame(self.chat_canvas)

        self.chat_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )

        self.chat_canvas.create_window((0, 0), window=self.chat_scrollable_frame, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind mouse wheel
        self.chat_canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Configure scrollable frame
        self.chat_scrollable_frame.columnconfigure(0, weight=1)

        # Add typing indicator
        self.typing_indicator = TypingIndicator(self.chat_scrollable_frame)

        # Add welcome message
        self._add_welcome_message()

    def _create_input_area(self, parent):
        """Create the message input area."""
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        input_frame.columnconfigure(0, weight=1)

        # Quick actions row
        quick_frame = ttk.Frame(input_frame)
        quick_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(quick_frame, text="Quick actions:").pack(side=tk.LEFT)

        quick_actions = [
            ("Find project", "Find project 17741"),
            ("List recent", "Show me my recent projects"),
            ("Search docs", "Find LLD documents in project"),
            ("Help", "What can you help me with?")
        ]

        for text, command in quick_actions:
            btn = ttk.Button(
                quick_frame,
                text=text,
                width=12,
                command=lambda cmd=command: self._insert_quick_action(cmd)
            )
            btn.pack(side=tk.LEFT, padx=2)

        # Message input row
        message_frame = ttk.Frame(input_frame)
        message_frame.grid(row=1, column=0, sticky="ew")
        message_frame.columnconfigure(0, weight=1)

        # Input text widget
        self.input_text = tk.Text(
            message_frame,
            height=3,
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.input_text.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        # Input scrollbar
        input_scrollbar = ttk.Scrollbar(
            message_frame,
            orient="vertical",
            command=self.input_text.yview
        )
        input_scrollbar.grid(row=0, column=1, sticky="ns")
        self.input_text.configure(yscrollcommand=input_scrollbar.set)

        # Send button
        self.send_button = ttk.Button(
            message_frame,
            text="Send",
            width=8,
            command=self.send_message
        )
        self.send_button.grid(row=0, column=2, padx=(5, 0), sticky="ns")

        # Bind enter key
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())
        self.input_text.bind("<Shift-Return>", lambda e: None)  # Allow shift+enter for new lines

        # Placeholder text
        self._add_input_placeholder()

    def _create_status_bar(self, parent):
        """Create the status bar."""
        self.status_frame = ttk.Frame(parent)
        self.status_frame.grid(row=3, column=0, sticky="ew")
        self.status_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready - Ask me anything about your projects!",
            font=("Arial", 9)
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        # Model indicator
        if self.ai_client and self.ai_client.is_enabled():
            model_name = getattr(self.ai_client, 'default_model', 'Unknown')
            self.model_label = ttk.Label(
                self.status_frame,
                text=f"Model: {model_name}",
                font=("Arial", 8),
                foreground="gray"
            )
            self.model_label.grid(row=0, column=1, sticky="e")

    def _add_welcome_message(self):
        """Add welcome message to chat."""
        welcome_msg = {
            "role": "assistant",
            "content": """👋 Welcome to QuickNav AI Assistant!

I can help you with:
• Finding project folders by number or name
• Locating specific documents within projects
• Analyzing project structures and contents
• Providing project navigation suggestions
• Answering questions about your projects

Try asking me something like:
• "Find project 17741"
• "Show me recent projects"
• "Find LLD documents in project Test Project"
• "What's in project 10123?"

How can I assist you today?""",
            "timestamp": datetime.now().isoformat()
        }

        self._add_message_bubble(welcome_msg)

    def _add_input_placeholder(self):
        """Add placeholder text to input."""
        placeholder_text = "Type your message here... (Ctrl+Enter to send)"

        def on_focus_in(event):
            if self.input_text.get(1.0, tk.END).strip() == placeholder_text:
                self.input_text.delete(1.0, tk.END)
                self.input_text.config(fg="black")

        def on_focus_out(event):
            if not self.input_text.get(1.0, tk.END).strip():
                self.input_text.insert(1.0, placeholder_text)
                self.input_text.config(fg="gray")

        self.input_text.insert(1.0, placeholder_text)
        self.input_text.config(fg="gray")
        self.input_text.bind("<FocusIn>", on_focus_in)
        self.input_text.bind("<FocusOut>", on_focus_out)

    def _insert_quick_action(self, command: str):
        """Insert a quick action command into input."""
        # Clear placeholder if present
        current_text = self.input_text.get(1.0, tk.END).strip()
        if current_text in ["", "Type your message here... (Ctrl+Enter to send)"]:
            self.input_text.delete(1.0, tk.END)
            self.input_text.config(fg="black")

        # Insert command
        self.input_text.insert(tk.END, command)
        self.input_text.focus_set()

    def _add_message_bubble(self, message: Dict[str, Any]):
        """Add a message bubble to the chat."""
        bubble = MessageBubble(
            self.chat_scrollable_frame,
            message,
            self.theme_manager
        )

        # Grid the bubble
        row = len(self.messages)
        bubble.grid(row=row, column=0, sticky="ew", pady=2)

        # Configure column weight
        self.chat_scrollable_frame.columnconfigure(0, weight=1)

        # Store message
        self.messages.append(message)

        # Scroll to bottom
        self.after(100, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """Scroll chat to bottom."""
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def send_message(self):
        """Send a message to the AI."""
        if self.is_processing:
            return

        # Get message text
        message_text = self.input_text.get(1.0, tk.END).strip()

        # Check for placeholder text
        if message_text in ["", "Type your message here... (Ctrl+Enter to send)"]:
            return

        # Clear input
        self.input_text.delete(1.0, tk.END)
        self.input_text.config(fg="black")

        # Add user message
        user_message = {
            "role": "user",
            "content": message_text,
            "timestamp": datetime.now().isoformat()
        }
        self._add_message_bubble(user_message)

        # Check if AI is available
        if not self.ai_client or not self.ai_client.is_enabled():
            error_message = {
                "role": "assistant",
                "content": "❌ AI features are not available. Please install LiteLLM and configure an API key to enable AI assistance.",
                "timestamp": datetime.now().isoformat()
            }
            self._add_message_bubble(error_message)
            return

        # Start processing
        self.is_processing = True
        self.send_button.config(state="disabled")
        self.status_label.config(text="AI is thinking...")
        self.typing_indicator.start_typing()

        # Process message in background thread
        thread = threading.Thread(
            target=self._process_message_async,
            args=(message_text,),
            daemon=True
        )
        thread.start()

    def _process_message_async(self, message: str):
        """Process message asynchronously."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Get AI response
            response = loop.run_until_complete(self.ai_client.chat(message))

            # Add response to GUI (must be done in main thread)
            self.after(0, lambda: self._handle_ai_response(response))

        except Exception as e:
            logger.exception("Error processing AI message")
            error_msg = f"❌ Error: {str(e)}"
            self.after(0, lambda: self._handle_ai_response(error_msg))

        finally:
            # Clean up
            self.after(0, self._finish_processing)

    def _handle_ai_response(self, response: str):
        """Handle AI response in main thread."""
        ai_message = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        }
        self._add_message_bubble(ai_message)

    def _finish_processing(self):
        """Finish processing and reset UI."""
        self.is_processing = False
        self.send_button.config(state="normal")
        self.status_label.config(text="Ready - Ask me anything about your projects!")
        self.typing_indicator.stop_typing()

    def clear_chat(self):
        """Clear the chat history."""
        result = messagebox.askyesno(
            "Clear Chat",
            "This will clear all chat history. Are you sure?"
        )

        if result:
            # Clear messages
            self.messages = []

            # Clear chat area
            for widget in self.chat_scrollable_frame.winfo_children():
                if not isinstance(widget, TypingIndicator):
                    widget.destroy()

            # Clear AI conversation
            if self.ai_client:
                self.ai_client.clear_conversation()

            # Add welcome message
            self._add_welcome_message()

    def export_chat(self):
        """Export chat history to file."""
        if not self.messages:
            messagebox.showinfo("Export Chat", "No messages to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Chat History",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.messages, f, indent=2, ensure_ascii=False)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for msg in self.messages:
                            timestamp = msg.get('timestamp', '')
                            role = msg.get('role', '').title()
                            content = msg.get('content', '')
                            f.write(f"[{timestamp}] {role}: {content}\n\n")

                messagebox.showinfo("Export Complete", f"Chat history exported to:\n{file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export chat:\n{e}")

    def show_ai_settings(self):
        """Show AI settings dialog."""
        # This will be implemented as part of the main settings
        messagebox.showinfo(
            "AI Settings",
            "AI settings can be configured in the main Settings dialog."
        )

    def set_ai_client(self, ai_client):
        """Set the AI client."""
        self.ai_client = ai_client

        # Update status indicator
        if hasattr(self, 'status_indicator'):
            self.status_indicator.config(
                fg="green" if ai_client and ai_client.is_enabled() else "gray"
            )

        # Update model label
        if hasattr(self, 'model_label') and ai_client and ai_client.is_enabled():
            model_name = getattr(ai_client, 'default_model', 'Unknown')
            self.model_label.config(text=f"Model: {model_name}")

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history."""
        return self.messages.copy()