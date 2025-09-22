"""
Smart Navigation Integration for Project QuickNav GUI

This module enhances the existing Tkinter GUI with ML-powered features including:
- Intelligent project recommendations
- Predictive navigation suggestions
- Smart search with auto-completion
- Context-aware shortcuts
- User behavior learning

Features:
- Real-time ML predictions
- Adaptive UI based on usage patterns
- Smart autocomplete with ranking
- Contextual recommendations
- Usage analytics integration
"""

import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import queue
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
from pathlib import Path

# Import core components
try:
    from .recommendation_engine import RecommendationEngine
    from .analytics_dashboard import AnalyticsDashboard
    from .gui_controller import GuiController
    from .gui_theme import ThemeManager
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from recommendation_engine import RecommendationEngine
    from analytics_dashboard import AnalyticsDashboard
    from gui_controller import GuiController
    from gui_theme import ThemeManager

logger = logging.getLogger(__name__)

class SmartAutoComplete:
    """Smart autocomplete with ML-enhanced suggestions"""

    def __init__(self, entry_widget: tk.Entry, recommendation_engine: RecommendationEngine):
        self.entry = entry_widget
        self.recommendation_engine = recommendation_engine
        self.suggestions_listbox = None
        self.suggestions_window = None
        self.current_suggestions = []
        self.suggestion_scores = {}

        # Bind events
        self.entry.bind('<KeyRelease>', self._on_key_release)
        self.entry.bind('<FocusOut>', self._hide_suggestions)
        self.entry.bind('<Tab>', self._on_tab)
        self.entry.bind('<Return>', self._on_return)
        self.entry.bind('<Up>', self._on_up_arrow)
        self.entry.bind('<Down>', self._on_down_arrow)

    def _on_key_release(self, event):
        """Handle key release events for autocomplete"""
        if event.keysym in ['Up', 'Down', 'Tab', 'Return', 'Escape']:
            return

        current_text = self.entry.get().strip()

        if len(current_text) >= 2:  # Start suggesting after 2 characters
            self._update_suggestions(current_text)
        else:
            self._hide_suggestions()

    def _update_suggestions(self, query: str):
        """Update suggestions based on current query"""
        try:
            # Get ML-enhanced suggestions
            suggestions = self._get_smart_suggestions(query)

            if suggestions:
                self._show_suggestions(suggestions)
            else:
                self._hide_suggestions()

        except Exception as e:
            logger.error(f"Error updating suggestions: {e}")
            self._hide_suggestions()

    def _get_smart_suggestions(self, query: str) -> List[Tuple[str, float]]:
        """Get smart suggestions using ML"""
        suggestions = []

        # Get project suggestions from ML engine
        if hasattr(self.recommendation_engine, 'project_features'):
            project_features = self.recommendation_engine.project_features

            for project_id, features in project_features.items():
                metadata = features.get('metadata', {})
                project_name = metadata.get('project_folder', project_id)

                # Simple text matching with scoring
                if query.lower() in project_name.lower():
                    # Calculate relevance score
                    score = self._calculate_relevance_score(query, project_name, metadata)
                    suggestions.append((project_name, score))

        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:10]

    def _calculate_relevance_score(self, query: str, project_name: str, metadata: Dict) -> float:
        """Calculate relevance score for suggestion"""
        score = 0.0

        # Exact match bonus
        if query.lower() == project_name.lower():
            score += 1.0

        # Prefix match bonus
        if project_name.lower().startswith(query.lower()):
            score += 0.8

        # Substring match
        if query.lower() in project_name.lower():
            score += 0.5

        # Recency bonus (if available)
        if 'last_accessed' in metadata:
            try:
                last_accessed = datetime.fromisoformat(metadata['last_accessed'])
                days_ago = (datetime.now() - last_accessed).days
                recency_score = max(0, 1 - days_ago / 30)  # Decay over 30 days
                score += recency_score * 0.3
            except:
                pass

        # Popularity bonus
        if 'access_count' in metadata:
            popularity_score = min(1.0, metadata['access_count'] / 100)
            score += popularity_score * 0.2

        return score

    def _show_suggestions(self, suggestions: List[Tuple[str, float]]):
        """Show suggestions in dropdown"""
        if not suggestions:
            self._hide_suggestions()
            return

        # Create suggestions window if not exists
        if not self.suggestions_window:
            self.suggestions_window = tk.Toplevel(self.entry)
            self.suggestions_window.wm_overrideredirect(True)
            self.suggestions_window.configure(bg='white', relief='solid', bd=1)

            self.suggestions_listbox = tk.Listbox(
                self.suggestions_window,
                height=min(8, len(suggestions)),
                font=('Arial', 9),
                selectmode=tk.SINGLE,
                activestyle='dotbox'
            )
            self.suggestions_listbox.pack()

            # Bind listbox events
            self.suggestions_listbox.bind('<Button-1>', self._on_suggestion_click)
            self.suggestions_listbox.bind('<Return>', self._on_suggestion_select)

        # Position window below entry
        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()
        self.suggestions_window.geometry(f"+{x}+{y}")

        # Clear and populate suggestions
        self.suggestions_listbox.delete(0, tk.END)
        self.current_suggestions = []
        self.suggestion_scores = {}

        for suggestion, score in suggestions:
            self.suggestions_listbox.insert(tk.END, suggestion)
            self.current_suggestions.append(suggestion)
            self.suggestion_scores[suggestion] = score

        # Show window
        self.suggestions_window.deiconify()
        self.suggestions_window.lift()

    def _hide_suggestions(self, event=None):
        """Hide suggestions window"""
        if self.suggestions_window:
            self.suggestions_window.withdraw()

    def _on_suggestion_click(self, event):
        """Handle suggestion click"""
        selection = self.suggestions_listbox.curselection()
        if selection:
            self._select_suggestion(selection[0])

    def _on_suggestion_select(self, event):
        """Handle suggestion selection with Return key"""
        selection = self.suggestions_listbox.curselection()
        if selection:
            self._select_suggestion(selection[0])

    def _select_suggestion(self, index: int):
        """Select a suggestion"""
        if 0 <= index < len(self.current_suggestions):
            suggestion = self.current_suggestions[index]
            self.entry.delete(0, tk.END)
            self.entry.insert(0, suggestion)
            self._hide_suggestions()

            # Trigger selection event
            self.entry.event_generate('<Return>')

    def _on_tab(self, event):
        """Handle Tab key for suggestion completion"""
        if self.suggestions_window and self.suggestions_window.winfo_viewable():
            if self.current_suggestions:
                self._select_suggestion(0)  # Select first suggestion
                return 'break'

    def _on_return(self, event):
        """Handle Return key"""
        if self.suggestions_window and self.suggestions_window.winfo_viewable():
            selection = self.suggestions_listbox.curselection()
            if selection:
                self._select_suggestion(selection[0])
                return 'break'

    def _on_up_arrow(self, event):
        """Handle Up arrow in suggestions"""
        if self.suggestions_window and self.suggestions_window.winfo_viewable():
            current = self.suggestions_listbox.curselection()
            if current:
                index = current[0]
                if index > 0:
                    self.suggestions_listbox.selection_clear(0, tk.END)
                    self.suggestions_listbox.selection_set(index - 1)
                    self.suggestions_listbox.activate(index - 1)
            return 'break'

    def _on_down_arrow(self, event):
        """Handle Down arrow in suggestions"""
        if self.suggestions_window and self.suggestions_window.winfo_viewable():
            current = self.suggestions_listbox.curselection()
            if current:
                index = current[0]
                if index < len(self.current_suggestions) - 1:
                    self.suggestions_listbox.selection_clear(0, tk.END)
                    self.suggestions_listbox.selection_set(index + 1)
                    self.suggestions_listbox.activate(index + 1)
            else:
                # Select first item if none selected
                self.suggestions_listbox.selection_set(0)
                self.suggestions_listbox.activate(0)
            return 'break'

class SmartRecommendationPanel:
    """Smart recommendation panel with context-aware suggestions"""

    def __init__(self, parent, recommendation_engine: RecommendationEngine,
                 theme_manager: ThemeManager):
        self.parent = parent
        self.recommendation_engine = recommendation_engine
        self.theme_manager = theme_manager

        # User context
        self.current_user_id = "default_user"
        self.session_context = {}
        self.last_update = None

        # UI components
        self.frame = None
        self.recommendations_tree = None
        self.context_label = None
        self.refresh_button = None

        # Data
        self.current_recommendations = []
        self.interaction_history = []

        self.create_panel()

    def create_panel(self):
        """Create the recommendation panel UI"""
        # Main frame
        self.frame = ttk.LabelFrame(self.parent, text="Smart Recommendations", padding=10)

        # Header with context info
        header_frame = ttk.Frame(self.frame)
        header_frame.pack(fill="x", pady=(0, 10))

        self.context_label = ttk.Label(header_frame, text="Loading recommendations...",
                                      font=('Arial', 9))
        self.context_label.pack(side="left")

        self.refresh_button = ttk.Button(header_frame, text="Refresh",
                                        command=self.refresh_recommendations,
                                        width=8)
        self.refresh_button.pack(side="right")

        # Recommendations tree
        columns = ("Project", "Score", "Reason")
        self.recommendations_tree = ttk.Treeview(self.frame, columns=columns,
                                               show="headings", height=8)

        # Configure columns
        self.recommendations_tree.heading("Project", text="Project")
        self.recommendations_tree.heading("Score", text="Score")
        self.recommendations_tree.heading("Reason", text="Reason")

        self.recommendations_tree.column("Project", width=150)
        self.recommendations_tree.column("Score", width=60)
        self.recommendations_tree.column("Reason", width=200)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical",
                                 command=self.recommendations_tree.yview)
        self.recommendations_tree.configure(yscrollcommand=scrollbar.set)

        # Pack tree and scrollbar
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill="both", expand=True)

        self.recommendations_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind events
        self.recommendations_tree.bind('<Double-1>', self._on_recommendation_select)

        # Quick actions frame
        actions_frame = ttk.Frame(self.frame)
        actions_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(actions_frame, text="Open Selected",
                  command=self._open_selected).pack(side="left", padx=(0, 5))
        ttk.Button(actions_frame, text="Mark as Used",
                  command=self._mark_as_used).pack(side="left", padx=(0, 5))
        ttk.Button(actions_frame, text="Hide",
                  command=self._hide_selected).pack(side="left")

    def set_user_context(self, user_id: str, context: Dict[str, Any]):
        """Set current user context"""
        self.current_user_id = user_id
        self.session_context = context
        self.refresh_recommendations()

    def refresh_recommendations(self):
        """Refresh recommendations based on current context"""
        def refresh_thread():
            try:
                # Get recommendations from ML engine
                recommendations = self.recommendation_engine.get_recommendations(
                    user_id=self.current_user_id,
                    context=self.session_context,
                    num_recommendations=10
                )

                # Update UI in main thread
                self.parent.after(0, self._update_recommendations_display, recommendations)

            except Exception as e:
                logger.error(f"Error refreshing recommendations: {e}")
                self.parent.after(0, self._show_error, str(e))

        threading.Thread(target=refresh_thread, daemon=True).start()

    def _update_recommendations_display(self, recommendations: List[Dict]):
        """Update recommendations display"""
        # Clear existing items
        for item in self.recommendations_tree.get_children():
            self.recommendations_tree.delete(item)

        self.current_recommendations = recommendations

        # Add new recommendations
        for rec in recommendations:
            score_str = f"{rec['score']:.2f}"
            reason = rec['explanation'][:50] + "..." if len(rec['explanation']) > 50 else rec['explanation']

            self.recommendations_tree.insert('', 'end', values=(
                rec['project_id'],
                score_str,
                reason
            ))

        # Update context label
        context_text = f"Recommendations for {self.current_user_id} ({len(recommendations)} items)"
        if self.session_context:
            context_text += f" | Context: {list(self.session_context.keys())}"

        self.context_label.config(text=context_text)
        self.last_update = datetime.now()

    def _show_error(self, error_message: str):
        """Show error in UI"""
        self.context_label.config(text=f"Error: {error_message}")

    def _on_recommendation_select(self, event):
        """Handle recommendation double-click"""
        self._open_selected()

    def _open_selected(self):
        """Open selected recommendation"""
        selection = self.recommendations_tree.selection()
        if not selection:
            return

        item = self.recommendations_tree.item(selection[0])
        project_id = item['values'][0]

        # Track interaction
        self._track_interaction(project_id, 'open')

        # Trigger navigation event
        if hasattr(self.parent, 'navigate_to_project'):
            self.parent.navigate_to_project(project_id)

    def _mark_as_used(self):
        """Mark selected recommendation as used"""
        selection = self.recommendations_tree.selection()
        if not selection:
            return

        item = self.recommendations_tree.item(selection[0])
        project_id = item['values'][0]

        # Track interaction
        self._track_interaction(project_id, 'used')

        # Update display (could dim the item or move it)
        self.recommendations_tree.set(selection[0], "Reason", "✓ Used")

    def _hide_selected(self):
        """Hide selected recommendation"""
        selection = self.recommendations_tree.selection()
        if not selection:
            return

        item = self.recommendations_tree.item(selection[0])
        project_id = item['values'][0]

        # Track interaction
        self._track_interaction(project_id, 'hidden')

        # Remove from display
        self.recommendations_tree.delete(selection[0])

    def _track_interaction(self, project_id: str, interaction_type: str):
        """Track user interaction with recommendation"""
        interaction = {
            'project_id': project_id,
            'interaction_type': interaction_type,
            'timestamp': datetime.now(),
            'user_id': self.current_user_id,
            'context': self.session_context.copy()
        }

        self.interaction_history.append(interaction)

        # Update recommendation engine
        self.recommendation_engine.update_user_interaction(
            user_id=self.current_user_id,
            project_id=project_id,
            interaction_type=interaction_type,
            metadata=interaction['context']
        )

        logger.info(f"Tracked interaction: {interaction_type} for project {project_id}")

    def get_panel_widget(self) -> ttk.LabelFrame:
        """Get the panel widget for embedding in GUI"""
        return self.frame

    def get_interaction_history(self) -> List[Dict]:
        """Get interaction history for analytics"""
        return self.interaction_history.copy()

class PredictiveNavigationAssistant:
    """Predictive navigation assistant with smart suggestions"""

    def __init__(self, gui_controller: GuiController,
                 recommendation_engine: RecommendationEngine):
        self.gui_controller = gui_controller
        self.recommendation_engine = recommendation_engine

        # Navigation patterns
        self.navigation_history = []
        self.pattern_predictions = {}
        self.context_suggestions = {}

        # UI elements
        self.suggestion_popup = None
        self.current_suggestions = []

    def track_navigation(self, action: str, target: str, context: Dict[str, Any]):
        """Track navigation action for pattern learning"""
        nav_event = {
            'action': action,
            'target': target,
            'context': context,
            'timestamp': datetime.now(),
            'session_id': getattr(self, 'current_session_id', 'default')
        }

        self.navigation_history.append(nav_event)

        # Analyze patterns if we have enough history
        if len(self.navigation_history) >= 5:
            self._analyze_navigation_patterns()

    def get_next_action_prediction(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction for next likely action"""
        return self.recommendation_engine.predict_next_action(
            user_id=getattr(self, 'current_user_id', 'default'),
            current_context=current_context
        )

    def show_smart_suggestions(self, widget: tk.Widget, context: Dict[str, Any]):
        """Show smart suggestions near a widget"""
        prediction = self.get_next_action_prediction(context)

        if prediction['confidence'] > 0.5:  # Only show if confident
            self._create_suggestion_popup(widget, prediction)

    def _analyze_navigation_patterns(self):
        """Analyze navigation patterns for prediction"""
        # Simple pattern analysis - can be enhanced with ML
        recent_actions = [nav['action'] for nav in self.navigation_history[-5:]]

        # Look for sequential patterns
        for i in range(len(recent_actions) - 2):
            pattern = tuple(recent_actions[i:i+3])
            if pattern not in self.pattern_predictions:
                self.pattern_predictions[pattern] = {}

            # If there's a next action, record it
            if i + 3 < len(recent_actions):
                next_action = recent_actions[i + 3]
                self.pattern_predictions[pattern][next_action] = \
                    self.pattern_predictions[pattern].get(next_action, 0) + 1

    def _create_suggestion_popup(self, widget: tk.Widget, prediction: Dict[str, Any]):
        """Create suggestion popup near widget"""
        if self.suggestion_popup:
            self.suggestion_popup.destroy()

        self.suggestion_popup = tk.Toplevel(widget)
        self.suggestion_popup.wm_overrideredirect(True)
        self.suggestion_popup.configure(bg='#ffffcc', relief='solid', bd=1)

        # Position near widget
        x = widget.winfo_rootx() + widget.winfo_width()
        y = widget.winfo_rooty()
        self.suggestion_popup.geometry(f"+{x}+{y}")

        # Content
        content_frame = ttk.Frame(self.suggestion_popup)
        content_frame.pack(padx=5, pady=5)

        ttk.Label(content_frame, text="💡 Smart Suggestion",
                 font=('Arial', 9, 'bold')).pack()

        ttk.Label(content_frame, text=f"Next: {prediction['prediction']}",
                 font=('Arial', 8)).pack()

        ttk.Label(content_frame, text=f"Confidence: {prediction['confidence']:.0%}",
                 font=('Arial', 8)).pack()

        # Suggestions
        for suggestion in prediction.get('suggestions', [])[:2]:
            ttk.Label(content_frame, text=f"• {suggestion}",
                     font=('Arial', 8)).pack(anchor='w')

        # Auto-hide after 5 seconds
        self.suggestion_popup.after(5000, self._hide_suggestion_popup)

    def _hide_suggestion_popup(self):
        """Hide suggestion popup"""
        if self.suggestion_popup:
            self.suggestion_popup.destroy()
            self.suggestion_popup = None

class SmartNavigationIntegration:
    """Main integration class for smart navigation features"""

    def __init__(self, main_gui, gui_controller: GuiController,
                 theme_manager: ThemeManager):
        self.main_gui = main_gui
        self.gui_controller = gui_controller
        self.theme_manager = theme_manager

        # Initialize ML components
        self.recommendation_engine = RecommendationEngine()

        # Initialize smart components
        self.smart_autocomplete = None
        self.recommendation_panel = None
        self.navigation_assistant = None

        # User session
        self.current_session = {
            'user_id': 'default_user',
            'start_time': datetime.now(),
            'context': {}
        }

    def integrate_with_gui(self, project_entry: tk.Entry, main_frame: tk.Widget):
        """Integrate smart features with existing GUI"""
        # Add smart autocomplete to project entry
        self.smart_autocomplete = SmartAutoComplete(project_entry, self.recommendation_engine)

        # Create recommendation panel
        self.recommendation_panel = SmartRecommendationPanel(
            main_frame, self.recommendation_engine, self.theme_manager
        )

        # Initialize navigation assistant
        self.navigation_assistant = PredictiveNavigationAssistant(
            self.gui_controller, self.recommendation_engine
        )

        # Load existing training data
        self._load_training_data()

        logger.info("Smart navigation integration completed")

    def get_recommendation_panel(self) -> ttk.LabelFrame:
        """Get recommendation panel widget"""
        if self.recommendation_panel:
            return self.recommendation_panel.get_panel_widget()
        return None

    def track_user_action(self, action: str, target: str, metadata: Dict[str, Any] = None):
        """Track user action for learning"""
        # Update session context
        self.current_session['context']['last_action'] = action
        self.current_session['context']['last_target'] = target
        self.current_session['context']['timestamp'] = datetime.now()

        # Track in navigation assistant
        if self.navigation_assistant:
            self.navigation_assistant.track_navigation(action, target,
                                                     self.current_session['context'])

        # Update recommendation engine
        if target and action in ['open', 'search', 'navigate']:
            self.recommendation_engine.update_user_interaction(
                user_id=self.current_session['user_id'],
                project_id=target,
                interaction_type=action,
                metadata=metadata or {}
            )

    def get_smart_suggestions(self, query: str) -> List[str]:
        """Get smart suggestions for query"""
        if self.smart_autocomplete:
            suggestions = self.smart_autocomplete._get_smart_suggestions(query)
            return [s[0] for s in suggestions]
        return []

    def show_contextual_help(self, widget: tk.Widget):
        """Show contextual help with smart suggestions"""
        if self.navigation_assistant:
            self.navigation_assistant.show_smart_suggestions(
                widget, self.current_session['context']
            )

    def refresh_recommendations(self):
        """Refresh recommendations panel"""
        if self.recommendation_panel:
            self.recommendation_panel.set_user_context(
                self.current_session['user_id'],
                self.current_session['context']
            )

    def get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        data = {
            'session': self.current_session,
            'recommendation_engine_stats': self.recommendation_engine.get_analytics_insights(),
            'navigation_patterns': len(getattr(self.navigation_assistant, 'pattern_predictions', {})),
            'interaction_history': len(getattr(self.recommendation_panel, 'interaction_history', []))
        }

        return data

    def train_models_with_data(self, training_data: List[Dict], user_history: List[Dict]):
        """Train ML models with provided data"""
        try:
            metrics = self.recommendation_engine.train_models(training_data, user_history)
            logger.info(f"Model training completed: {metrics}")

            # Refresh recommendations after training
            self.refresh_recommendations()

            return metrics
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'error': str(e)}

    def _load_training_data(self):
        """Load existing training data for model initialization"""
        try:
            # Look for training data files
            training_data_dir = Path("training_data")
            if not training_data_dir.exists():
                logger.info("No training data directory found")
                return

            # Load all JSON training files
            training_data = []
            for json_file in training_data_dir.glob("training_data_*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        training_data.extend(data)
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")

            if training_data:
                # Load user history (would come from MCP server or logs)
                user_history = self._load_user_history()

                # Train models
                self.train_models_with_data(training_data, user_history)
                logger.info(f"Loaded and trained with {len(training_data)} documents")
            else:
                logger.info("No training data found")

        except Exception as e:
            logger.error(f"Error loading training data: {e}")

    def _load_user_history(self) -> List[Dict]:
        """Load user history from available sources"""
        # This would integrate with MCP server or other user tracking
        # For now, return minimal sample data
        return [
            {
                'user_id': 'default_user',
                'project_code': '17741',
                'action': 'open',
                'timestamp': datetime.now() - timedelta(hours=1)
            }
        ]


# Example usage and testing
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk

    # Create test GUI
    root = tk.Tk()
    root.title("Smart Navigation Test")
    root.geometry("800x600")

    # Mock components
    class MockController:
        def navigate_to_project(self, project_id):
            print(f"Navigate to project: {project_id}")

    class MockThemeManager:
        def get_color(self, name):
            return "#ffffff"

    # Create main frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Create test entry
    entry_frame = ttk.Frame(main_frame)
    entry_frame.pack(fill="x", pady=10)

    ttk.Label(entry_frame, text="Project Code:").pack(side="left")
    project_entry = ttk.Entry(entry_frame, width=20)
    project_entry.pack(side="left", padx=10)

    # Initialize smart navigation
    controller = MockController()
    theme_manager = MockThemeManager()

    smart_nav = SmartNavigationIntegration(root, controller, theme_manager)
    smart_nav.integrate_with_gui(project_entry, main_frame)

    # Add recommendation panel to GUI
    rec_panel = smart_nav.get_recommendation_panel()
    if rec_panel:
        rec_panel.pack(fill="both", expand=True, pady=10)

    # Test tracking
    def test_action():
        smart_nav.track_user_action("search", "17741", {"test": True})
        smart_nav.refresh_recommendations()

    ttk.Button(main_frame, text="Test Action", command=test_action).pack(pady=10)

    root.mainloop()