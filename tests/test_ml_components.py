"""
Comprehensive Test Suite for ML Components in Project QuickNav

This test suite covers all machine learning and data-driven features including:
- Recommendation engine functionality
- Analytics dashboard components
- Data ingestion pipeline
- Smart navigation features
- Integration testing

Tests include unit tests, integration tests, and performance benchmarks.
"""

import unittest
import tempfile
import shutil
import json
import time
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Import components to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'quicknav'))

try:
    from src.ml.recommendation_engine import RecommendationEngine
    from src.data.ingestion_pipeline import DataIngestionPipeline, DocumentEvent, ProcessingResult
    from quicknav.analytics_dashboard import AnalyticsDashboard
    from quicknav.smart_navigation import SmartNavigationIntegration, SmartAutoComplete
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all components are properly installed")
    sys.exit(1)

class TestRecommendationEngine(unittest.TestCase):
    """Test cases for the ML recommendation engine"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = RecommendationEngine(data_dir=self.temp_dir)

        # Sample training data
        self.sample_training_data = [
            {
                'project_folder': '17741 - QPS MTR RM Upgrades',
                'document_name': 'Project Handover Document.pdf',
                'document_path': '/path/to/doc1.pdf',
                'extracted_info': {'type': 'handover', 'priority': 'high'}
            },
            {
                'project_folder': '17742 - Conference Room Setup',
                'document_name': 'System Design.pdf',
                'document_path': '/path/to/doc2.pdf',
                'extracted_info': {'type': 'design', 'category': 'audiovisual'}
            },
            {
                'project_folder': '18810 - Network Infrastructure',
                'document_name': 'Technical Specifications.pdf',
                'document_path': '/path/to/doc3.pdf',
                'extracted_info': {'type': 'technical', 'category': 'network'}
            }
        ]

        # Sample user history
        self.sample_user_history = [
            {
                'user_id': 'test_user1',
                'project_code': '17741',
                'action': 'open',
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'user_id': 'test_user1',
                'project_code': '17742',
                'action': 'search',
                'timestamp': datetime.now() - timedelta(hours=1)
            },
            {
                'user_id': 'test_user2',
                'project_code': '17741',
                'action': 'open',
                'timestamp': datetime.now() - timedelta(minutes=30)
            }
        ]

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_engine_initialization(self):
        """Test recommendation engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertIsInstance(self.engine.config, dict)
        self.assertEqual(self.engine.data_dir, Path(self.temp_dir))

    def test_model_training(self):
        """Test ML model training"""
        # Train models
        metrics = self.engine.train_models(
            self.sample_training_data,
            self.sample_user_history
        )

        # Verify training metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('training_time', metrics)
        self.assertIn('content_based', metrics)
        self.assertIn('collaborative', metrics)
        self.assertGreater(metrics['training_time'], 0)

        # Verify models are created
        self.assertIsNotNone(self.engine.tfidf_vectorizer)
        self.assertIsNotNone(self.engine.content_similarity_matrix)

    def test_recommendations_generation(self):
        """Test recommendation generation"""
        # Train models first
        self.engine.train_models(self.sample_training_data, self.sample_user_history)

        # Get recommendations
        recommendations = self.engine.get_recommendations(
            user_id='test_user1',
            num_recommendations=5
        )

        # Verify recommendations
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)

        if recommendations:
            rec = recommendations[0]
            self.assertIn('project_id', rec)
            self.assertIn('score', rec)
            self.assertIn('rank', rec)
            self.assertIn('explanation', rec)
            self.assertIn('confidence', rec)

    def test_user_interaction_tracking(self):
        """Test user interaction tracking"""
        user_id = 'test_user'
        project_id = '17741'

        # Track interaction
        self.engine.update_user_interaction(
            user_id=user_id,
            project_id=project_id,
            interaction_type='open',
            metadata={'source': 'test'}
        )

        # Verify user profile was created
        self.assertIn(user_id, self.engine.user_profiles)
        profile = self.engine.user_profiles[user_id]
        self.assertIn('interactions', profile)
        self.assertGreater(len(profile['interactions']), 0)

        # Check interaction details
        interaction = profile['interactions'][0]
        self.assertEqual(interaction['project_id'], project_id)
        self.assertEqual(interaction['interaction_type'], 'open')

    def test_similar_projects(self):
        """Test similar projects functionality"""
        # Train models first
        self.engine.train_models(self.sample_training_data, self.sample_user_history)

        # Get similar projects
        similar = self.engine.get_similar_projects('17741 - QPS MTR RM Upgrades', num_similar=3)

        self.assertIsInstance(similar, list)
        for item in similar:
            self.assertIn('project_id', item)
            self.assertIn('similarity_score', item)

    def test_next_action_prediction(self):
        """Test next action prediction"""
        user_id = 'test_user'

        # Add some interaction history
        for i in range(5):
            self.engine.update_user_interaction(
                user_id=user_id,
                project_id=f'1774{i}',
                interaction_type='search' if i % 2 == 0 else 'open'
            )

        # Get prediction
        prediction = self.engine.predict_next_action(user_id, {'current_time': datetime.now()})

        self.assertIsInstance(prediction, dict)
        self.assertIn('prediction', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('suggestions', prediction)

    def test_analytics_insights(self):
        """Test analytics insights generation"""
        # Add some data
        self.engine.train_models(self.sample_training_data, self.sample_user_history)

        # Get insights
        insights = self.engine.get_analytics_insights()

        self.assertIsInstance(insights, dict)
        self.assertIn('model_status', insights)
        self.assertIn('user_engagement', insights)
        self.assertIn('recommendation_performance', insights)

    def test_model_persistence(self):
        """Test model saving and loading"""
        # Train models
        self.engine.train_models(self.sample_training_data, self.sample_user_history)

        # Create new engine instance
        new_engine = RecommendationEngine(data_dir=self.temp_dir)

        # Verify models were loaded
        self.assertIsNotNone(new_engine.tfidf_vectorizer)
        self.assertEqual(len(new_engine.project_features), len(self.engine.project_features))

class TestDataIngestionPipeline(unittest.TestCase):
    """Test cases for data ingestion pipeline"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'watch_directories': [self.temp_dir],
            'max_workers': 2,
            'batch_size': 5,
            'enable_monitoring': False,  # Disable for testing
            'db_path': os.path.join(self.temp_dir, 'test.db')
        }
        self.pipeline = DataIngestionPipeline(self.test_config)

        # Create test files
        self.test_files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f'test_document_{i}.pdf')
            with open(test_file, 'w') as f:
                f.write(f'Test content {i}')
            self.test_files.append(test_file)

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.config['max_workers'], 2)
        self.assertTrue(os.path.exists(self.pipeline.db_path))

    def test_file_processing_queue(self):
        """Test file queuing for processing"""
        test_file = self.test_files[0]

        # Queue file for processing
        self.pipeline.queue_file_for_processing(test_file, 'test')

        # Verify queue is not empty
        self.assertGreater(self.pipeline.processing_queue.qsize(), 0)

    def test_document_processing(self):
        """Test individual document processing"""
        test_file = self.test_files[0]

        # Create document event
        event = DocumentEvent(
            event_type='test',
            file_path=test_file,
            project_id='17741',
            timestamp=datetime.now(),
            metadata={'test': True},
            checksum='test_checksum',
            size=100
        )

        # Process document
        result = self.pipeline._process_document(event)

        # Verify result
        self.assertIsInstance(result, ProcessingResult)
        self.assertTrue(result.success)
        self.assertEqual(result.document_path, test_file)
        self.assertGreater(result.processing_time, 0)

    def test_batch_processing(self):
        """Test batch processing of existing documents"""
        # Start pipeline
        self.pipeline.start()

        # Process existing documents
        results = self.pipeline.process_existing_documents([self.temp_dir])

        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn('total_files', results)
        self.assertIn('processed_files', results)
        self.assertGreater(results['total_files'], 0)

        # Stop pipeline
        self.pipeline.stop()

    def test_database_tracking(self):
        """Test database tracking functionality"""
        # Process a file
        test_file = self.test_files[0]
        self.pipeline.queue_file_for_processing(test_file, 'test')

        # Start and stop to process queue
        self.pipeline.start()
        time.sleep(1)  # Give time to process
        self.pipeline.stop()

        # Check database
        with sqlite3.connect(self.pipeline.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM processing_events')
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)

    def test_processing_stats(self):
        """Test processing statistics"""
        stats = self.pipeline.get_processing_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn('total_processed', stats)
        self.assertIn('successful', stats)
        self.assertIn('failed', stats)
        self.assertIn('uptime_seconds', stats)

    def test_project_summary(self):
        """Test project summary functionality"""
        # Mock some data in database
        with sqlite3.connect(self.pipeline.db_path) as conn:
            conn.execute('''
                INSERT INTO processing_events
                (event_type, file_path, timestamp, project_id, success, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ('test', '/test/path', datetime.now(), '17741', True, 0.1))
            conn.commit()

        # Get project summary
        summary = self.pipeline.get_project_summary('17741')

        self.assertIsInstance(summary, dict)
        self.assertIn('project_id', summary)
        self.assertIn('statistics', summary)
        self.assertEqual(summary['project_id'], '17741')

class TestAnalyticsDashboard(unittest.TestCase):
    """Test cases for analytics dashboard"""

    def setUp(self):
        """Set up test environment"""
        # Mock dependencies
        self.mock_theme_manager = Mock()
        self.mock_theme_manager.get_color.return_value = "#ffffff"

        self.mock_recommendation_engine = Mock()
        self.mock_recommendation_engine.get_analytics_insights.return_value = {
            'user_engagement': {'total_users': 5, 'active_users_7d': 3},
            'model_status': {'content_model_ready': True, 'total_projects': 10},
            'recommendation_performance': {'avg_score': 0.75},
            'popular_projects': [{'project_id': '17741', 'popularity_score': 0.9}],
            'temporal_insights': {'peak_hour': 14, 'peak_day': 2}
        }

        # Mock tkinter components for testing
        self.mock_parent = Mock()

    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        dashboard = AnalyticsDashboard(
            self.mock_parent,
            self.mock_theme_manager,
            self.mock_recommendation_engine
        )

        self.assertIsNotNone(dashboard)
        self.assertEqual(dashboard.auto_refresh, True)
        self.assertEqual(dashboard.refresh_interval, 30)

    def test_metrics_update(self):
        """Test metrics update functionality"""
        dashboard = AnalyticsDashboard(
            self.mock_parent,
            self.mock_theme_manager,
            self.mock_recommendation_engine
        )

        # Mock insights data
        insights = {
            'user_engagement': {'total_users': 10, 'active_users_7d': 8},
            'model_status': {'content_model_ready': True, 'total_projects': 20}
        }

        # Update metrics
        dashboard.update_metrics_display(insights)

        # Verify calls were made
        self.assertTrue(hasattr(dashboard, 'metrics_vars'))

    def test_dashboard_data_export(self):
        """Test dashboard data retrieval"""
        dashboard = AnalyticsDashboard(
            self.mock_parent,
            self.mock_theme_manager,
            self.mock_recommendation_engine
        )

        # Get dashboard data
        data = dashboard.get_dashboard_data()

        self.assertIsInstance(data, dict)
        self.assertIn('analytics_cache', data)
        self.assertIn('last_updated', data)

class TestSmartNavigation(unittest.TestCase):
    """Test cases for smart navigation features"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

        # Mock dependencies
        self.mock_gui_controller = Mock()
        self.mock_theme_manager = Mock()
        self.mock_entry = Mock()

        # Create recommendation engine
        self.recommendation_engine = RecommendationEngine(data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_smart_autocomplete_initialization(self):
        """Test smart autocomplete initialization"""
        autocomplete = SmartAutoComplete(self.mock_entry, self.recommendation_engine)

        self.assertIsNotNone(autocomplete)
        self.assertEqual(autocomplete.entry, self.mock_entry)
        self.assertEqual(autocomplete.recommendation_engine, self.recommendation_engine)

    def test_suggestion_scoring(self):
        """Test suggestion relevance scoring"""
        autocomplete = SmartAutoComplete(self.mock_entry, self.recommendation_engine)

        # Test scoring
        score = autocomplete._calculate_relevance_score(
            query='MTR',
            project_name='17741 - QPS MTR RM Upgrades',
            metadata={'access_count': 10, 'last_accessed': datetime.now().isoformat()}
        )

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_navigation_integration(self):
        """Test smart navigation integration"""
        integration = SmartNavigationIntegration(
            Mock(), self.mock_gui_controller, self.mock_theme_manager
        )

        self.assertIsNotNone(integration)
        self.assertIsNotNone(integration.recommendation_engine)

    def test_user_action_tracking(self):
        """Test user action tracking"""
        integration = SmartNavigationIntegration(
            Mock(), self.mock_gui_controller, self.mock_theme_manager
        )

        # Track action
        integration.track_user_action('search', '17741', {'test': True})

        # Verify session context was updated
        self.assertIn('last_action', integration.current_session['context'])
        self.assertEqual(integration.current_session['context']['last_action'], 'search')

class TestIntegration(unittest.TestCase):
    """Integration tests for ML components"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end ML workflow"""
        # 1. Initialize components
        recommendation_engine = RecommendationEngine(data_dir=self.temp_dir)

        pipeline_config = {
            'watch_directories': [self.temp_dir],
            'max_workers': 1,
            'enable_monitoring': False,
            'db_path': os.path.join(self.temp_dir, 'integration_test.db')
        }
        ingestion_pipeline = DataIngestionPipeline(pipeline_config)

        # 2. Create test data
        test_documents = [
            {
                'project_folder': '17741 - Test Project A',
                'document_name': 'Document A.pdf',
                'document_path': '/test/a.pdf',
                'extracted_info': {'type': 'handover'}
            },
            {
                'project_folder': '17742 - Test Project B',
                'document_name': 'Document B.pdf',
                'document_path': '/test/b.pdf',
                'extracted_info': {'type': 'design'}
            }
        ]

        test_history = [
            {
                'user_id': 'integration_user',
                'project_code': '17741',
                'action': 'open',
                'timestamp': datetime.now()
            }
        ]

        # 3. Train ML models
        training_metrics = recommendation_engine.train_models(test_documents, test_history)
        self.assertIn('training_time', training_metrics)

        # 4. Generate recommendations
        recommendations = recommendation_engine.get_recommendations(
            user_id='integration_user',
            num_recommendations=5
        )
        self.assertIsInstance(recommendations, list)

        # 5. Test analytics
        insights = recommendation_engine.get_analytics_insights()
        self.assertIn('model_status', insights)

        # 6. Clean up
        ingestion_pipeline.stop()

    def test_performance_benchmarks(self):
        """Test performance benchmarks for ML components"""
        recommendation_engine = RecommendationEngine(data_dir=self.temp_dir)

        # Generate larger test dataset
        large_training_data = []
        for i in range(100):
            large_training_data.append({
                'project_folder': f'{17000 + i} - Test Project {i}',
                'document_name': f'Document {i}.pdf',
                'document_path': f'/test/{i}.pdf',
                'extracted_info': {'type': 'test', 'index': i}
            })

        large_user_history = []
        for i in range(50):
            large_user_history.append({
                'user_id': f'user_{i % 10}',
                'project_code': str(17000 + i),
                'action': 'open',
                'timestamp': datetime.now() - timedelta(hours=i)
            })

        # Benchmark training
        start_time = time.time()
        training_metrics = recommendation_engine.train_models(large_training_data, large_user_history)
        training_time = time.time() - start_time

        self.assertLess(training_time, 30)  # Should complete within 30 seconds

        # Benchmark recommendations
        start_time = time.time()
        recommendations = recommendation_engine.get_recommendations('user_1', num_recommendations=10)
        recommendation_time = time.time() - start_time

        self.assertLess(recommendation_time, 1)  # Should complete within 1 second

        print(f"Performance benchmarks:")
        print(f"  Training time: {training_time:.2f}s for {len(large_training_data)} documents")
        print(f"  Recommendation time: {recommendation_time:.3f}s for 10 recommendations")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_empty_training_data(self):
        """Test handling of empty training data"""
        engine = RecommendationEngine()

        # Train with empty data
        metrics = engine.train_models([], [])

        self.assertIsInstance(metrics, dict)
        # Should handle gracefully without crashing

    def test_invalid_user_interactions(self):
        """Test handling of invalid user interactions"""
        engine = RecommendationEngine()

        # Try to get recommendations for non-existent user
        recommendations = engine.get_recommendations('nonexistent_user')

        self.assertIsInstance(recommendations, list)
        # Should return fallback recommendations

    def test_corrupted_data_handling(self):
        """Test handling of corrupted or malformed data"""
        engine = RecommendationEngine()

        # Test with malformed training data
        corrupted_data = [
            {'invalid': 'data'},
            None,
            {'project_folder': 'Test', 'missing_fields': True}
        ]

        # Should not crash
        try:
            metrics = engine.train_models(corrupted_data, [])
            self.assertIsInstance(metrics, dict)
        except Exception as e:
            self.fail(f"Should handle corrupted data gracefully, but raised: {e}")

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestRecommendationEngine,
        TestDataIngestionPipeline,
        TestAnalyticsDashboard,
        TestSmartNavigation,
        TestIntegration,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Error:')[-1].strip()}")

    return result.wasSuccessful()

if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing

    print("Running comprehensive ML component tests...")
    print(f"Python version: {sys.version}")
    print(f"Test directory: {os.path.dirname(__file__)}")
    print("-" * 60)

    success = run_all_tests()
    sys.exit(0 if success else 1)