"""
Performance Benchmarks for Project QuickNav Database

This module provides comprehensive performance testing and benchmarking
for the analytics database implementation.

Benchmark Categories:
1. Database Operations (CRUD performance)
2. Query Performance (Analytics queries)
3. Indexing Effectiveness
4. Concurrent Access Performance
5. Storage Efficiency
6. Memory Usage Analysis
"""

import time
import random
import string
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
import uuid
import psutil
import os

from database_manager import DatabaseManager
from analytics_queries import AnalyticsQueries


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite for the analytics database."""

    def __init__(self, db_path: str = None):
        self.db = DatabaseManager(db_path)
        self.analytics = AnalyticsQueries(self.db)
        self.results = {}
        self.test_data = {}

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks and return comprehensive results."""
        print("Starting comprehensive performance benchmark suite...")
        start_time = time.time()

        # Generate test data first
        print("Generating test data...")
        self._generate_test_data()

        # Run benchmark categories
        benchmark_categories = [
            ("Database Operations", self._benchmark_database_operations),
            ("Query Performance", self._benchmark_query_performance),
            ("Indexing Effectiveness", self._benchmark_indexing),
            ("Concurrent Access", self._benchmark_concurrent_access),
            ("Storage Efficiency", self._benchmark_storage_efficiency),
            ("Memory Usage", self._benchmark_memory_usage),
        ]

        for category_name, benchmark_func in benchmark_categories:
            print(f"\nRunning {category_name} benchmarks...")
            try:
                category_results = benchmark_func()
                self.results[category_name.lower().replace(' ', '_')] = category_results
                print(f"✅ {category_name} completed")
            except Exception as e:
                print(f"❌ {category_name} failed: {e}")
                self.results[category_name.lower().replace(' ', '_')] = {"error": str(e)}

        total_time = time.time() - start_time
        self.results['benchmark_metadata'] = {
            'total_runtime_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
            'database_type': 'DuckDB' if self.db.use_duckdb else 'SQLite',
            'test_data_size': self.test_data
        }

        print(f"\n🎉 All benchmarks completed in {total_time:.2f} seconds")
        return self.results

    def _generate_test_data(self):
        """Generate realistic test data for benchmarking."""
        # Generate projects
        project_count = 100
        projects = []
        for i in range(project_count):
            project_id = f"{10000 + i}"
            project_data = {
                'project_id': project_id,
                'project_code': project_id,
                'project_name': f"Test Project {i:03d}",
                'full_path': f"/projects/{10000 + i // 10}00-{10000 + i // 10}99/{project_id} - Test Project {i:03d}",
                'range_folder': f"{10000 + i // 10}00 - {10000 + i // 10}99",
                'metadata': {'test': True, 'category': random.choice(['commercial', 'residential', 'industrial'])}
            }
            projects.append(project_data)
            self.db.upsert_project(**project_data)

        # Generate documents
        document_count = 500
        documents = []
        doc_types = ['lld', 'hld', 'co', 'floor_plan', 'scope', 'quote']
        for i in range(document_count):
            project = random.choice(projects)
            doc_data = {
                'document_id': str(uuid.uuid4()),
                'project_id': project['project_id'],
                'file_path': f"{project['full_path']}/documents/doc_{i:04d}.pdf",
                'filename': f"Document_{i:04d}_REV{random.randint(100, 999)}.pdf",
                'file_extension': '.pdf',
                'file_size_bytes': random.randint(50000, 5000000),
                'document_type': random.choice(doc_types),
                'folder_category': random.choice(['System Designs', 'Sales Handover', 'BOM & Orders']),
                'version_numeric': random.randint(100, 999),
                'status_weight': random.uniform(0, 2),
                'word_count': random.randint(500, 5000),
                'page_count': random.randint(1, 50)
            }
            documents.append(doc_data)
            self.db.upsert_document(doc_data)

        # Generate user sessions and activities
        session_count = 50
        activity_count = 2000
        sessions = []

        for i in range(session_count):
            session_id = self.db.start_session(
                user_id=f"test_user_{i % 10}",
                app_version="2.0.0-benchmark",
                os_platform="TestOS"
            )
            sessions.append(session_id)

        # Generate activities
        activity_types = ['navigate', 'search', 'document_open', 'ai_query']
        for i in range(activity_count):
            session_id = random.choice(sessions)
            project = random.choice(projects)
            document = random.choice([d for d in documents if d['project_id'] == project['project_id']])

            self.db.record_activity(
                session_id=session_id,
                activity_type=random.choice(activity_types),
                project_id=project['project_id'],
                document_id=document['document_id'] if random.random() > 0.5 else None,
                search_query=f"search term {i}" if random.random() > 0.7 else None,
                response_time_ms=random.randint(50, 2000),
                success=random.random() > 0.05  # 95% success rate
            )

        self.test_data = {
            'projects': project_count,
            'documents': document_count,
            'sessions': session_count,
            'activities': activity_count
        }

    # =====================================================
    # DATABASE OPERATIONS BENCHMARKS
    # =====================================================

    def _benchmark_database_operations(self) -> Dict[str, Any]:
        """Benchmark basic CRUD operations."""
        results = {}

        # INSERT performance
        insert_times = []
        for i in range(100):
            start_time = time.time()
            project_data = {
                'project_id': f"bench_{i}",
                'project_code': f"bench_{i}",
                'project_name': f"Benchmark Project {i}",
                'full_path': f"/benchmark/projects/bench_{i}",
                'range_folder': "90000 - 90999"
            }
            self.db.upsert_project(**project_data)
            insert_times.append((time.time() - start_time) * 1000)

        results['insert_performance'] = {
            'avg_time_ms': statistics.mean(insert_times),
            'median_time_ms': statistics.median(insert_times),
            'p95_time_ms': self._percentile(insert_times, 95),
            'p99_time_ms': self._percentile(insert_times, 99),
            'operations_per_second': 1000 / statistics.mean(insert_times)
        }

        # SELECT performance
        select_times = []
        for i in range(100):
            start_time = time.time()
            self.db.get_project(f"bench_{i}")
            select_times.append((time.time() - start_time) * 1000)

        results['select_performance'] = {
            'avg_time_ms': statistics.mean(select_times),
            'median_time_ms': statistics.median(select_times),
            'p95_time_ms': self._percentile(select_times, 95),
            'operations_per_second': 1000 / statistics.mean(select_times)
        }

        # UPDATE performance
        update_times = []
        for i in range(50):
            start_time = time.time()
            update_query = "UPDATE projects SET project_name = ? WHERE project_id = ?"
            self.db.execute_write(update_query, [f"Updated Project {i}", f"bench_{i}"])
            update_times.append((time.time() - start_time) * 1000)

        results['update_performance'] = {
            'avg_time_ms': statistics.mean(update_times),
            'median_time_ms': statistics.median(update_times),
            'operations_per_second': 1000 / statistics.mean(update_times)
        }

        # DELETE performance
        delete_times = []
        for i in range(50):
            start_time = time.time()
            delete_query = "DELETE FROM projects WHERE project_id = ?"
            self.db.execute_write(delete_query, [f"bench_{i}"])
            delete_times.append((time.time() - start_time) * 1000)

        results['delete_performance'] = {
            'avg_time_ms': statistics.mean(delete_times),
            'median_time_ms': statistics.median(delete_times),
            'operations_per_second': 1000 / statistics.mean(delete_times)
        }

        return results

    # =====================================================
    # QUERY PERFORMANCE BENCHMARKS
    # =====================================================

    def _benchmark_query_performance(self) -> Dict[str, Any]:
        """Benchmark analytical query performance."""
        results = {}

        # Define test queries with expected performance targets
        test_queries = [
            {
                'name': 'popular_projects',
                'function': lambda: self.analytics.get_popular_projects(days=30, limit=10),
                'target_ms': 100,
                'description': 'Get most popular projects'
            },
            {
                'name': 'recent_activity',
                'function': lambda: self.analytics.get_recent_activity(days=7),
                'target_ms': 50,
                'description': 'Get recent activity summary'
            },
            {
                'name': 'user_engagement',
                'function': lambda: self.analytics.get_user_engagement_metrics(days=30),
                'target_ms': 200,
                'description': 'Calculate user engagement metrics'
            },
            {
                'name': 'document_versions',
                'function': lambda: self.db.get_document_versions('10001', 'lld'),
                'target_ms': 30,
                'description': 'Get document versions for project'
            },
            {
                'name': 'project_search',
                'function': lambda: self.db.search_projects('Test', limit=10),
                'target_ms': 50,
                'description': 'Search projects by name'
            },
            {
                'name': 'user_similarity_features',
                'function': lambda: self.analytics.get_user_similarity_features(days=90),
                'target_ms': 300,
                'description': 'Extract ML user similarity features'
            }
        ]

        for query_test in test_queries:
            times = []
            errors = 0

            # Run each query multiple times
            for _ in range(10):
                try:
                    start_time = time.time()
                    result = query_test['function']()
                    execution_time = (time.time() - start_time) * 1000
                    times.append(execution_time)
                except Exception as e:
                    errors += 1
                    print(f"Query error: {e}")

            if times:
                avg_time = statistics.mean(times)
                results[query_test['name']] = {
                    'description': query_test['description'],
                    'avg_time_ms': avg_time,
                    'median_time_ms': statistics.median(times),
                    'p95_time_ms': self._percentile(times, 95),
                    'target_ms': query_test['target_ms'],
                    'performance_ratio': query_test['target_ms'] / avg_time,
                    'meets_target': avg_time <= query_test['target_ms'],
                    'errors': errors,
                    'queries_per_second': 1000 / avg_time if avg_time > 0 else 0
                }
            else:
                results[query_test['name']] = {
                    'description': query_test['description'],
                    'error': 'All queries failed',
                    'errors': errors
                }

        return results

    # =====================================================
    # INDEXING EFFECTIVENESS BENCHMARKS
    # =====================================================

    def _benchmark_indexing(self) -> Dict[str, Any]:
        """Test the effectiveness of database indexes."""
        results = {}

        # Test queries that should benefit from indexes
        index_tests = [
            {
                'name': 'project_code_lookup',
                'query': "SELECT * FROM projects WHERE project_code = ?",
                'params': ['10001'],
                'description': 'Project lookup by code (should use index)'
            },
            {
                'name': 'activities_by_timestamp',
                'query': "SELECT * FROM user_activities WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 100",
                'params': [datetime.now() - timedelta(days=1)],
                'description': 'Recent activities (should use timestamp index)'
            },
            {
                'name': 'documents_by_project',
                'query': "SELECT * FROM documents WHERE project_id = ? ORDER BY version_numeric DESC",
                'params': ['10001'],
                'description': 'Documents by project (should use composite index)'
            },
            {
                'name': 'activities_by_type',
                'query': "SELECT * FROM user_activities WHERE activity_type = ? AND timestamp > ?",
                'params': ['navigate', datetime.now() - timedelta(days=7)],
                'description': 'Activities by type and time (should use composite index)'
            }
        ]

        for test in index_tests:
            times = []
            for _ in range(20):
                start_time = time.time()
                self.db.execute_query(test['query'], test['params'], prefer_analytics=True)
                times.append((time.time() - start_time) * 1000)

            results[test['name']] = {
                'description': test['description'],
                'avg_time_ms': statistics.mean(times),
                'median_time_ms': statistics.median(times),
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0
            }

        return results

    # =====================================================
    # CONCURRENT ACCESS BENCHMARKS
    # =====================================================

    def _benchmark_concurrent_access(self) -> Dict[str, Any]:
        """Test performance under concurrent access patterns."""
        results = {}

        def worker_function(worker_id: int, operation_count: int, results_list: List):
            """Worker function for concurrent testing."""
            worker_times = []
            errors = 0

            for i in range(operation_count):
                try:
                    start_time = time.time()

                    # Mix of read and write operations
                    if i % 10 == 0:  # 10% writes
                        session_id = self.db.start_session(f"concurrent_user_{worker_id}")
                        if session_id:
                            self.db.record_activity(
                                session_id=session_id,
                                activity_type='navigate',
                                project_id='10001',
                                response_time_ms=random.randint(50, 500)
                            )
                    else:  # 90% reads
                        self.db.get_popular_projects(days=7, limit=5)

                    execution_time = (time.time() - start_time) * 1000
                    worker_times.append(execution_time)

                except Exception as e:
                    errors += 1

            results_list.append({
                'worker_id': worker_id,
                'avg_time_ms': statistics.mean(worker_times) if worker_times else 0,
                'total_operations': len(worker_times),
                'errors': errors
            })

        # Test with different concurrency levels
        concurrency_levels = [1, 2, 5, 10]
        operations_per_worker = 50

        for concurrency in concurrency_levels:
            worker_results = []
            threads = []

            start_time = time.time()

            # Start worker threads
            for worker_id in range(concurrency):
                thread = threading.Thread(
                    target=worker_function,
                    args=(worker_id, operations_per_worker, worker_results)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            total_time = time.time() - start_time

            # Aggregate results
            total_operations = sum(w['total_operations'] for w in worker_results)
            total_errors = sum(w['errors'] for w in worker_results)
            avg_worker_time = statistics.mean([w['avg_time_ms'] for w in worker_results if w['avg_time_ms'] > 0])

            results[f'concurrency_{concurrency}'] = {
                'concurrency_level': concurrency,
                'total_operations': total_operations,
                'total_time_seconds': total_time,
                'operations_per_second': total_operations / total_time,
                'avg_operation_time_ms': avg_worker_time,
                'total_errors': total_errors,
                'error_rate': total_errors / total_operations if total_operations > 0 else 0,
                'worker_results': worker_results
            }

        return results

    # =====================================================
    # STORAGE EFFICIENCY BENCHMARKS
    # =====================================================

    def _benchmark_storage_efficiency(self) -> Dict[str, Any]:
        """Analyze storage efficiency and compression."""
        results = {}

        try:
            # Get database file sizes
            db_stats = self.db.get_database_stats()

            # Calculate storage efficiency metrics
            total_records = sum(db_stats.get(table, 0) for table in [
                'projects', 'documents', 'user_activities', 'ai_messages'
            ])

            storage_per_record = db_stats.get('database_size_mb', 0) * 1024 * 1024 / total_records if total_records > 0 else 0

            results['storage_metrics'] = {
                'total_size_mb': db_stats.get('database_size_mb', 0),
                'total_records': total_records,
                'bytes_per_record': storage_per_record,
                'records_per_mb': total_records / db_stats.get('database_size_mb', 1) if db_stats.get('database_size_mb', 0) > 0 else 0
            }

            # Test compression effectiveness (if using DuckDB)
            if self.db.use_duckdb:
                # Sample data compression test
                sample_data = ['{"test": "data", "timestamp": "2024-01-01T00:00:00", "value": ' + str(i) + '}' for i in range(1000)]

                start_time = time.time()
                for i, data in enumerate(sample_data):
                    self.db.execute_write(
                        "INSERT INTO user_activities (activity_id, session_id, activity_type, action_details) VALUES (?, ?, ?, ?)",
                        [f"compress_test_{i}", "test_session", "compress_test", data]
                    )
                compression_time = time.time() - start_time

                results['compression_test'] = {
                    'records_inserted': len(sample_data),
                    'insertion_time_seconds': compression_time,
                    'records_per_second': len(sample_data) / compression_time
                }

        except Exception as e:
            results['error'] = str(e)

        return results

    # =====================================================
    # MEMORY USAGE BENCHMARKS
    # =====================================================

    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Monitor memory usage during database operations."""
        results = {}

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory usage during large query
        memory_before_large_query = process.memory_info().rss / 1024 / 1024
        self.analytics.get_user_similarity_features(days=90)
        memory_after_large_query = process.memory_info().rss / 1024 / 1024

        # Memory usage during bulk operations
        memory_before_bulk = process.memory_info().rss / 1024 / 1024
        for i in range(100):
            self.db.record_activity(
                session_id="memory_test",
                activity_type="memory_test",
                action_details={"test_data": "x" * 1000}  # 1KB per record
            )
        memory_after_bulk = process.memory_info().rss / 1024 / 1024

        results['memory_usage'] = {
            'initial_memory_mb': initial_memory,
            'large_query_memory_increase_mb': memory_after_large_query - memory_before_large_query,
            'bulk_operations_memory_increase_mb': memory_after_bulk - memory_before_bulk,
            'total_memory_increase_mb': memory_after_bulk - initial_memory,
            'memory_efficiency_mb_per_1000_records': (memory_after_bulk - memory_before_bulk) * 10  # Extrapolate to 1000 records
        }

        return results

    # =====================================================
    # UTILITY METHODS
    # =====================================================

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value from a list of numbers."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def print_benchmark_summary(self):
        """Print a formatted summary of benchmark results."""
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return

        print("\n" + "="*80)
        print("PROJECT QUICKNAV DATABASE PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)

        metadata = self.results.get('benchmark_metadata', {})
        print(f"Database Type: {metadata.get('database_type', 'Unknown')}")
        print(f"Total Runtime: {metadata.get('total_runtime_seconds', 0):.2f} seconds")
        print(f"Test Data Size: {metadata.get('test_data_size', {})}")
        print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")

        # Database Operations Summary
        if 'database_operations' in self.results:
            ops = self.results['database_operations']
            print("\n📊 DATABASE OPERATIONS PERFORMANCE")
            print("-" * 50)
            for op_type, metrics in ops.items():
                if isinstance(metrics, dict) and 'avg_time_ms' in metrics:
                    print(f"{op_type.replace('_', ' ').title()}: {metrics['avg_time_ms']:.2f}ms avg, {metrics['operations_per_second']:.1f} ops/sec")

        # Query Performance Summary
        if 'query_performance' in self.results:
            queries = self.results['query_performance']
            print("\n🔍 QUERY PERFORMANCE")
            print("-" * 50)
            for query_name, metrics in queries.items():
                if isinstance(metrics, dict) and 'avg_time_ms' in metrics:
                    status = "✅" if metrics.get('meets_target', False) else "⚠️"
                    print(f"{status} {query_name}: {metrics['avg_time_ms']:.2f}ms (target: {metrics.get('target_ms', 'N/A')}ms)")

        # Concurrency Performance Summary
        if 'concurrent_access' in self.results:
            concurrency = self.results['concurrent_access']
            print("\n🚀 CONCURRENT ACCESS PERFORMANCE")
            print("-" * 50)
            for level_name, metrics in concurrency.items():
                if isinstance(metrics, dict) and 'operations_per_second' in metrics:
                    print(f"Concurrency {metrics['concurrency_level']}: {metrics['operations_per_second']:.1f} ops/sec, {metrics['error_rate']*100:.1f}% errors")

        # Storage and Memory Summary
        if 'storage_efficiency' in self.results:
            storage = self.results['storage_efficiency'].get('storage_metrics', {})
            print("\n💾 STORAGE EFFICIENCY")
            print("-" * 50)
            print(f"Total Size: {storage.get('total_size_mb', 0):.2f} MB")
            print(f"Records: {storage.get('total_records', 0)}")
            print(f"Bytes per Record: {storage.get('bytes_per_record', 0):.2f}")

        if 'memory_usage' in self.results:
            memory = self.results['memory_usage'].get('memory_usage', {})
            print("\n🧠 MEMORY USAGE")
            print("-" * 50)
            print(f"Initial Memory: {memory.get('initial_memory_mb', 0):.2f} MB")
            print(f"Large Query Increase: {memory.get('large_query_memory_increase_mb', 0):.2f} MB")
            print(f"Bulk Operations Increase: {memory.get('bulk_operations_memory_increase_mb', 0):.2f} MB")

        print("\n" + "="*80)

    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"Benchmark results saved to {filename}")
        except Exception as e:
            print(f"Failed to save results: {e}")

    def cleanup_test_data(self):
        """Clean up test data created during benchmarking."""
        try:
            # Clean up benchmark projects
            self.db.execute_write("DELETE FROM projects WHERE project_id LIKE 'bench_%'")
            self.db.execute_write("DELETE FROM user_activities WHERE activity_type = 'memory_test'")
            self.db.execute_write("DELETE FROM user_activities WHERE activity_type = 'compress_test'")
            print("Test data cleaned up successfully")
        except Exception as e:
            print(f"Failed to clean up test data: {e}")

    def close(self):
        """Close database connection and clean up."""
        self.cleanup_test_data()
        self.db.close()


def main():
    """Run the complete benchmark suite."""
    print("Starting Project QuickNav Database Performance Benchmark")
    print("This will create test data and run comprehensive performance tests.")

    benchmark = PerformanceBenchmark()

    try:
        # Run all benchmarks
        results = benchmark.run_all_benchmarks()

        # Print summary
        benchmark.print_benchmark_summary()

        # Save results
        benchmark.save_results()

        # Return results for potential further analysis
        return results

    finally:
        benchmark.close()


if __name__ == "__main__":
    main()