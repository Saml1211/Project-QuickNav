# Project QuickNav Data Pipeline Architecture

## Executive Summary

This document presents a comprehensive data pipeline architecture for Project QuickNav, designed to scale the existing Python-based project navigation system with robust data engineering capabilities. The architecture integrates ETL/ELT processes, real-time streaming, feature stores, and embedded analytics while maintaining cross-platform compatibility.

## Current System Analysis

### Existing Components
- **Python Backend**: `find_project_path.py` - Core project resolution logic
- **Document Navigator**: `doc_navigator.py` - Advanced document parsing and ranking
- **Tkinter GUI**: Cross-platform interface with AI integration
- **MCP Server**: Model Context Protocol integration for AI agents
- **Training Data System**: JSON-based document cataloging

### Data Flow Patterns
1. **Project Resolution**: OneDrive → File System Scan → Project Mapping
2. **Document Discovery**: Recursive File Walking → Metadata Extraction → Classification
3. **Training Data**: Document Catalog → JSON Export → AI Analysis
4. **User Interaction**: GUI/CLI → Backend API → File System Operations

## Proposed Data Pipeline Architecture

### 1. Data Ingestion Layer

#### Batch Ingestion Pipeline
```python
# Apache Airflow DAG Structure
quicknav_etl_dag = {
    "schedule_interval": "@daily",
    "max_active_runs": 1,
    "catchup": False,
    "tasks": [
        "scan_onedrive_changes",
        "extract_document_metadata",
        "transform_training_data",
        "load_feature_store",
        "update_search_indices"
    ]
}
```

**Technology Stack:**
- **Orchestration**: Apache Airflow (lightweight deployment)
- **File Monitoring**: Watchdog + inotify/ReadDirectoryChangesW
- **Data Extraction**: Custom parsers extending existing `DocumentParser`
- **Change Detection**: File system timestamps + MD5 hashing

#### Streaming Ingestion Pipeline
```python
# Real-time user behavior tracking
stream_processors = {
    "user_activity": {
        "source": "GUI/CLI interactions",
        "processor": "Kafka Streams / Python asyncio",
        "sink": "DuckDB + time-series tables"
    },
    "document_access": {
        "source": "File system events",
        "processor": "Event-driven Python workers",
        "sink": "Feature store + analytics DB"
    }
}
```

### 2. Data Storage Architecture

#### Primary Storage: DuckDB (Embedded OLAP)
**Rationale**:
- Zero-configuration embedded database
- Excellent analytics performance (vectorized execution)
- Native Python integration
- Cross-platform compatibility
- Columnar storage for analytics workloads

```sql
-- Core schema design
CREATE SCHEMA quicknav;

-- Projects dimension table
CREATE TABLE quicknav.projects (
    project_id VARCHAR PRIMARY KEY,
    project_name VARCHAR NOT NULL,
    project_path VARCHAR NOT NULL,
    onedrive_sync_status VARCHAR,
    created_date TIMESTAMP,
    last_accessed TIMESTAMP,
    folder_structure JSON,
    INDEX(project_id, last_accessed)
);

-- Documents fact table
CREATE TABLE quicknav.documents (
    document_id VARCHAR PRIMARY KEY,
    project_id VARCHAR REFERENCES projects(project_id),
    file_path VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    doc_type VARCHAR NOT NULL,
    version_info JSON,
    metadata JSON,
    file_size BIGINT,
    created_date TIMESTAMP,
    modified_date TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    INDEX(project_id, doc_type, modified_date)
);

-- User behavior tracking
CREATE TABLE quicknav.user_interactions (
    interaction_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    session_id VARCHAR,
    timestamp TIMESTAMP,
    action_type VARCHAR, -- 'search', 'navigate', 'open_doc', 'ai_query'
    project_id VARCHAR,
    document_id VARCHAR,
    search_query VARCHAR,
    response_time_ms INTEGER,
    success BOOLEAN
);

-- Feature store tables
CREATE TABLE quicknav.project_features (
    project_id VARCHAR PRIMARY KEY,
    features JSON, -- Pre-computed features for ML models
    feature_version INTEGER,
    computed_at TIMESTAMP
);
```

#### Secondary Storage: SQLite (Lightweight Operations)
**Use Cases**:
- GUI settings and preferences
- Session state management
- Offline caching when OneDrive unavailable

#### Cache Layer: Redis (Optional for Production)
**Use Cases**:
- Search result caching
- Session data for multi-user deployments
- Real-time feature serving

### 3. ETL/ELT Pipeline Design

#### Extract Phase
```python
class OneDriveExtractor:
    """Extracts project and document metadata from OneDrive structure"""

    def __init__(self, config: Dict):
        self.onedrive_root = resolve_project_root()
        self.file_monitor = ProjectFileMonitor()
        self.change_detector = FileChangeDetector()

    def extract_incremental(self, last_sync_time: datetime) -> Iterator[Dict]:
        """Extract only changed files since last sync"""
        changes = self.change_detector.get_changes_since(last_sync_time)
        for change in changes:
            yield {
                'file_path': change.path,
                'change_type': change.type,  # 'created', 'modified', 'deleted'
                'timestamp': change.timestamp,
                'metadata': self.extract_file_metadata(change.path)
            }

    def extract_full_scan(self) -> Iterator[Dict]:
        """Full project structure scan for initial load"""
        for project_path in self.scan_project_directories():
            project_data = self.extract_project_metadata(project_path)
            yield project_data

            for doc_path in self.scan_documents(project_path):
                doc_data = self.extract_document_metadata(doc_path)
                yield doc_data
```

#### Transform Phase
```python
class DocumentTransformer:
    """Transforms raw document data into analytics-ready format"""

    def __init__(self):
        self.parser = DocumentParser()  # Existing component
        self.classifier = DocumentTypeClassifier()
        self.feature_engineer = DocumentFeatureEngineer()

    def transform_document(self, raw_doc: Dict) -> Dict:
        """Transform raw document data"""
        # Parse filename metadata (existing logic)
        parsed = self.parser.parse_filename(raw_doc['file_name'])

        # Classify document type
        doc_type = self.classifier.classify_document(raw_doc['file_path'])

        # Engineer features for ML
        features = self.feature_engineer.extract_features(raw_doc, parsed)

        return {
            'document_id': generate_document_id(raw_doc['file_path']),
            'project_id': extract_project_id(raw_doc['file_path']),
            'file_path': raw_doc['file_path'],
            'file_name': raw_doc['file_name'],
            'doc_type': doc_type['type'] if doc_type else 'unknown',
            'version_info': parsed,
            'metadata': {
                'original_metadata': raw_doc,
                'parsed_metadata': parsed,
                'classification': doc_type,
                'features': features
            },
            'file_size': raw_doc.get('file_size', 0),
            'created_date': raw_doc.get('created_date'),
            'modified_date': raw_doc.get('modified_date'),
            'processed_at': datetime.utcnow()
        }
```

#### Load Phase
```python
class DuckDBLoader:
    """Loads transformed data into DuckDB analytics database"""

    def __init__(self, db_path: str):
        self.db = duckdb.connect(db_path)
        self.setup_schema()

    def load_batch(self, records: List[Dict], table: str):
        """Load batch of records with upsert logic"""
        df = pd.DataFrame(records)

        # Use DuckDB's native upsert capabilities
        self.db.execute(f"""
            INSERT INTO quicknav.{table}
            SELECT * FROM df
            ON CONFLICT DO UPDATE SET
                modified_date = EXCLUDED.modified_date,
                metadata = EXCLUDED.metadata,
                last_accessed = EXCLUDED.last_accessed
        """)

    def create_materialized_views(self):
        """Create performance-optimized views"""
        self.db.execute("""
            CREATE OR REPLACE VIEW quicknav.project_summary AS
            SELECT
                p.project_id,
                p.project_name,
                COUNT(d.document_id) as document_count,
                MAX(d.modified_date) as last_document_update,
                SUM(d.file_size) as total_size_bytes,
                COUNT(DISTINCT d.doc_type) as doc_types_count
            FROM quicknav.projects p
            LEFT JOIN quicknav.documents d ON p.project_id = d.project_id
            GROUP BY p.project_id, p.project_name
        """)
```

### 4. Streaming Pipeline Architecture

#### Real-time Event Processing
```python
class UserBehaviorStream:
    """Processes real-time user interactions"""

    def __init__(self):
        self.event_queue = asyncio.Queue()
        self.batch_size = 100
        self.flush_interval = 30  # seconds

    async def process_events(self):
        """Process events in micro-batches"""
        batch = []
        last_flush = time.time()

        while True:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                batch.append(event)

                # Flush if batch full or time elapsed
                if (len(batch) >= self.batch_size or
                    time.time() - last_flush > self.flush_interval):
                    await self.flush_batch(batch)
                    batch.clear()
                    last_flush = time.time()

            except asyncio.TimeoutError:
                # Flush partial batch on timeout
                if batch:
                    await self.flush_batch(batch)
                    batch.clear()
                    last_flush = time.time()

    async def flush_batch(self, events: List[Dict]):
        """Flush batch to database"""
        loader = DuckDBLoader(self.db_path)
        loader.load_batch(events, 'user_interactions')
```

#### Integration with Existing GUI
```python
# Modified GUI event tracking
class EnhancedGUIController(GUIController):
    """Extended controller with event streaming"""

    def __init__(self):
        super().__init__()
        self.event_stream = UserBehaviorStream()
        self.session_id = str(uuid.uuid4())

    def track_user_action(self, action_type: str, **kwargs):
        """Track user interaction"""
        event = {
            'interaction_id': str(uuid.uuid4()),
            'session_id': self.session_id,
            'timestamp': datetime.utcnow(),
            'action_type': action_type,
            'metadata': kwargs
        }

        # Async send to stream
        asyncio.create_task(
            self.event_stream.event_queue.put(event)
        )

    def search_projects(self, query: str):
        """Enhanced search with tracking"""
        start_time = time.time()
        results = super().search_projects(query)
        response_time = (time.time() - start_time) * 1000

        self.track_user_action(
            'search',
            query=query,
            result_count=len(results),
            response_time_ms=response_time
        )

        return results
```

### 5. Feature Store Architecture

#### Feature Engineering Pipeline
```python
class FeatureEngineer:
    """Generates features for ML models"""

    def compute_project_features(self, project_id: str) -> Dict:
        """Compute project-level features"""
        return {
            'document_count': self.count_documents(project_id),
            'avg_doc_age_days': self.avg_document_age(project_id),
            'doc_type_diversity': self.document_type_diversity(project_id),
            'access_frequency': self.access_frequency(project_id),
            'version_churn_rate': self.version_churn_rate(project_id),
            'folder_depth_avg': self.avg_folder_depth(project_id),
            'file_size_stats': self.file_size_statistics(project_id)
        }

    def compute_document_features(self, doc_id: str) -> Dict:
        """Compute document-level features"""
        return {
            'version_recency': self.version_recency_score(doc_id),
            'access_pattern': self.access_pattern_features(doc_id),
            'filename_complexity': self.filename_complexity_score(doc_id),
            'folder_relevance': self.folder_relevance_score(doc_id),
            'temporal_features': self.temporal_features(doc_id)
        }
```

#### Feature Serving
```python
class FeatureStore:
    """Serves features for real-time ML inference"""

    def __init__(self, db_path: str):
        self.db = duckdb.connect(db_path)
        self.cache = {}  # In-memory cache for hot features

    def get_project_features(self, project_id: str) -> Dict:
        """Get project features with caching"""
        cache_key = f"project_features:{project_id}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        features = self.db.execute("""
            SELECT features FROM quicknav.project_features
            WHERE project_id = ?
        """, [project_id]).fetchone()

        if features:
            self.cache[cache_key] = json.loads(features[0])
            return self.cache[cache_key]

        return {}

    def update_features_batch(self, feature_updates: List[Dict]):
        """Update features in batch"""
        df = pd.DataFrame(feature_updates)
        self.db.execute("""
            INSERT OR REPLACE INTO quicknav.project_features
            SELECT * FROM df
        """)

        # Invalidate cache
        for update in feature_updates:
            cache_key = f"project_features:{update['project_id']}"
            self.cache.pop(cache_key, None)
```

### 6. Integration Patterns

#### Enhanced Backend Integration
```python
class DataPipelineIntegration:
    """Integrates data pipeline with existing backend"""

    def __init__(self):
        self.feature_store = FeatureStore()
        self.user_behavior = UserBehaviorStream()
        self.existing_backend = find_project_path  # Existing module

    def enhanced_project_search(self, query: str, user_context: Dict) -> List[Dict]:
        """ML-enhanced project search"""

        # Get base results from existing logic
        base_results = self.existing_backend.search_by_name(query, self.project_root)

        # Enhance with ML features
        enhanced_results = []
        for result_path in base_results:
            project_id = extract_project_id(result_path)
            features = self.feature_store.get_project_features(project_id)

            # Calculate relevance score using features + user history
            relevance_score = self.calculate_relevance(
                query, project_id, features, user_context
            )

            enhanced_results.append({
                'path': result_path,
                'project_id': project_id,
                'relevance_score': relevance_score,
                'features': features
            })

        # Sort by relevance score
        enhanced_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        return enhanced_results
```

#### Document Navigation Enhancement
```python
class EnhancedDocumentNavigator(DocumentNavigator):
    """Enhanced document navigation with ML recommendations"""

    def __init__(self):
        super().__init__()
        self.feature_store = FeatureStore()
        self.ml_model = DocumentRecommendationModel()

    def navigate_with_recommendations(self, project_path: str,
                                    doc_type: str,
                                    user_context: Dict) -> Dict:
        """Navigate with ML-powered recommendations"""

        # Get base navigation results
        base_result = super().navigate_to_document(project_path, doc_type)

        if isinstance(base_result, str) and base_result.startswith("SUCCESS:"):
            # Single result - get related recommendations
            doc_path = base_result.replace("SUCCESS:", "")
            recommendations = self.get_related_documents(doc_path, user_context)

            return {
                'primary_document': doc_path,
                'recommendations': recommendations,
                'confidence_score': self.ml_model.predict_relevance(doc_path, user_context)
            }

        elif isinstance(base_result, str) and base_result.startswith("SELECT:"):
            # Multiple results - rank using ML
            candidates = base_result.replace("SELECT:", "").split("|")
            ranked_candidates = self.rank_candidates(candidates, user_context)

            return {
                'candidates': ranked_candidates,
                'auto_select': ranked_candidates[0] if ranked_candidates else None
            }

        return {'error': base_result}
```

### 7. Performance Optimization

#### Caching Strategy
```python
class MultiLevelCache:
    """Multi-level caching for performance optimization"""

    def __init__(self):
        self.l1_cache = {}  # In-memory (256MB limit)
        self.l2_cache = SQLiteCache()  # On-disk SQLite
        self.l3_cache = FileSystemCache()  # Computed results cache

    def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback through levels"""
        # L1: Memory
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2: SQLite
        value = self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value  # Promote to L1
            return value

        # L3: File system
        value = self.l3_cache.get(key)
        if value:
            self.l2_cache.set(key, value)  # Store in L2
            self.l1_cache[key] = value     # Store in L1
            return value

        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in all cache levels"""
        self.l1_cache[key] = value
        self.l2_cache.set(key, value, ttl)
        self.l3_cache.set(key, value, ttl)
```

#### Database Optimization
```sql
-- Optimized indices for common query patterns
CREATE INDEX idx_documents_project_type ON quicknav.documents(project_id, doc_type);
CREATE INDEX idx_documents_modified ON quicknav.documents(modified_date DESC);
CREATE INDEX idx_interactions_user_time ON quicknav.user_interactions(user_id, timestamp DESC);

-- Partitioned table for large datasets
CREATE TABLE quicknav.user_interactions_partitioned (
    LIKE quicknav.user_interactions
) PARTITION BY RANGE (timestamp);

-- Materialized aggregates for dashboards
CREATE MATERIALIZED VIEW quicknav.daily_usage_stats AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_interactions,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT project_id) as projects_accessed,
    AVG(response_time_ms) as avg_response_time
FROM quicknav.user_interactions
GROUP BY DATE(timestamp);
```

### 8. Monitoring and Observability

#### Pipeline Monitoring
```python
class PipelineMonitor:
    """Monitors data pipeline health and performance"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = AlertManager()

    def track_pipeline_run(self, pipeline_name: str, duration: float,
                          records_processed: int, success: bool):
        """Track pipeline execution metrics"""

        metric = {
            'timestamp': datetime.utcnow(),
            'pipeline': pipeline_name,
            'duration_seconds': duration,
            'records_processed': records_processed,
            'records_per_second': records_processed / duration if duration > 0 else 0,
            'success': success
        }

        self.metrics[pipeline_name].append(metric)

        # Check for alerts
        if not success:
            self.alerts.send_alert(f"Pipeline {pipeline_name} failed")

        if duration > self.get_sla_threshold(pipeline_name):
            self.alerts.send_alert(f"Pipeline {pipeline_name} exceeded SLA")

    def get_pipeline_health(self) -> Dict:
        """Get overall pipeline health metrics"""
        health = {}

        for pipeline_name, metrics in self.metrics.items():
            recent_metrics = [m for m in metrics
                            if m['timestamp'] > datetime.utcnow() - timedelta(hours=24)]

            if recent_metrics:
                health[pipeline_name] = {
                    'success_rate': sum(m['success'] for m in recent_metrics) / len(recent_metrics),
                    'avg_duration': sum(m['duration_seconds'] for m in recent_metrics) / len(recent_metrics),
                    'total_records': sum(m['records_processed'] for m in recent_metrics),
                    'last_run': max(m['timestamp'] for m in recent_metrics)
                }

        return health
```

### 9. Deployment Architecture

#### Local Development/Single-User Deployment
```yaml
# docker-compose.yml for local development
version: '3.8'
services:
  quicknav-pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - /Users/${USER}/OneDrive - Pro AV Solutions:/app/onedrive:ro
    environment:
      - QUICKNAV_DB_PATH=/app/data/quicknav.duckdb
      - QUICKNAV_LOG_LEVEL=INFO
    ports:
      - "8080:8080"  # Pipeline monitoring dashboard

  airflow-scheduler:
    image: apache/airflow:2.7.0-python3.9
    volumes:
      - ./dags:/opt/airflow/dags
      - ./data:/app/data
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////app/data/airflow.db
```

#### Production/Multi-User Deployment
```yaml
# kubernetes deployment for enterprise use
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quicknav-pipeline
spec:
  replicas: 2
  selector:
    matchLabels:
      app: quicknav-pipeline
  template:
    spec:
      containers:
      - name: quicknav-api
        image: quicknav:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: QUICKNAV_DB_PATH
          value: "/data/quicknav.duckdb"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: onedrive-volume
          mountPath: /onedrive
          readOnly: true
```

### 10. Cost Analysis and Scaling

#### Resource Requirements by Scale

| Scale | Users | Projects | Documents | Storage | Compute | Estimated Cost/Month |
|-------|-------|----------|-----------|---------|---------|---------------------|
| Small | 1-10 | <1K | <10K | 1GB | 2 vCPU, 4GB RAM | $0 (Local) |
| Medium | 10-100 | 1K-10K | 10K-100K | 10GB | 4 vCPU, 8GB RAM | $200-500 |
| Large | 100-1K | 10K-100K | 100K-1M | 100GB | 8 vCPU, 16GB RAM | $1K-2K |
| Enterprise | 1K+ | 100K+ | 1M+ | 1TB+ | 16+ vCPU, 32GB+ RAM | $5K+ |

#### Technology Scaling Path

1. **Phase 1 (Current)**: SQLite + JSON files
2. **Phase 2**: DuckDB + basic ETL
3. **Phase 3**: DuckDB + Airflow + streaming
4. **Phase 4**: DuckDB/PostgreSQL + Kafka + ML features
5. **Phase 5**: Distributed storage + Spark + real-time ML

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [x] Analyze existing codebase
- [ ] Set up DuckDB integration
- [ ] Basic ETL pipeline with Airflow
- [ ] Enhanced document parser integration

### Phase 2: Streaming (Weeks 3-4)
- [ ] User behavior tracking
- [ ] Real-time event processing
- [ ] Basic feature engineering
- [ ] Performance monitoring

### Phase 3: ML Integration (Weeks 5-6)
- [ ] Feature store implementation
- [ ] Document recommendation engine
- [ ] Search relevance scoring
- [ ] A/B testing framework

### Phase 4: Production (Weeks 7-8)
- [ ] Monitoring and alerting
- [ ] Production deployment
- [ ] Performance optimization
- [ ] Documentation and training

## Conclusion

This data pipeline architecture transforms Project QuickNav from a simple file navigation tool into a comprehensive, data-driven project management system. The design prioritizes:

1. **Backward Compatibility**: All existing functionality preserved
2. **Incremental Adoption**: Phased implementation allows gradual rollout
3. **Cross-Platform Support**: Works on Windows, macOS, and Linux
4. **Embedded-First**: DuckDB provides powerful analytics without complex infrastructure
5. **Scalability**: Architecture scales from single-user to enterprise deployment

The pipeline enables advanced features like ML-powered search, intelligent document recommendations, user behavior analytics, and predictive project insights while maintaining the simplicity and reliability of the existing system.