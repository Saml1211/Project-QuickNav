# Database Schema Design for Project QuickNav Analytics & ML

## Executive Summary

This document outlines an optimal database schema and storage strategy for Project QuickNav's analytics and ML features. The design supports both transactional (user preferences, history) and analytical workloads (usage patterns, ML features) using **DuckDB** as the primary embedded database with SQLite fallback.

## Storage Strategy Recommendation

### Primary: DuckDB
- **Analytical workloads**: Superior columnar storage and OLAP queries
- **Time-series data**: Excellent performance for temporal analytics
- **ML integration**: Native vector operations and efficient aggregations
- **Cross-platform**: Single-file embedded database
- **SQL compatibility**: PostgreSQL-like syntax with modern features

### Fallback: SQLite
- **Transactional workloads**: ACID compliance for user preferences
- **Simplicity**: Universal availability and reliability
- **Compatibility**: When DuckDB is unavailable

## Core Schema Design

### 1. Projects Table
```sql
CREATE TABLE projects (
    project_id VARCHAR PRIMARY KEY,           -- "17741"
    project_code VARCHAR NOT NULL,            -- "17741"
    project_name VARCHAR NOT NULL,            -- "Test Project"
    full_path TEXT NOT NULL,                  -- Full filesystem path
    range_folder VARCHAR,                     -- "17000 - 17999"
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    metadata JSON,                            -- Flexible project metadata

    -- Indexes
    INDEX idx_projects_code (project_code),
    INDEX idx_projects_name (project_name),
    INDEX idx_projects_range (range_folder),
    INDEX idx_projects_active (is_active)
);
```

### 2. Documents Table
```sql
CREATE TABLE documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR NOT NULL,
    file_path TEXT NOT NULL,
    filename VARCHAR NOT NULL,
    file_extension VARCHAR,
    file_size_bytes BIGINT,
    document_type VARCHAR,                    -- "lld", "hld", "co", "floor_plan"
    folder_category VARCHAR,                  -- "System Designs", "Sales Handover"

    -- Document metadata (from DocumentParser)
    version_string VARCHAR,                   -- Original version string
    version_numeric INTEGER,                  -- Normalized version for sorting
    version_type VARCHAR,                     -- "rev_numeric", "period", "letter"
    status_tags VARCHAR[],                    -- ["AS-BUILT", "SIGNED"]
    status_weight DECIMAL(4,2),              -- Calculated status priority

    -- Dates
    document_date DATE,                       -- Parsed from filename
    date_format VARCHAR,                      -- Format used for parsing
    created_at TIMESTAMP DEFAULT NOW(),
    modified_at TIMESTAMP,                    -- File system modification time
    last_accessed TIMESTAMP,

    -- Content analysis
    content_hash VARCHAR,                     -- For change detection
    word_count INTEGER,
    page_count INTEGER,
    content_preview TEXT,                     -- First 500 chars

    -- ML features
    embedding_vector FLOAT[],                 -- Document embedding (1536 dims)
    classification_confidence DECIMAL(3,2),   -- AI classification confidence

    FOREIGN KEY (project_id) REFERENCES projects(project_id),

    -- Indexes
    INDEX idx_documents_project (project_id),
    INDEX idx_documents_type (document_type),
    INDEX idx_documents_folder (folder_category),
    INDEX idx_documents_version (version_numeric DESC),
    INDEX idx_documents_status (status_weight DESC),
    INDEX idx_documents_date (document_date DESC),
    INDEX idx_documents_modified (modified_at DESC),
    UNIQUE (project_id, file_path)
);
```

### 3. User Sessions Table
```sql
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL DEFAULT 'default_user',
    session_start TIMESTAMP DEFAULT NOW(),
    session_end TIMESTAMP,
    session_duration_seconds INTEGER,
    app_version VARCHAR,
    os_platform VARCHAR,

    -- Session metadata
    total_navigations INTEGER DEFAULT 0,
    total_searches INTEGER DEFAULT 0,
    ai_interactions INTEGER DEFAULT 0,

    INDEX idx_sessions_user (user_id),
    INDEX idx_sessions_start (session_start),
    INDEX idx_sessions_duration (session_duration_seconds)
);
```

### 4. User Activities Table (Time-series)
```sql
CREATE TABLE user_activities (
    activity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    activity_type VARCHAR NOT NULL,           -- "navigate", "search", "ai_query", "document_open"

    -- Navigation context
    project_id VARCHAR,
    document_id UUID,
    search_query TEXT,
    search_results_count INTEGER,

    -- Interaction details
    action_details JSON,                      -- Flexible activity data
    response_time_ms INTEGER,                 -- Time to complete action
    success BOOLEAN DEFAULT true,
    error_message TEXT,

    -- User behavior
    input_method VARCHAR,                     -- "keyboard", "hotkey", "ai_chat"
    ui_component VARCHAR,                     -- "main_search", "ai_panel", "settings"

    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id),

    -- Time-series optimized indexes
    INDEX idx_activities_timestamp (timestamp DESC),
    INDEX idx_activities_type_time (activity_type, timestamp DESC),
    INDEX idx_activities_session (session_id, timestamp DESC),
    INDEX idx_activities_project_time (project_id, timestamp DESC)
);
```

### 5. User Preferences Table
```sql
CREATE TABLE user_preferences (
    user_id VARCHAR PRIMARY KEY DEFAULT 'default_user',
    preferences JSON NOT NULL,                -- Complete settings from SettingsManager
    last_updated TIMESTAMP DEFAULT NOW(),
    version VARCHAR DEFAULT '2.0.0',

    -- Quick access columns for common preferences
    theme VARCHAR DEFAULT 'system',
    default_mode VARCHAR DEFAULT 'folder',
    ai_enabled BOOLEAN DEFAULT true,
    hotkeys JSON,

    INDEX idx_prefs_updated (last_updated)
);
```

### 6. ML Features Table
```sql
CREATE TABLE ml_features (
    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_type VARCHAR NOT NULL,            -- "user_item", "document_similarity", "temporal"
    entity_id VARCHAR NOT NULL,               -- project_id, document_id, or user_id
    feature_vector FLOAT[] NOT NULL,
    feature_metadata JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- ML model tracking
    model_version VARCHAR,
    training_timestamp TIMESTAMP,

    INDEX idx_ml_features_type (feature_type),
    INDEX idx_ml_features_entity (entity_id),
    INDEX idx_ml_features_updated (updated_at DESC)
);
```

### 7. Analytics Aggregates Table
```sql
CREATE TABLE analytics_aggregates (
    aggregate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR NOT NULL,             -- "daily_project_access", "popular_documents"
    time_period VARCHAR NOT NULL,             -- "daily", "weekly", "monthly"
    date_bucket DATE NOT NULL,                -- Aggregation date

    -- Dimensions
    project_id VARCHAR,
    document_type VARCHAR,
    user_id VARCHAR,

    -- Metrics
    metric_value DECIMAL(12,2),
    metric_count INTEGER,
    metric_details JSON,

    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_aggregates_metric_date (metric_name, date_bucket DESC),
    INDEX idx_aggregates_project_date (project_id, date_bucket DESC),
    UNIQUE (metric_name, time_period, date_bucket, project_id, document_type, user_id)
);
```

## Advanced Features

### 8. AI Conversations Table
```sql
CREATE TABLE ai_conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,

    -- Conversation metadata
    total_messages INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    model_used VARCHAR,
    conversation_summary TEXT,

    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id),
    INDEX idx_conversations_session (session_id),
    INDEX idx_conversations_started (started_at DESC)
);

CREATE TABLE ai_messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    role VARCHAR NOT NULL,                    -- "user", "assistant", "system", "tool"
    content TEXT,

    -- Tool calls
    tool_calls JSON,
    tool_call_id VARCHAR,
    tool_results JSON,

    -- Metadata
    token_count INTEGER,
    response_time_ms INTEGER,
    model_temperature DECIMAL(3,2),

    FOREIGN KEY (conversation_id) REFERENCES ai_conversations(conversation_id),
    INDEX idx_messages_conversation (conversation_id, timestamp),
    INDEX idx_messages_timestamp (timestamp DESC)
);
```

### 9. Search Analytics Table
```sql
CREATE TABLE search_analytics (
    search_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),

    -- Search details
    query_text TEXT NOT NULL,
    query_type VARCHAR,                       -- "project_code", "project_name", "fuzzy"
    results_count INTEGER,
    selected_result_index INTEGER,

    -- Performance
    search_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT false,

    -- Results
    returned_projects VARCHAR[],              -- Array of project IDs
    returned_documents UUID[],                -- Array of document IDs

    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id),
    INDEX idx_search_timestamp (timestamp DESC),
    INDEX idx_search_query (query_text),
    INDEX idx_search_type (query_type, timestamp DESC)
);
```

## Indexing Strategy

### Performance Indexes
```sql
-- Time-series queries (most common)
CREATE INDEX idx_activities_time_series ON user_activities
    (timestamp DESC, activity_type, project_id);

-- User behavior analysis
CREATE INDEX idx_user_behavior ON user_activities
    (session_id, activity_type, timestamp DESC);

-- Document ranking
CREATE INDEX idx_document_ranking ON documents
    (project_id, document_type, status_weight DESC, version_numeric DESC);

-- Popular projects
CREATE INDEX idx_project_popularity ON user_activities
    (project_id, timestamp DESC)
    WHERE activity_type IN ('navigate', 'document_open');

-- ML feature lookups
CREATE INDEX idx_ml_entity_features ON ml_features
    (entity_id, feature_type, updated_at DESC);
```

### Specialized Indexes for Analytics
```sql
-- Document lifecycle analysis
CREATE INDEX idx_document_lifecycle ON documents
    (document_type, created_at, version_numeric);

-- User productivity metrics
CREATE INDEX idx_user_productivity ON user_activities
    (timestamp::date, activity_type, response_time_ms);

-- AI usage patterns
CREATE INDEX idx_ai_usage ON ai_messages
    (timestamp::date, role, token_count);
```

## Data Archival and Retention Policies

### Hot Data (Active)
- **User sessions**: Last 90 days
- **User activities**: Last 30 days (detailed)
- **AI conversations**: Last 60 days
- **Search analytics**: Last 45 days

### Warm Data (Compressed)
- **Aggregated metrics**: 2 years
- **Document metadata**: All time (lightweight)
- **Project data**: All time
- **ML features**: Last 6 months

### Cold Data (Archived)
- **Raw activities**: >30 days → Aggregate and archive
- **AI message content**: >60 days → Summarize and archive
- **Search logs**: >45 days → Statistical summary only

### Implementation
```sql
-- Archival procedures
CREATE TABLE user_activities_archive AS SELECT * FROM user_activities WHERE false;

-- Archive old activities (run daily)
INSERT INTO user_activities_archive
SELECT * FROM user_activities
WHERE timestamp < NOW() - INTERVAL '30 days';

DELETE FROM user_activities
WHERE timestamp < NOW() - INTERVAL '30 days';

-- Cleanup old AI conversations
DELETE FROM ai_messages
WHERE timestamp < NOW() - INTERVAL '60 days';
```

## Query Optimization Patterns

### 1. User Dashboard Queries
```sql
-- Recent activity summary
SELECT
    activity_type,
    COUNT(*) as count,
    AVG(response_time_ms) as avg_response_time
FROM user_activities
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY activity_type
ORDER BY count DESC;

-- Most accessed projects (last 30 days)
SELECT
    p.project_code,
    p.project_name,
    COUNT(*) as access_count,
    MAX(ua.timestamp) as last_accessed
FROM user_activities ua
JOIN projects p ON ua.project_id = p.project_id
WHERE ua.timestamp > NOW() - INTERVAL '30 days'
    AND ua.activity_type IN ('navigate', 'document_open')
GROUP BY p.project_id, p.project_code, p.project_name
ORDER BY access_count DESC
LIMIT 10;
```

### 2. Document Intelligence Queries
```sql
-- Find latest documents by type and project
WITH ranked_docs AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY project_id, document_type
            ORDER BY status_weight DESC, version_numeric DESC, document_date DESC
        ) as rn
    FROM documents
    WHERE is_active = true
)
SELECT * FROM ranked_docs WHERE rn = 1;

-- Document version timeline
SELECT
    filename,
    version_string,
    version_numeric,
    document_date,
    status_tags
FROM documents
WHERE project_id = ?
    AND document_type = ?
ORDER BY version_numeric DESC, document_date DESC;
```

### 3. ML Feature Queries
```sql
-- User-item interaction matrix for recommendations
SELECT
    session_id as user_proxy,
    project_id,
    COUNT(*) as interaction_count,
    SUM(CASE WHEN activity_type = 'document_open' THEN 2 ELSE 1 END) as weighted_score,
    MAX(timestamp) as last_interaction
FROM user_activities
WHERE timestamp > NOW() - INTERVAL '90 days'
    AND project_id IS NOT NULL
GROUP BY session_id, project_id;

-- Document similarity features
SELECT
    d1.document_id,
    d2.document_id,
    cosine_similarity(d1.embedding_vector, d2.embedding_vector) as similarity
FROM documents d1
CROSS JOIN documents d2
WHERE d1.project_id = d2.project_id
    AND d1.document_id != d2.document_id
    AND d1.document_type = d2.document_type;
```

### 4. Analytics Aggregation Queries
```sql
-- Daily usage metrics (for real-time dashboard)
SELECT
    DATE(timestamp) as date,
    COUNT(DISTINCT session_id) as unique_users,
    COUNT(*) as total_activities,
    COUNT(DISTINCT project_id) as projects_accessed,
    AVG(response_time_ms) as avg_response_time
FROM user_activities
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Project popularity trends
SELECT
    p.project_code,
    p.project_name,
    DATE_TRUNC('week', ua.timestamp) as week,
    COUNT(*) as weekly_access_count
FROM user_activities ua
JOIN projects p ON ua.project_id = p.project_id
WHERE ua.timestamp > NOW() - INTERVAL '12 weeks'
    AND ua.activity_type IN ('navigate', 'document_open')
GROUP BY p.project_id, p.project_code, p.project_name, DATE_TRUNC('week', ua.timestamp)
ORDER BY p.project_code, week DESC;
```

## Implementation Example

### DuckDB Setup
```python
import duckdb
import json
from pathlib import Path

class QuickNavDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use platform-appropriate data directory
            if os.name == 'nt':  # Windows
                data_dir = Path(os.environ.get('APPDATA', '')) / 'QuickNav'
            else:  # Unix-like
                data_dir = Path.home() / '.local' / 'share' / 'quicknav'

            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / 'quicknav.duckdb'

        self.conn = duckdb.connect(str(db_path))
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize database schema with all tables and indexes."""
        schema_sql = Path(__file__).parent / 'schema.sql'
        if schema_sql.exists():
            self.conn.execute(schema_sql.read_text())

    def record_activity(self, session_id: str, activity_type: str,
                       project_id: str = None, **kwargs):
        """Record a user activity."""
        self.conn.execute("""
            INSERT INTO user_activities
            (session_id, activity_type, project_id, action_details, response_time_ms)
            VALUES (?, ?, ?, ?, ?)
        """, [
            session_id,
            activity_type,
            project_id,
            json.dumps(kwargs.get('details', {})),
            kwargs.get('response_time_ms')
        ])

    def get_popular_projects(self, days: int = 30, limit: int = 10):
        """Get most popular projects in the last N days."""
        return self.conn.execute("""
            SELECT
                p.project_code,
                p.project_name,
                COUNT(*) as access_count,
                MAX(ua.timestamp) as last_accessed
            FROM user_activities ua
            JOIN projects p ON ua.project_id = p.project_id
            WHERE ua.timestamp > NOW() - INTERVAL '{} days'
                AND ua.activity_type IN ('navigate', 'document_open')
            GROUP BY p.project_id, p.project_code, p.project_name
            ORDER BY access_count DESC
            LIMIT {}
        """.format(days, limit)).fetchall()

    def update_document_embedding(self, document_id: str, embedding: List[float]):
        """Update document embedding for ML features."""
        self.conn.execute("""
            UPDATE documents
            SET embedding_vector = ?, updated_at = NOW()
            WHERE document_id = ?
        """, [embedding, document_id])
```

## Performance Benchmarks

Expected performance characteristics:

- **Document search**: <50ms for 10,000+ documents
- **Activity logging**: <5ms per insert
- **Popular projects query**: <100ms over 90 days of data
- **ML feature extraction**: <200ms for user-item matrix
- **Analytics aggregation**: <500ms for monthly reports

## Migration Strategy

1. **Phase 1**: Implement core tables (projects, documents, user_activities)
2. **Phase 2**: Add analytics and ML features
3. **Phase 3**: Implement archival and optimization
4. **Phase 4**: Add advanced AI conversation tracking

This schema provides a solid foundation for both operational needs and advanced analytics while maintaining performance and scalability.