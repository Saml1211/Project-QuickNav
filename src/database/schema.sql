-- Project QuickNav Database Schema
-- Optimized for both transactional and analytical workloads
-- Primary target: DuckDB with SQLite fallback compatibility

-- Enable extensions (DuckDB specific)
-- INSTALL 'json';
-- LOAD 'json';

-- =====================================================
-- CORE ENTITY TABLES
-- =====================================================

-- Projects table: Core project information
CREATE TABLE IF NOT EXISTS projects (
    project_id VARCHAR PRIMARY KEY,           -- "17741"
    project_code VARCHAR NOT NULL,            -- "17741"
    project_name VARCHAR NOT NULL,            -- "Test Project"
    full_path TEXT NOT NULL,                  -- Full filesystem path
    range_folder VARCHAR,                     -- "17000 - 17999"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    metadata JSON,                            -- Flexible project metadata

    UNIQUE(project_code)
);

-- Documents table: Document metadata and analysis
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR PRIMARY KEY,          -- UUID or generated ID
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
    status_tags TEXT,                         -- JSON array as text: ["AS-BUILT", "SIGNED"]
    status_weight DECIMAL(4,2),              -- Calculated status priority

    -- Dates
    document_date DATE,                       -- Parsed from filename
    date_format VARCHAR,                      -- Format used for parsing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP,                    -- File system modification time
    last_accessed TIMESTAMP,

    -- Content analysis
    content_hash VARCHAR,                     -- For change detection
    word_count INTEGER,
    page_count INTEGER,
    content_preview TEXT,                     -- First 500 chars

    -- ML features (stored as JSON for SQLite compatibility)
    embedding_vector TEXT,                    -- JSON array of floats
    classification_confidence DECIMAL(3,2),   -- AI classification confidence

    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    UNIQUE (project_id, file_path)
);

-- =====================================================
-- USER ACTIVITY TRACKING
-- =====================================================

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL DEFAULT 'default_user',
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP,
    session_duration_seconds INTEGER,
    app_version VARCHAR,
    os_platform VARCHAR,

    -- Session metadata
    total_navigations INTEGER DEFAULT 0,
    total_searches INTEGER DEFAULT 0,
    ai_interactions INTEGER DEFAULT 0
);

-- User activities table (time-series data)
CREATE TABLE IF NOT EXISTS user_activities (
    activity_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    activity_type VARCHAR NOT NULL,           -- "navigate", "search", "ai_query", "document_open"

    -- Navigation context
    project_id VARCHAR,
    document_id VARCHAR,
    search_query TEXT,
    search_results_count INTEGER,

    -- Interaction details
    action_details TEXT,                      -- JSON as text
    response_time_ms INTEGER,                 -- Time to complete action
    success BOOLEAN DEFAULT true,
    error_message TEXT,

    -- User behavior
    input_method VARCHAR,                     -- "keyboard", "hotkey", "ai_chat"
    ui_component VARCHAR,                     -- "main_search", "ai_panel", "settings"

    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

-- =====================================================
-- USER PREFERENCES AND SETTINGS
-- =====================================================

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR PRIMARY KEY DEFAULT 'default_user',
    preferences TEXT NOT NULL,                -- JSON as text (complete settings)
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version VARCHAR DEFAULT '2.0.0',

    -- Quick access columns for common preferences
    theme VARCHAR DEFAULT 'system',
    default_mode VARCHAR DEFAULT 'folder',
    ai_enabled BOOLEAN DEFAULT true,
    hotkeys TEXT                              -- JSON as text
);

-- =====================================================
-- MACHINE LEARNING FEATURES
-- =====================================================

-- ML features table for recommendation and classification
CREATE TABLE IF NOT EXISTS ml_features (
    feature_id VARCHAR PRIMARY KEY,
    feature_type VARCHAR NOT NULL,            -- "user_item", "document_similarity", "temporal"
    entity_id VARCHAR NOT NULL,               -- project_id, document_id, or user_id
    feature_vector TEXT NOT NULL,             -- JSON array of floats
    feature_metadata TEXT,                    -- JSON metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- ML model tracking
    model_version VARCHAR,
    training_timestamp TIMESTAMP
);

-- =====================================================
-- ANALYTICS AND AGGREGATION
-- =====================================================

-- Analytics aggregates table for performance
CREATE TABLE IF NOT EXISTS analytics_aggregates (
    aggregate_id VARCHAR PRIMARY KEY,
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
    metric_details TEXT,                      -- JSON metadata

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (metric_name, time_period, date_bucket, COALESCE(project_id, ''), COALESCE(document_type, ''), COALESCE(user_id, ''))
);

-- =====================================================
-- AI CONVERSATION TRACKING
-- =====================================================

-- AI conversations table
CREATE TABLE IF NOT EXISTS ai_conversations (
    conversation_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,

    -- Conversation metadata
    total_messages INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    model_used VARCHAR,
    conversation_summary TEXT,

    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
);

-- AI messages table
CREATE TABLE IF NOT EXISTS ai_messages (
    message_id VARCHAR PRIMARY KEY,
    conversation_id VARCHAR NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR NOT NULL,                    -- "user", "assistant", "system", "tool"
    content TEXT,

    -- Tool calls (stored as JSON text)
    tool_calls TEXT,
    tool_call_id VARCHAR,
    tool_results TEXT,

    -- Metadata
    token_count INTEGER,
    response_time_ms INTEGER,
    model_temperature DECIMAL(3,2),

    FOREIGN KEY (conversation_id) REFERENCES ai_conversations(conversation_id)
);

-- =====================================================
-- SEARCH ANALYTICS
-- =====================================================

-- Search analytics table
CREATE TABLE IF NOT EXISTS search_analytics (
    search_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Search details
    query_text TEXT NOT NULL,
    query_type VARCHAR,                       -- "project_code", "project_name", "fuzzy"
    results_count INTEGER,
    selected_result_index INTEGER,

    -- Performance
    search_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT false,

    -- Results (stored as JSON text arrays)
    returned_projects TEXT,                   -- JSON array of project IDs
    returned_documents TEXT,                  -- JSON array of document IDs

    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Core entity indexes
CREATE INDEX IF NOT EXISTS idx_projects_code ON projects(project_code);
CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(project_name);
CREATE INDEX IF NOT EXISTS idx_projects_range ON projects(range_folder);
CREATE INDEX IF NOT EXISTS idx_projects_active ON projects(is_active);

-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_folder ON documents(folder_category);
CREATE INDEX IF NOT EXISTS idx_documents_version ON documents(version_numeric DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status_weight DESC);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(document_date DESC);
CREATE INDEX IF NOT EXISTS idx_documents_modified ON documents(modified_at DESC);

-- User session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON user_sessions(session_start);
CREATE INDEX IF NOT EXISTS idx_sessions_duration ON user_sessions(session_duration_seconds);

-- Time-series activity indexes (most important for performance)
CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON user_activities(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_activities_type_time ON user_activities(activity_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_activities_session ON user_activities(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_activities_project_time ON user_activities(project_id, timestamp DESC);

-- User behavior analysis
CREATE INDEX IF NOT EXISTS idx_user_behavior ON user_activities(session_id, activity_type, timestamp DESC);

-- Document ranking composite index
CREATE INDEX IF NOT EXISTS idx_document_ranking ON documents(project_id, document_type, status_weight DESC, version_numeric DESC);

-- ML feature indexes
CREATE INDEX IF NOT EXISTS idx_ml_features_type ON ml_features(feature_type);
CREATE INDEX IF NOT EXISTS idx_ml_features_entity ON ml_features(entity_id);
CREATE INDEX IF NOT EXISTS idx_ml_features_updated ON ml_features(updated_at DESC);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_aggregates_metric_date ON analytics_aggregates(metric_name, date_bucket DESC);
CREATE INDEX IF NOT EXISTS idx_aggregates_project_date ON analytics_aggregates(project_id, date_bucket DESC);

-- AI conversation indexes
CREATE INDEX IF NOT EXISTS idx_conversations_session ON ai_conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_started ON ai_conversations(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON ai_messages(conversation_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON ai_messages(timestamp DESC);

-- Search analytics indexes
CREATE INDEX IF NOT EXISTS idx_search_timestamp ON search_analytics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_search_query ON search_analytics(query_text);
CREATE INDEX IF NOT EXISTS idx_search_type ON search_analytics(query_type, timestamp DESC);

-- Preferences index
CREATE INDEX IF NOT EXISTS idx_prefs_updated ON user_preferences(last_updated);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Recent activity summary view
CREATE VIEW IF NOT EXISTS v_recent_activity AS
SELECT
    activity_type,
    COUNT(*) as count,
    AVG(response_time_ms) as avg_response_time,
    DATE(timestamp) as activity_date
FROM user_activities
WHERE timestamp > datetime('now', '-7 days')
GROUP BY activity_type, DATE(timestamp)
ORDER BY activity_date DESC, count DESC;

-- Popular projects view
CREATE VIEW IF NOT EXISTS v_popular_projects AS
SELECT
    p.project_code,
    p.project_name,
    COUNT(*) as access_count,
    MAX(ua.timestamp) as last_accessed,
    COUNT(DISTINCT ua.session_id) as unique_users
FROM user_activities ua
JOIN projects p ON ua.project_id = p.project_id
WHERE ua.timestamp > datetime('now', '-30 days')
    AND ua.activity_type IN ('navigate', 'document_open')
GROUP BY p.project_id, p.project_code, p.project_name
ORDER BY access_count DESC;

-- Latest documents by project and type
CREATE VIEW IF NOT EXISTS v_latest_documents AS
WITH ranked_docs AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY project_id, document_type
            ORDER BY
                COALESCE(status_weight, 0) DESC,
                COALESCE(version_numeric, 0) DESC,
                COALESCE(document_date, '1900-01-01') DESC,
                modified_at DESC
        ) as rn
    FROM documents
    WHERE is_active != 0 OR is_active IS NULL  -- Handle NULL as true
)
SELECT
    project_id,
    document_type,
    filename,
    version_string,
    status_weight,
    document_date,
    file_path
FROM ranked_docs
WHERE rn = 1;

-- User productivity metrics view
CREATE VIEW IF NOT EXISTS v_user_productivity AS
SELECT
    DATE(timestamp) as date,
    COUNT(DISTINCT session_id) as unique_users,
    COUNT(*) as total_activities,
    COUNT(DISTINCT project_id) as projects_accessed,
    AVG(response_time_ms) as avg_response_time,
    COUNT(CASE WHEN activity_type = 'navigate' THEN 1 END) as navigations,
    COUNT(CASE WHEN activity_type = 'search' THEN 1 END) as searches,
    COUNT(CASE WHEN activity_type = 'ai_query' THEN 1 END) as ai_queries
FROM user_activities
WHERE timestamp > datetime('now', '-30 days')
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- =====================================================
-- INITIAL DATA SETUP
-- =====================================================

-- Insert default user preferences if not exists
INSERT OR IGNORE INTO user_preferences (user_id, preferences, theme, default_mode, ai_enabled)
VALUES (
    'default_user',
    '{"version": "2.0.0", "initialized": true}',
    'system',
    'folder',
    true
);

-- Create initial analytics aggregates table structure verification
INSERT OR IGNORE INTO analytics_aggregates (
    aggregate_id, metric_name, time_period, date_bucket, metric_value, metric_count
) VALUES (
    'init_check', 'system_initialized', 'daily', DATE('now'), 1.0, 1
);