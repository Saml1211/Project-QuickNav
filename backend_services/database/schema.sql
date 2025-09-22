-- Database Schema for Project QuickNav Backend Services
-- This schema supports the analytics, recommendation, and search services
-- with proper indexing, partitioning, and performance optimizations

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create separate databases for each service (run individually)
-- CREATE DATABASE quicknav_analytics;
-- CREATE DATABASE quicknav_recommendations;
-- CREATE DATABASE quicknav_search;

-- ================================================================
-- ANALYTICS SERVICE SCHEMA
-- ================================================================

-- User events table for tracking all user interactions
CREATE TABLE user_events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    project_id VARCHAR(20),
    document_id VARCHAR(255),
    document_path TEXT,
    query TEXT,
    result_count INTEGER,
    response_time_ms REAL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',

    -- Indexes for performance
    CONSTRAINT user_events_event_type_check CHECK (event_type IN (
        'project_view', 'project_search', 'document_access', 'document_search',
        'user_login', 'recommendation_click', 'navigation', 'export', 'favorite'
    ))
);

-- Partition by timestamp for better performance
CREATE TABLE user_events_2024 PARTITION OF user_events
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE user_events_2025 PARTITION OF user_events
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Indexes for user_events
CREATE INDEX idx_user_events_user_id ON user_events (user_id);
CREATE INDEX idx_user_events_timestamp ON user_events (timestamp);
CREATE INDEX idx_user_events_event_type ON user_events (event_type);
CREATE INDEX idx_user_events_project_id ON user_events (project_id) WHERE project_id IS NOT NULL;
CREATE INDEX idx_user_events_composite ON user_events (user_id, event_type, timestamp);
CREATE INDEX idx_user_events_query_gin ON user_events USING GIN (query gin_trgm_ops) WHERE query IS NOT NULL;
CREATE INDEX idx_user_events_metadata_gin ON user_events USING GIN (metadata);

-- Project metrics aggregation table
CREATE TABLE project_metrics (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(20) UNIQUE NOT NULL,
    project_name VARCHAR(500),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    unique_users INTEGER DEFAULT 0,
    document_count INTEGER DEFAULT 0,
    total_size_mb REAL DEFAULT 0.0,
    average_response_time_ms REAL DEFAULT 0.0,
    popularity_score REAL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for project_metrics
CREATE INDEX idx_project_metrics_project_id ON project_metrics (project_id);
CREATE INDEX idx_project_metrics_popularity ON project_metrics (popularity_score DESC);
CREATE INDEX idx_project_metrics_access_count ON project_metrics (access_count DESC);
CREATE INDEX idx_project_metrics_last_accessed ON project_metrics (last_accessed DESC);

-- Dashboard metrics for pre-computed aggregations
CREATE TABLE dashboard_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_type VARCHAR(20) NOT NULL, -- daily, weekly, monthly
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB DEFAULT '{}',

    CONSTRAINT dashboard_metrics_type_check CHECK (metric_type IN ('hourly', 'daily', 'weekly', 'monthly'))
);

-- Partition dashboard_metrics by date
CREATE TABLE dashboard_metrics_2024 PARTITION OF dashboard_metrics
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE dashboard_metrics_2025 PARTITION OF dashboard_metrics
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Indexes for dashboard_metrics
CREATE INDEX idx_dashboard_metrics_name_date ON dashboard_metrics (metric_name, date);
CREATE INDEX idx_dashboard_metrics_type_date ON dashboard_metrics (metric_type, date);
CREATE UNIQUE INDEX idx_dashboard_metrics_unique ON dashboard_metrics (metric_name, metric_type, date);

-- Performance monitoring table
CREATE TABLE performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    labels JSONB DEFAULT '{}'
);

-- Partition performance_metrics by timestamp
CREATE TABLE performance_metrics_2024 PARTITION OF performance_metrics
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE performance_metrics_2025 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Indexes for performance_metrics
CREATE INDEX idx_performance_metrics_service ON performance_metrics (service_name, timestamp);
CREATE INDEX idx_performance_metrics_name ON performance_metrics (metric_name, timestamp);
CREATE INDEX idx_performance_metrics_labels_gin ON performance_metrics USING GIN (labels);

-- ================================================================
-- RECOMMENDATION SERVICE SCHEMA
-- ================================================================

-- User profiles for personalization
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) UNIQUE NOT NULL,
    preferences JSONB DEFAULT '{}',
    interaction_count INTEGER DEFAULT 0,
    favorite_projects JSONB DEFAULT '[]',
    preferred_doc_types JSONB DEFAULT '[]',
    work_hours JSONB DEFAULT '{}', -- Peak activity hours pattern
    behavior_vector JSONB DEFAULT '[]', -- ML features
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user_profiles
CREATE INDEX idx_user_profiles_user_id ON user_profiles (user_id);
CREATE INDEX idx_user_profiles_interaction_count ON user_profiles (interaction_count DESC);
CREATE INDEX idx_user_profiles_preferences_gin ON user_profiles USING GIN (preferences);

-- Project profiles for content-based recommendations
CREATE TABLE project_profiles (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(20) UNIQUE NOT NULL,
    project_name VARCHAR(500),
    keywords JSONB DEFAULT '[]',
    categories JSONB DEFAULT '[]',
    popularity_score REAL DEFAULT 0.0,
    content_vector JSONB DEFAULT '[]', -- TF-IDF or embedding features
    similar_projects JSONB DEFAULT '[]', -- Pre-computed similar projects
    document_types JSONB DEFAULT '[]', -- Available document types
    last_analyzed TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for project_profiles
CREATE INDEX idx_project_profiles_project_id ON project_profiles (project_id);
CREATE INDEX idx_project_profiles_popularity ON project_profiles (popularity_score DESC);
CREATE INDEX idx_project_profiles_keywords_gin ON project_profiles USING GIN (keywords);
CREATE INDEX idx_project_profiles_categories_gin ON project_profiles USING GIN (categories);

-- User-project interactions for collaborative filtering
CREATE TABLE user_project_interactions (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    project_id VARCHAR(20) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    interaction_score REAL DEFAULT 1.0,
    context JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT user_interactions_type_check CHECK (interaction_type IN (
        'view', 'search', 'document_access', 'favorite', 'share', 'export', 'bookmark'
    ))
);

-- Partition by timestamp for better performance
CREATE TABLE user_project_interactions_2024 PARTITION OF user_project_interactions
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE user_project_interactions_2025 PARTITION OF user_project_interactions
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Indexes for user_project_interactions
CREATE INDEX idx_user_interactions_user_id ON user_project_interactions (user_id, timestamp DESC);
CREATE INDEX idx_user_interactions_project_id ON user_project_interactions (project_id, timestamp DESC);
CREATE INDEX idx_user_interactions_type ON user_project_interactions (interaction_type);
CREATE INDEX idx_user_interactions_score ON user_project_interactions (interaction_score DESC);
CREATE INDEX idx_user_interactions_composite ON user_project_interactions (user_id, project_id, interaction_type);

-- Recommendation cache for performance
CREATE TABLE recommendation_cache (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    recommendation_type VARCHAR(50) NOT NULL,
    recommendations JSONB NOT NULL,
    context_hash VARCHAR(64) NOT NULL,
    confidence_score REAL DEFAULT 0.0,
    algorithm_used VARCHAR(100),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT recommendation_cache_type_check CHECK (recommendation_type IN (
        'projects', 'documents', 'similar_projects', 'trending'
    ))
);

-- Indexes for recommendation_cache
CREATE INDEX idx_recommendation_cache_user ON recommendation_cache (user_id, recommendation_type);
CREATE INDEX idx_recommendation_cache_context ON recommendation_cache (context_hash);
CREATE INDEX idx_recommendation_cache_expires ON recommendation_cache (expires_at);

-- User similarity matrix for collaborative filtering
CREATE TABLE user_similarity (
    id SERIAL PRIMARY KEY,
    user_id_1 VARCHAR(100) NOT NULL,
    user_id_2 VARCHAR(100) NOT NULL,
    similarity_score REAL NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT user_similarity_users_different CHECK (user_id_1 != user_id_2),
    CONSTRAINT user_similarity_score_range CHECK (similarity_score >= 0 AND similarity_score <= 1)
);

-- Indexes for user_similarity
CREATE INDEX idx_user_similarity_user1 ON user_similarity (user_id_1, similarity_score DESC);
CREATE INDEX idx_user_similarity_user2 ON user_similarity (user_id_2, similarity_score DESC);
CREATE UNIQUE INDEX idx_user_similarity_pair ON user_similarity (LEAST(user_id_1, user_id_2), GREATEST(user_id_1, user_id_2));

-- ================================================================
-- SEARCH SERVICE SCHEMA
-- ================================================================

-- Search index for full-text search capabilities
CREATE TABLE search_index (
    id BIGSERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    project_id VARCHAR(20),
    title TEXT NOT NULL,
    content TEXT,
    path TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    keywords JSONB DEFAULT '[]',
    categories JSONB DEFAULT '[]',
    popularity_score REAL DEFAULT 0.0,
    search_count INTEGER DEFAULT 0,
    last_indexed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    content_hash VARCHAR(64), -- For detecting changes

    CONSTRAINT search_index_type_check CHECK (document_type IN (
        'project', 'document', 'folder', 'image', 'video'
    ))
);

-- Indexes for search_index (full-text search capabilities)
CREATE INDEX idx_search_index_document_id ON search_index (document_id);
CREATE INDEX idx_search_index_type ON search_index (document_type);
CREATE INDEX idx_search_index_project_id ON search_index (project_id) WHERE project_id IS NOT NULL;
CREATE INDEX idx_search_index_popularity ON search_index (popularity_score DESC);
CREATE INDEX idx_search_index_title_gin ON search_index USING GIN (to_tsvector('english', title));
CREATE INDEX idx_search_index_content_gin ON search_index USING GIN (to_tsvector('english', content)) WHERE content IS NOT NULL;
CREATE INDEX idx_search_index_keywords_gin ON search_index USING GIN (keywords);
CREATE INDEX idx_search_index_categories_gin ON search_index USING GIN (categories);
CREATE INDEX idx_search_index_metadata_gin ON search_index USING GIN (metadata);
CREATE INDEX idx_search_index_path_gin ON search_index USING GIN (path gin_trgm_ops);

-- Search queries table for analytics and suggestions
CREATE TABLE search_queries (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    user_id VARCHAR(100),
    results_count INTEGER NOT NULL,
    response_time_ms REAL NOT NULL,
    filters_used JSONB DEFAULT '{}',
    clicked_results JSONB DEFAULT '[]',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Partition search_queries by timestamp
CREATE TABLE search_queries_2024 PARTITION OF search_queries
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE search_queries_2025 PARTITION OF search_queries
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Indexes for search_queries
CREATE INDEX idx_search_queries_hash ON search_queries (query_hash);
CREATE INDEX idx_search_queries_user ON search_queries (user_id, timestamp DESC) WHERE user_id IS NOT NULL;
CREATE INDEX idx_search_queries_text_gin ON search_queries USING GIN (query_text gin_trgm_ops);
CREATE INDEX idx_search_queries_timestamp ON search_queries (timestamp);
CREATE INDEX idx_search_queries_results ON search_queries (results_count);

-- Document relationships for enhanced search
CREATE TABLE document_relationships (
    id SERIAL PRIMARY KEY,
    source_document_id VARCHAR(255) NOT NULL,
    target_document_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    strength REAL DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT doc_relationships_type_check CHECK (relationship_type IN (
        'similar', 'related', 'version', 'supersedes', 'references', 'contains'
    )),
    CONSTRAINT doc_relationships_strength_check CHECK (strength >= 0 AND strength <= 1),
    CONSTRAINT doc_relationships_different CHECK (source_document_id != target_document_id)
);

-- Indexes for document_relationships
CREATE INDEX idx_doc_relationships_source ON document_relationships (source_document_id, strength DESC);
CREATE INDEX idx_doc_relationships_target ON document_relationships (target_document_id, strength DESC);
CREATE INDEX idx_doc_relationships_type ON document_relationships (relationship_type);
CREATE UNIQUE INDEX idx_doc_relationships_unique ON document_relationships (source_document_id, target_document_id, relationship_type);

-- Search suggestions for autocomplete
CREATE TABLE search_suggestions (
    id SERIAL PRIMARY KEY,
    suggestion_text VARCHAR(200) NOT NULL,
    suggestion_type VARCHAR(50) NOT NULL,
    frequency INTEGER DEFAULT 1,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',

    CONSTRAINT search_suggestions_type_check CHECK (suggestion_type IN (
        'query', 'project', 'document_type', 'category', 'keyword'
    ))
);

-- Indexes for search_suggestions
CREATE INDEX idx_search_suggestions_text_gin ON search_suggestions USING GIN (suggestion_text gin_trgm_ops);
CREATE INDEX idx_search_suggestions_type ON search_suggestions (suggestion_type);
CREATE INDEX idx_search_suggestions_frequency ON search_suggestions (frequency DESC);
CREATE UNIQUE INDEX idx_search_suggestions_unique ON search_suggestions (suggestion_text, suggestion_type);

-- ================================================================
-- SHARED TABLES (across all services)
-- ================================================================

-- System configuration table
CREATE TABLE system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for system_config
CREATE INDEX idx_system_config_key ON system_config (config_key);

-- API rate limiting table
CREATE TABLE rate_limits (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(100) NOT NULL, -- user_id, ip_address, api_key
    identifier_type VARCHAR(20) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    request_count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    window_size_seconds INTEGER NOT NULL,
    limit_per_window INTEGER NOT NULL,

    CONSTRAINT rate_limits_type_check CHECK (identifier_type IN ('user', 'ip', 'api_key'))
);

-- Indexes for rate_limits
CREATE INDEX idx_rate_limits_identifier ON rate_limits (identifier, endpoint, window_start);
CREATE INDEX idx_rate_limits_window ON rate_limits (window_start);

-- Audit log for important actions
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(100),
    old_values JSONB,
    new_values JSONB,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Partition audit_log by timestamp
CREATE TABLE audit_log_2024 PARTITION OF audit_log
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE audit_log_2025 PARTITION OF audit_log
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Indexes for audit_log
CREATE INDEX idx_audit_log_entity ON audit_log (entity_type, entity_id);
CREATE INDEX idx_audit_log_user ON audit_log (user_id, timestamp DESC) WHERE user_id IS NOT NULL;
CREATE INDEX idx_audit_log_action ON audit_log (action, timestamp DESC);
CREATE INDEX idx_audit_log_timestamp ON audit_log (timestamp);

-- ================================================================
-- VIEWS FOR COMMON QUERIES
-- ================================================================

-- User activity summary view
CREATE VIEW user_activity_summary AS
SELECT
    user_id,
    COUNT(*) as total_events,
    COUNT(DISTINCT project_id) as unique_projects,
    COUNT(DISTINCT DATE(timestamp)) as active_days,
    MAX(timestamp) as last_activity,
    AVG(response_time_ms) as avg_response_time
FROM user_events
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY user_id;

-- Project popularity view
CREATE VIEW project_popularity AS
SELECT
    p.project_id,
    p.project_name,
    p.access_count,
    p.unique_users,
    p.popularity_score,
    COUNT(e.id) as recent_events,
    MAX(e.timestamp) as last_event
FROM project_metrics p
LEFT JOIN user_events e ON p.project_id = e.project_id
    AND e.timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY p.project_id, p.project_name, p.access_count, p.unique_users, p.popularity_score
ORDER BY p.popularity_score DESC;

-- Search performance view
CREATE VIEW search_performance AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_searches,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    AVG(results_count) as avg_results,
    COUNT(DISTINCT user_id) as unique_users
FROM search_queries
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- ================================================================
-- FUNCTIONS AND TRIGGERS
-- ================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_project_metrics_updated_at
    BEFORE UPDATE ON project_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_profiles_updated_at
    BEFORE UPDATE ON project_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at
    BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate project popularity score
CREATE OR REPLACE FUNCTION calculate_popularity_score(
    access_count INTEGER,
    unique_users INTEGER,
    recent_activity INTEGER DEFAULT 0,
    days_since_last_access INTEGER DEFAULT 0
) RETURNS REAL AS $$
BEGIN
    RETURN (
        (access_count * 0.3) +
        (unique_users * 1.0) +
        (recent_activity * 0.5) -
        (days_since_last_access * 0.1)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    cutoff_date TIMESTAMP WITH TIME ZONE;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (days_to_keep || ' days')::INTERVAL;

    -- Clean old user events
    DELETE FROM user_events WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Clean old search queries
    DELETE FROM search_queries WHERE timestamp < cutoff_date;

    -- Clean old performance metrics
    DELETE FROM performance_metrics WHERE timestamp < cutoff_date;

    -- Clean expired recommendation cache
    DELETE FROM recommendation_cache WHERE expires_at < CURRENT_TIMESTAMP;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- INITIAL DATA AND CONFIGURATION
-- ================================================================

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('analytics.retention_days', '90', 'Number of days to retain analytics data'),
('search.max_results', '100', 'Maximum search results per query'),
('recommendations.cache_ttl_minutes', '30', 'Recommendation cache TTL in minutes'),
('rate_limit.default_requests_per_minute', '60', 'Default rate limit per minute'),
('elasticsearch.refresh_interval', '30s', 'Elasticsearch index refresh interval')
ON CONFLICT (config_key) DO NOTHING;

-- ================================================================
-- PERFORMANCE OPTIMIZATION PROCEDURES
-- ================================================================

-- Procedure to update table statistics
CREATE OR REPLACE FUNCTION update_table_statistics()
RETURNS VOID AS $$
BEGIN
    -- Analyze frequently queried tables
    ANALYZE user_events;
    ANALYZE project_metrics;
    ANALYZE user_project_interactions;
    ANALYZE search_index;
    ANALYZE search_queries;

    RAISE NOTICE 'Table statistics updated successfully';
END;
$$ LANGUAGE plpgsql;

-- Procedure to reindex tables for performance
CREATE OR REPLACE FUNCTION reindex_performance_critical_tables()
RETURNS VOID AS $$
BEGIN
    -- Reindex GIN indexes that may become fragmented
    REINDEX INDEX CONCURRENTLY idx_user_events_metadata_gin;
    REINDEX INDEX CONCURRENTLY idx_search_index_title_gin;
    REINDEX INDEX CONCURRENTLY idx_search_index_content_gin;
    REINDEX INDEX CONCURRENTLY idx_search_queries_text_gin;

    RAISE NOTICE 'Performance critical indexes rebuilt successfully';
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- MONITORING AND MAINTENANCE
-- ================================================================

-- View for monitoring table sizes
CREATE VIEW table_sizes AS
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- View for monitoring index usage
CREATE VIEW index_usage AS
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- View for slow queries (requires pg_stat_statements extension)
-- CREATE VIEW slow_queries AS
-- SELECT
--     query,
--     calls,
--     total_time,
--     mean_time,
--     rows
-- FROM pg_stat_statements
-- WHERE mean_time > 1000 -- queries taking more than 1 second
-- ORDER BY mean_time DESC;

COMMENT ON SCHEMA public IS 'Project QuickNav Backend Services Database Schema';
COMMENT ON TABLE user_events IS 'Tracks all user interactions for analytics';
COMMENT ON TABLE project_metrics IS 'Aggregated project-level metrics and statistics';
COMMENT ON TABLE user_profiles IS 'User profiles for personalized recommendations';
COMMENT ON TABLE search_index IS 'Full-text search index for all documents';
COMMENT ON TABLE recommendation_cache IS 'Cached recommendations for performance';
COMMENT ON FUNCTION calculate_popularity_score IS 'Calculates project popularity score based on multiple factors';
COMMENT ON FUNCTION cleanup_old_data IS 'Removes old data beyond retention period';