# Project QuickNav Backend Services Architecture

## Overview

This document outlines the comprehensive backend services architecture for Project QuickNav, designed to support data-driven features including analytics, intelligent recommendations, enhanced search capabilities, and predictive navigation.

## Architecture Principles

### 1. Microservices Design
- **Service Separation**: Each service handles a specific domain (Analytics, Recommendations, Search, etc.)
- **Independent Deployment**: Services can be deployed and scaled independently
- **Technology Diversity**: Each service can use the most appropriate technology stack
- **Fault Isolation**: Failures in one service don't cascade to others

### 2. API-First Design
- **Standardized Contracts**: All services expose RESTful APIs with consistent patterns
- **OpenAPI Documentation**: Comprehensive API documentation for all endpoints
- **Versioning Strategy**: Semantic versioning with backward compatibility
- **Error Handling**: Standardized error responses across all services

### 3. Data Consistency & CQRS
- **Command Query Responsibility Segregation**: Separate read and write operations
- **Event Sourcing**: Track all changes as immutable events
- **Eventual Consistency**: Accept temporary inconsistency for better performance
- **Compensating Actions**: Handle distributed transaction failures gracefully

## Service Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        GUI[Tkinter GUI]
        MCP[MCP Server]
        CLI[CLI Tools]
    end

    subgraph "API Gateway Layer"
        Gateway[API Gateway]
        Auth[Authentication]
        RateLimit[Rate Limiting]
        Cache[Response Cache]
    end

    subgraph "Core Services"
        Analytics[Analytics Service]
        Recommend[Recommendation Service]
        Search[Search Service]
        Predict[Prediction Service]
        ML[ML Model Service]
        Ingest[Data Ingestion]
    end

    subgraph "Data Layer"
        Postgres[(PostgreSQL)]
        Elasticsearch[(Elasticsearch)]
        Redis[(Redis)]
        RabbitMQ[Message Queue]
    end

    subgraph "External Data"
        OneDrive[OneDrive for Business]
        FileSystem[File System Events]
    end

    %% Client connections
    GUI --> Gateway
    MCP --> Gateway
    CLI --> Gateway

    %% Gateway routing
    Gateway --> Analytics
    Gateway --> Recommend
    Gateway --> Search
    Gateway --> Predict
    Gateway --> ML

    %% Service dependencies
    Analytics --> Postgres
    Analytics --> Redis
    Recommend --> Postgres
    Recommend --> Redis
    Search --> Postgres
    Search --> Elasticsearch
    Search --> Redis
    Predict --> Postgres
    Predict --> RabbitMQ
    ML --> Redis
    Ingest --> RabbitMQ
    Ingest --> OneDrive
    Ingest --> FileSystem

    %% Cross-service communication
    Ingest --> Search
    Ingest --> Analytics
    Search --> Recommend
    Analytics --> Recommend
```

## Service Specifications

### 1. API Gateway (Port 8000)

**Purpose**: Unified entry point for all backend services

**Key Features**:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Response caching
- Request/response logging
- Circuit breaker pattern

**Technology Stack**:
- FastAPI for high-performance async operations
- Redis for caching and rate limiting
- nginx for load balancing (in production)

**Endpoints**:
```
GET  /health                    # Health check
POST /api/v1/projects/search    # Project search
POST /api/v1/documents/search   # Document search
POST /api/v1/recommendations/*  # Recommendation endpoints
POST /api/v1/analytics/*        # Analytics endpoints
POST /api/v1/prediction/*       # Prediction endpoints
POST /api/v1/ml/*              # ML model endpoints
```

### 2. Analytics Service (Port 8001)

**Purpose**: Comprehensive analytics and metrics collection

**Key Features**:
- Real-time event tracking
- Dashboard data aggregation
- Performance monitoring
- User behavior analysis
- Custom reporting

**Database Schema**:
```sql
-- Core tables
user_events (partitioned by timestamp)
project_metrics (aggregated data)
dashboard_metrics (pre-computed aggregations)
performance_metrics (system performance)

-- Indexes for performance
idx_user_events_user_id_timestamp
idx_user_events_project_id
idx_project_metrics_popularity
```

**Performance Optimizations**:
- Table partitioning by timestamp
- Materialized views for complex aggregations
- Background aggregation jobs
- Redis caching for frequently accessed data

### 3. Recommendation Service (Port 8002)

**Purpose**: Intelligent project and document recommendations

**Key Features**:
- Collaborative filtering (user-based and item-based)
- Content-based filtering
- Hybrid recommendation algorithms
- Real-time learning from user interactions
- Context-aware recommendations

**Algorithms**:
1. **Content-Based**: TF-IDF vectorization + cosine similarity
2. **Collaborative**: Matrix factorization using SVD
3. **Hybrid**: Weighted combination of multiple algorithms
4. **Context-Aware**: Time, location, and usage pattern considerations

**Database Schema**:
```sql
-- Core tables
user_profiles (user preferences and behavior)
project_profiles (project features and similarity)
user_project_interactions (interaction history)
recommendation_cache (cached recommendations)
user_similarity (collaborative filtering matrix)
```

### 4. Enhanced Search Service (Port 8003)

**Purpose**: Advanced search with ML-enhanced ranking

**Key Features**:
- Full-text search with Elasticsearch
- Faceted search and filtering
- Auto-completion and suggestions
- Semantic search capabilities
- Real-time indexing

**Search Pipeline**:
1. **Query Processing**: Query parsing, expansion, and optimization
2. **Retrieval**: Elasticsearch-based document retrieval
3. **Ranking**: ML-enhanced relevance scoring
4. **Post-processing**: Highlighting, facets, and suggestions

**Elasticsearch Indices**:
```json
{
  "quicknav_projects": {
    "mappings": {
      "properties": {
        "title": {"type": "text", "analyzer": "standard"},
        "content": {"type": "text", "analyzer": "standard"},
        "keywords": {"type": "keyword"},
        "categories": {"type": "keyword"},
        "popularity_score": {"type": "float"},
        "title_suggest": {"type": "completion"},
        "keyword_suggest": {"type": "completion"}
      }
    }
  }
}
```

### 5. Prediction Service (Port 8004)

**Purpose**: Predictive navigation and user behavior modeling

**Key Features**:
- Next action prediction
- Navigation pattern analysis
- Anomaly detection
- Usage forecasting

**ML Models**:
1. **Sequence Models**: LSTM for navigation prediction
2. **Classification**: Random Forest for action prediction
3. **Clustering**: K-means for user segmentation
4. **Time Series**: ARIMA for usage forecasting

### 6. ML Model Service (Port 8005)

**Purpose**: Centralized ML model serving and management

**Key Features**:
- Model versioning and deployment
- A/B testing framework
- Model performance monitoring
- Real-time inference serving

**Supported Models**:
- Document classification
- Similarity computation
- Recommendation scoring
- Anomaly detection

## Data Flow Architecture

### 1. Real-time Event Processing

```mermaid
sequenceDiagram
    participant User
    participant GUI
    participant Gateway
    participant Analytics
    participant Queue
    participant ML

    User->>GUI: Interact with project
    GUI->>Gateway: Send event data
    Gateway->>Analytics: Store event
    Gateway->>Queue: Publish event
    Queue->>ML: Process for learning
    ML->>Analytics: Update model metrics
```

### 2. Search Request Flow

```mermaid
sequenceDiagram
    participant User
    participant Gateway
    participant Cache
    participant Search
    participant ES
    participant Analytics

    User->>Gateway: Search request
    Gateway->>Cache: Check cache
    alt Cache hit
        Cache->>Gateway: Return cached results
    else Cache miss
        Gateway->>Search: Forward request
        Search->>ES: Execute search
        ES->>Search: Return results
        Search->>Gateway: Processed results
        Gateway->>Cache: Store results
    end
    Gateway->>Analytics: Log search event
    Gateway->>User: Return results
```

### 3. Recommendation Generation

```mermaid
sequenceDiagram
    participant User
    participant Gateway
    participant Recommend
    participant Analytics
    participant ML

    User->>Gateway: Request recommendations
    Gateway->>Recommend: Forward request
    Recommend->>Analytics: Get user behavior
    Recommend->>ML: Get similarity scores
    Recommend->>Recommend: Generate recommendations
    Recommend->>Gateway: Return recommendations
    Gateway->>User: Serve recommendations
```

## Performance & Scaling Considerations

### 1. Database Optimization

**PostgreSQL Optimizations**:
```sql
-- Partitioning strategy
CREATE TABLE user_events_2024 PARTITION OF user_events
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Composite indexes for common queries
CREATE INDEX idx_user_events_composite
    ON user_events (user_id, event_type, timestamp);

-- GIN indexes for JSON and full-text search
CREATE INDEX idx_user_events_metadata_gin
    ON user_events USING GIN (metadata);
```

**Query Optimization**:
- Use EXPLAIN ANALYZE for query planning
- Implement connection pooling (PgBouncer)
- Regular VACUUM and ANALYZE operations
- Monitor slow queries with pg_stat_statements

### 2. Caching Strategy

**Multi-level Caching**:
1. **Application Cache**: In-memory caching within services
2. **Redis Cache**: Distributed caching across services
3. **CDN Cache**: Static content and API responses
4. **Database Cache**: PostgreSQL shared buffers

**Cache Patterns**:
```python
# Cache-aside pattern
async def get_project_recommendations(user_id: str):
    cache_key = f"recommendations:{user_id}"

    # Try cache first
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # Generate recommendations
    recommendations = await generate_recommendations(user_id)

    # Cache for 30 minutes
    await redis.setex(cache_key, 1800, json.dumps(recommendations))

    return recommendations
```

### 3. Horizontal Scaling

**Service Scaling**:
```yaml
# Docker Compose scaling
services:
  search-service:
    deploy:
      replicas: 3
    environment:
      - INSTANCE_ID=${HOSTNAME}
```

**Database Scaling**:
- Read replicas for analytics queries
- Sharding by user_id or project_id
- Connection pooling and load balancing

**Elasticsearch Scaling**:
```json
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "refresh_interval": "30s"
  }
}
```

### 4. Monitoring & Observability

**Key Metrics**:
- Request rate and response times
- Error rates and types
- Database performance
- Cache hit ratios
- Service dependencies health

**Monitoring Stack**:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **Jaeger**: Distributed tracing
- **ELK Stack**: Centralized logging

**Custom Metrics**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
ACTIVE_USERS = Gauge('active_users', 'Currently active users')
```

## Security Considerations

### 1. Authentication & Authorization

**JWT Token-based Authentication**:
```python
# Token validation middleware
async def validate_token(request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid token")

    try:
        payload = jwt.decode(token[7:], SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
```

### 2. Rate Limiting

**Token Bucket Algorithm**:
```python
async def check_rate_limit(user_id: str, endpoint: str):
    key = f"rate_limit:{user_id}:{endpoint}"
    current = await redis.get(key)

    if current is None:
        await redis.setex(key, 60, 1)  # 1 request per minute window
        return True
    elif int(current) < RATE_LIMIT:
        await redis.incr(key)
        return True
    else:
        return False
```

### 3. Data Protection

**Encryption**:
- TLS 1.3 for data in transit
- AES-256 for sensitive data at rest
- Key rotation policies
- PII anonymization in logs

**Input Validation**:
```python
from pydantic import BaseModel, validator

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

    @validator('query')
    def validate_query(cls, v):
        if len(v) > 200:
            raise ValueError('Query too long')
        return v.strip()
```

## Integration Patterns

### 1. MCP Server Integration

**Tool Extensions**:
```python
@mcp.tool()
async def enhanced_search(query: str, filters: dict = None) -> dict:
    """Enhanced search with ML ranking."""
    response = await http_client.post(
        f"{SEARCH_SERVICE_URL}/search",
        json={"query": query, "filters": filters}
    )
    return response.json()
```

### 2. GUI Integration

**Real-time Updates**:
```python
# WebSocket for real-time updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()

    # Subscribe to user-specific events
    pubsub = redis.pubsub()
    pubsub.subscribe(f"user:{user_id}:updates")

    try:
        while True:
            message = await pubsub.get_message()
            if message and message['type'] == 'message':
                await websocket.send_json(json.loads(message['data']))
    except WebSocketDisconnect:
        pubsub.unsubscribe(f"user:{user_id}:updates")
```

### 3. Event-Driven Architecture

**Event Publishing**:
```python
async def publish_event(event_type: str, data: dict):
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }

    # Publish to Redis for real-time subscribers
    await redis.publish(f"events:{event_type}", json.dumps(event))

    # Queue for background processing
    await rabbitmq.publish(event, routing_key=f"events.{event_type}")
```

## Deployment & DevOps

### 1. Container Strategy

**Multi-stage Dockerfiles**:
```dockerfile
# Base stage
FROM python:3.11-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
COPY . .
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]

# Production stage
FROM base as production
COPY . .
RUN pip install gunicorn
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker"]
```

### 2. Environment Management

**.env Configuration**:
```bash
# Database
POSTGRES_PASSWORD=secure_password
DATABASE_URL=postgresql://user:pass@host:port/db

# Cache
REDIS_URL=redis://host:port
REDIS_PASSWORD=secure_password

# Search
ELASTICSEARCH_URL=http://host:port
ELASTICSEARCH_PASSWORD=secure_password

# Security
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# External APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### 3. Health Checks & Circuit Breakers

**Health Check Implementation**:
```python
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # Check database
    try:
        await db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception:
        health_status["services"]["database"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check Redis
    try:
        await redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception:
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"

    return health_status
```

**Circuit Breaker Pattern**:
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e
```

## Technology Recommendations

### 1. Core Technologies

| Component | Technology | Rationale |
|-----------|------------|-----------|
| API Framework | FastAPI | High performance, async support, auto-documentation |
| Database | PostgreSQL 15 | ACID compliance, JSON support, full-text search |
| Cache | Redis 7 | High performance, persistence, pub/sub support |
| Search | Elasticsearch 8 | Advanced full-text search, analytics, scalability |
| Message Queue | RabbitMQ | Reliable messaging, routing flexibility |
| Container | Docker | Consistent environments, easy deployment |
| Orchestration | Docker Compose | Local development, service coordination |

### 2. Optional Enhancements

| Component | Technology | Use Case |
|-----------|------------|----------|
| Service Mesh | Istio | Advanced traffic management, security |
| API Gateway | Kong/Nginx | Production-grade gateway with plugins |
| Monitoring | Datadog/New Relic | Enterprise monitoring and APM |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Secrets | HashiCorp Vault | Secure secrets management |
| Load Balancer | HAProxy/F5 | High-availability load balancing |

## Migration Strategy

### 1. Phased Rollout

**Phase 1**: Core Infrastructure
- Deploy databases and infrastructure
- Implement API Gateway
- Basic health checks and monitoring

**Phase 2**: Analytics Service
- Event tracking implementation
- Dashboard data collection
- Performance monitoring

**Phase 3**: Search Enhancement
- Elasticsearch integration
- Advanced search features
- Real-time indexing

**Phase 4**: Recommendations
- User profiling
- Basic recommendation algorithms
- A/B testing framework

**Phase 5**: ML & Predictions
- Model deployment
- Advanced analytics
- Predictive features

### 2. Data Migration

**Existing Data Integration**:
```python
async def migrate_existing_data():
    # Extract from existing MCP server
    existing_history = await mcp_client.get_user_history()

    # Transform to new schema
    for entry in existing_history:
        event = UserEvent(
            user_id=entry["user_id"],
            event_type=map_event_type(entry["action"]),
            project_id=entry.get("project_code"),
            timestamp=entry["timestamp"]
        )
        await analytics_service.track_event(event)
```

### 3. Backward Compatibility

**Legacy API Support**:
```python
# Legacy endpoint compatibility
@app.post("/legacy/navigate_project")
async def legacy_navigate_project(project_number: str):
    # Transform to new API format
    response = await search_service.search_projects(
        query=project_number,
        search_type="projects"
    )

    # Transform back to legacy format
    if response.projects:
        return {"status": "success", "path": response.projects[0].path}
    else:
        return {"status": "error", "message": "Project not found"}
```

## Performance Benchmarks

### Target Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Response Time | < 100ms (95th percentile) | Prometheus monitoring |
| Search Latency | < 200ms (95th percentile) | Elasticsearch metrics |
| Recommendation Generation | < 500ms | Custom timing |
| Database Query Time | < 50ms (average) | PostgreSQL logs |
| Cache Hit Ratio | > 80% | Redis metrics |
| Service Availability | > 99.9% | Health check monitoring |

### Load Testing Scenarios

**Concurrent Users**: 100-1000 simultaneous users
**Search Volume**: 10,000 searches per hour
**Event Volume**: 100,000 events per hour
**Data Volume**: 10GB+ of indexed documents

This architecture provides a robust, scalable foundation for Project QuickNav's data-driven features while maintaining compatibility with existing components and enabling future enhancements.