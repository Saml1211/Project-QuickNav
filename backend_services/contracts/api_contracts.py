"""
API Contracts and Integration Patterns for Project QuickNav Backend Services

This module defines the standardized contracts for service communication,
data models, and integration patterns between all backend services.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

# Common enums
class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class EventType(str, Enum):
    PROJECT_VIEW = "project_view"
    PROJECT_SEARCH = "project_search"
    DOCUMENT_ACCESS = "document_access"
    DOCUMENT_SEARCH = "document_search"
    USER_LOGIN = "user_login"
    RECOMMENDATION_CLICK = "recommendation_click"

class DocumentType(str, Enum):
    LLD = "lld"
    HLD = "hld"
    CHANGE_ORDER = "change_order"
    SALES_PO = "sales_po"
    FLOOR_PLANS = "floor_plans"
    SCOPE = "scope"
    QA_ITP = "qa_itp"
    SWMS = "swms"
    SUPPLIER_QUOTES = "supplier_quotes"
    PHOTOS = "photos"

class RecommendationType(str, Enum):
    PROJECTS = "projects"
    DOCUMENTS = "documents"
    SIMILAR_PROJECTS = "similar_projects"

# Base models
class BaseResponse(BaseModel):
    """Base response model for all API endpoints."""
    status: str = Field(..., description="Response status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ErrorResponse(BaseResponse):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: ServiceStatus
    timestamp: datetime
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    dependencies: Optional[Dict[str, ServiceStatus]] = None

# Project models
class ProjectInfo(BaseModel):
    """Project information model."""
    project_id: str = Field(..., description="Unique project identifier")
    project_name: str = Field(..., description="Human-readable project name")
    project_number: Optional[str] = Field(None, description="5-digit project number")
    path: str = Field(..., description="Full filesystem path")
    categories: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProjectSearchRequest(BaseModel):
    """Project search request model."""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    include_analytics: bool = Field(default=False, description="Include analytics data")

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class ProjectSearchResponse(BaseResponse):
    """Project search response model."""
    projects: List[ProjectInfo] = Field(..., description="Search results")
    total_count: int = Field(..., ge=0, description="Total matching projects")
    search_time_ms: float = Field(..., ge=0, description="Search execution time")
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None)
    suggestions: List[str] = Field(default_factory=list)
    from_cache: bool = Field(default=False)

# Document models
class DocumentInfo(BaseModel):
    """Document information model."""
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    path: str = Field(..., description="Full filesystem path")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    doc_type: Optional[DocumentType] = Field(None, description="Document type")
    size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")
    version: Optional[str] = Field(None, description="Document version")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentSearchRequest(BaseModel):
    """Document search request model."""
    project_id: str = Field(..., description="Project ID to search within")
    doc_type: Optional[DocumentType] = Field(None, description="Document type filter")
    query: Optional[str] = Field(None, description="Text search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")
    include_content: bool = Field(default=False, description="Include document content")

class DocumentSearchResponse(BaseResponse):
    """Document search response model."""
    documents: List[DocumentInfo] = Field(..., description="Search results")
    total_count: int = Field(..., ge=0, description="Total matching documents")
    project_id: str = Field(..., description="Project ID searched")

# Analytics models
class UserEvent(BaseModel):
    """User event model for analytics."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    event_type: EventType = Field(..., description="Type of event")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    document_id: Optional[str] = Field(None, description="Associated document ID")
    query: Optional[str] = Field(None, description="Search query if applicable")
    result_count: Optional[int] = Field(None, ge=0, description="Number of results")
    response_time_ms: Optional[float] = Field(None, ge=0, description="Response time")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AnalyticsMetrics(BaseModel):
    """Analytics metrics model."""
    total_projects: int = Field(..., ge=0)
    total_users: int = Field(..., ge=0)
    total_searches: int = Field(..., ge=0)
    avg_response_time_ms: float = Field(..., ge=0)
    error_rate_percent: float = Field(..., ge=0, le=100)

class DashboardData(BaseModel):
    """Dashboard data model."""
    metrics: AnalyticsMetrics
    top_projects: List[Dict[str, Any]]
    search_trends: List[Dict[str, Any]]
    user_activity: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    time_range: Dict[str, datetime]

# Recommendation models
class RecommendationItem(BaseModel):
    """Individual recommendation item."""
    item_id: str = Field(..., description="ID of recommended item")
    item_type: str = Field(..., description="Type of item (project, document)")
    title: str = Field(..., description="Item title")
    confidence_score: float = Field(..., ge=0, le=1, description="Recommendation confidence")
    reasoning: List[str] = Field(default_factory=list, description="Recommendation reasons")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecommendationRequest(BaseModel):
    """Recommendation request model."""
    user_id: str = Field(..., description="User requesting recommendations")
    recommendation_type: RecommendationType = Field(..., description="Type of recommendations")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context information")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum recommendations")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class RecommendationResponse(BaseResponse):
    """Recommendation response model."""
    recommendations: List[RecommendationItem] = Field(..., description="Recommendation list")
    algorithm_used: str = Field(..., description="Algorithm used for recommendations")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence")
    context_considered: List[str] = Field(..., description="Context factors considered")
    cached: bool = Field(default=False, description="Whether results were cached")

# ML/Prediction models
class PredictionRequest(BaseModel):
    """Prediction request model."""
    user_id: str = Field(..., description="User ID")
    context: Dict[str, Any] = Field(..., description="Current context")
    prediction_type: str = Field(..., description="Type of prediction requested")

class PredictionResponse(BaseResponse):
    """Prediction response model."""
    predictions: List[Dict[str, Any]] = Field(..., description="Prediction results")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_version: str = Field(..., description="ML model version used")

# Data ingestion models
class DataIngestionRequest(BaseModel):
    """Data ingestion request model."""
    source: str = Field(..., description="Data source identifier")
    data_type: str = Field(..., description="Type of data being ingested")
    data: Dict[str, Any] = Field(..., description="Data payload")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10, description="Processing priority")

class DataIngestionResponse(BaseResponse):
    """Data ingestion response model."""
    ingestion_id: str = Field(..., description="Unique ingestion identifier")
    estimated_processing_time_ms: Optional[int] = Field(None, description="Estimated processing time")

# Service communication models
class ServiceRequest(BaseModel):
    """Base service-to-service request model."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_service: str = Field(..., description="Source service name")
    target_service: str = Field(..., description="Target service name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)

class ServiceResponse(BaseModel):
    """Base service-to-service response model."""
    request_id: str = Field(..., description="Original request ID")
    source_service: str = Field(..., description="Responding service name")
    success: bool = Field(..., description="Whether request was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., ge=0)

# Integration patterns
class EventMessage(BaseModel):
    """Event message for pub/sub integration."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., description="Event type identifier")
    source_service: str = Field(..., description="Service that generated the event")
    payload: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")

class BatchRequest(BaseModel):
    """Batch processing request model."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)
    processing_options: Dict[str, Any] = Field(default_factory=dict)

class BatchResponse(BaseResponse):
    """Batch processing response model."""
    batch_id: str = Field(..., description="Original batch ID")
    total_items: int = Field(..., ge=0, description="Total items processed")
    successful_items: int = Field(..., ge=0, description="Successfully processed items")
    failed_items: int = Field(..., ge=0, description="Failed items")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")

# Configuration models
class ServiceConfig(BaseModel):
    """Service configuration model."""
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment (dev, staging, prod)")
    database_url: Optional[str] = Field(None, description="Database connection URL")
    redis_url: Optional[str] = Field(None, description="Redis connection URL")
    elasticsearch_url: Optional[str] = Field(None, description="Elasticsearch URL")
    api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys")
    feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Feature flags")
    rate_limits: Dict[str, int] = Field(default_factory=dict, description="Rate limits")

# MCP integration models
class MCPToolRequest(BaseModel):
    """MCP tool request model."""
    tool_name: str = Field(..., description="MCP tool name")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context")

class MCPToolResponse(BaseResponse):
    """MCP tool response model."""
    tool_name: str = Field(..., description="MCP tool name")
    result: Dict[str, Any] = Field(..., description="Tool execution result")

class MCPResourceRequest(BaseModel):
    """MCP resource request model."""
    resource_uri: str = Field(..., description="Resource URI")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Resource parameters")

class MCPResourceResponse(BaseResponse):
    """MCP resource response model."""
    resource_uri: str = Field(..., description="Resource URI")
    content: Union[str, bytes, Dict[str, Any]] = Field(..., description="Resource content")
    content_type: str = Field(..., description="Content type")

# GUI integration models
class GUIUpdateEvent(BaseModel):
    """GUI update event model."""
    event_type: str = Field(..., description="Update event type")
    component: str = Field(..., description="GUI component to update")
    data: Dict[str, Any] = Field(..., description="Update data")
    user_id: Optional[str] = Field(None, description="User ID if user-specific")

class GUICommand(BaseModel):
    """GUI command model."""
    command_type: str = Field(..., description="Command type")
    parameters: Dict[str, Any] = Field(..., description="Command parameters")
    callback_id: Optional[str] = Field(None, description="Callback identifier")

# Performance monitoring models
class PerformanceMetric(BaseModel):
    """Performance metric model."""
    metric_name: str = Field(..., description="Metric name")
    metric_value: float = Field(..., description="Metric value")
    metric_unit: str = Field(..., description="Metric unit (ms, count, etc.)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")

class PerformanceReport(BaseModel):
    """Performance report model."""
    service_name: str = Field(..., description="Service name")
    report_period: str = Field(..., description="Report period (hourly, daily, etc.)")
    metrics: List[PerformanceMetric] = Field(..., description="Performance metrics")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")

# Validation and utility functions
def validate_project_id(project_id: str) -> bool:
    """Validate project ID format."""
    return bool(project_id and len(project_id) >= 5)

def validate_user_id(user_id: str) -> bool:
    """Validate user ID format."""
    return bool(user_id and len(user_id) >= 3)

def create_error_response(
    error_message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """Create standardized error response."""
    return ErrorResponse(
        status="error",
        error=error_message,
        error_code=error_code,
        details=details
    )

def create_success_response(
    data: Dict[str, Any],
    message: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized success response."""
    response = {
        "status": "success",
        "timestamp": datetime.utcnow(),
        "request_id": str(uuid.uuid4()),
        **data
    }

    if message:
        response["message"] = message

    return response

# Service registry models
class ServiceRegistration(BaseModel):
    """Service registration model."""
    service_name: str = Field(..., description="Service name")
    service_version: str = Field(..., description="Service version")
    host: str = Field(..., description="Service host")
    port: int = Field(..., ge=1, le=65535, description="Service port")
    health_check_url: str = Field(..., description="Health check endpoint")
    capabilities: List[str] = Field(..., description="Service capabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=datetime.utcnow)

class ServiceDirectory(BaseModel):
    """Service directory model."""
    services: List[ServiceRegistration] = Field(..., description="Registered services")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Circuit breaker models
class CircuitBreakerState(BaseModel):
    """Circuit breaker state model."""
    service_name: str = Field(..., description="Target service name")
    state: str = Field(..., description="Circuit breaker state (closed, open, half-open)")
    failure_count: int = Field(..., ge=0, description="Consecutive failure count")
    last_failure_time: Optional[datetime] = Field(None, description="Last failure timestamp")
    next_attempt_time: Optional[datetime] = Field(None, description="Next attempt allowed time")

# Rate limiting models
class RateLimitInfo(BaseModel):
    """Rate limit information model."""
    limit: int = Field(..., ge=1, description="Request limit")
    remaining: int = Field(..., ge=0, description="Remaining requests")
    reset_time: datetime = Field(..., description="Reset timestamp")
    window_size_seconds: int = Field(..., ge=1, description="Rate limit window size")

# Authentication models
class AuthToken(BaseModel):
    """Authentication token model."""
    token: str = Field(..., description="JWT token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_at: datetime = Field(..., description="Token expiration time")
    user_id: str = Field(..., description="Associated user ID")
    scopes: List[str] = Field(default_factory=list, description="Token scopes")

class AuthRequest(BaseModel):
    """Authentication request model."""
    username: Optional[str] = Field(None, description="Username")
    password: Optional[str] = Field(None, description="Password")
    api_key: Optional[str] = Field(None, description="API key")
    token: Optional[str] = Field(None, description="Existing token for refresh")

class AuthResponse(BaseResponse):
    """Authentication response model."""
    auth_token: Optional[AuthToken] = Field(None, description="Authentication token")
    user_info: Optional[Dict[str, Any]] = Field(None, description="User information")

# Export all models for easy importing
__all__ = [
    # Enums
    "ServiceStatus", "EventType", "DocumentType", "RecommendationType",

    # Base models
    "BaseResponse", "ErrorResponse", "HealthResponse",

    # Project models
    "ProjectInfo", "ProjectSearchRequest", "ProjectSearchResponse",

    # Document models
    "DocumentInfo", "DocumentSearchRequest", "DocumentSearchResponse",

    # Analytics models
    "UserEvent", "AnalyticsMetrics", "DashboardData",

    # Recommendation models
    "RecommendationItem", "RecommendationRequest", "RecommendationResponse",

    # ML/Prediction models
    "PredictionRequest", "PredictionResponse",

    # Data ingestion models
    "DataIngestionRequest", "DataIngestionResponse",

    # Service communication models
    "ServiceRequest", "ServiceResponse", "EventMessage",

    # Batch processing models
    "BatchRequest", "BatchResponse",

    # Configuration models
    "ServiceConfig",

    # MCP integration models
    "MCPToolRequest", "MCPToolResponse", "MCPResourceRequest", "MCPResourceResponse",

    # GUI integration models
    "GUIUpdateEvent", "GUICommand",

    # Performance monitoring models
    "PerformanceMetric", "PerformanceReport",

    # Service registry models
    "ServiceRegistration", "ServiceDirectory",

    # Circuit breaker models
    "CircuitBreakerState",

    # Rate limiting models
    "RateLimitInfo",

    # Authentication models
    "AuthToken", "AuthRequest", "AuthResponse",

    # Utility functions
    "validate_project_id", "validate_user_id", "create_error_response", "create_success_response"
]