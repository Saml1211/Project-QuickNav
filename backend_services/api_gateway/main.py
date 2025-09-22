"""
API Gateway for Project QuickNav Backend Services

Provides unified access to all backend services with authentication,
rate limiting, caching, and request routing.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import asyncio
import time
import redis
import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
import httpx
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API contracts
class ProjectSearchRequest(BaseModel):
    query: str
    limit: int = 10
    include_analytics: bool = False
    user_id: Optional[str] = None

class ProjectSearchResponse(BaseModel):
    status: str
    projects: List[Dict[str, Any]]
    total_count: int
    search_time_ms: float
    from_cache: bool = False

class DocumentSearchRequest(BaseModel):
    project_id: str
    doc_type: Optional[str] = None
    filters: Dict[str, Any] = {}
    include_content: bool = False

class RecommendationRequest(BaseModel):
    user_id: str
    context: Dict[str, Any] = {}
    limit: int = 5

class AnalyticsRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    metrics: List[str]
    granularity: str = "daily"

# Configuration
class Config:
    REDIS_URL = "redis://localhost:6379"
    CACHE_TTL = 300  # 5 minutes
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 60  # 1 minute

    # Service endpoints
    ANALYTICS_SERVICE_URL = "http://localhost:8001"
    RECOMMENDATION_SERVICE_URL = "http://localhost:8002"
    SEARCH_SERVICE_URL = "http://localhost:8003"
    PREDICTION_SERVICE_URL = "http://localhost:8004"
    ML_SERVICE_URL = "http://localhost:8005"

# Redis client for caching and rate limiting
redis_client = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting API Gateway...")

    # Test service connections
    await test_service_connections()

    yield

    logger.info("Shutting down API Gateway...")

app = FastAPI(
    title="Project QuickNav API Gateway",
    description="Unified API for Project QuickNav backend services",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# HTTP client for service communication
http_client = httpx.AsyncClient(timeout=30.0)

async def test_service_connections():
    """Test connections to all backend services."""
    services = {
        "analytics": Config.ANALYTICS_SERVICE_URL,
        "recommendation": Config.RECOMMENDATION_SERVICE_URL,
        "search": Config.SEARCH_SERVICE_URL,
        "prediction": Config.PREDICTION_SERVICE_URL,
        "ml": Config.ML_SERVICE_URL
    }

    for service, url in services.items():
        try:
            response = await http_client.get(f"{url}/health")
            if response.status_code == 200:
                logger.info(f"✓ {service} service connected")
            else:
                logger.warning(f"⚠ {service} service unhealthy: {response.status_code}")
        except Exception as e:
            logger.error(f"✗ {service} service unavailable: {e}")

async def get_user_id(request: Request) -> Optional[str]:
    """Extract user ID from request headers or session."""
    # Simple implementation - enhance with proper authentication
    return request.headers.get("X-User-ID") or "anonymous"

async def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit."""
    key = f"rate_limit:{user_id}"
    current = redis_client.get(key)

    if current is None:
        redis_client.setex(key, Config.RATE_LIMIT_WINDOW, 1)
        return True
    elif int(current) < Config.RATE_LIMIT_REQUESTS:
        redis_client.incr(key)
        return True
    else:
        return False

async def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response if available."""
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    return None

async def set_cached_response(cache_key: str, data: Dict, ttl: int = Config.CACHE_TTL):
    """Cache response data."""
    try:
        redis_client.setex(cache_key, ttl, json.dumps(data))
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    user_id = await get_user_id(request)

    if not await check_rate_limit(user_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": Config.RATE_LIMIT_WINDOW}
        )

    response = await call_next(request)
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": "connected" if redis_client.ping() else "disconnected"
        }
    }

# Project search endpoints
@app.post("/api/v1/projects/search", response_model=ProjectSearchResponse)
async def search_projects(
    request: ProjectSearchRequest,
    user_id: str = Depends(get_user_id)
):
    """Search for projects with enhanced ranking and analytics."""
    start_time = time.time()

    # Generate cache key
    cache_key = f"search:{hash(str(request))}"

    # Check cache first
    cached_response = await get_cached_response(cache_key)
    if cached_response:
        cached_response["from_cache"] = True
        return ProjectSearchResponse(**cached_response)

    try:
        # Call search service
        response = await http_client.post(
            f"{Config.SEARCH_SERVICE_URL}/search",
            json=request.dict()
        )
        response.raise_for_status()

        search_result = response.json()

        # Add analytics data if requested
        if request.include_analytics and search_result.get("projects"):
            analytics_data = await get_project_analytics(
                [p["id"] for p in search_result["projects"]]
            )
            for project in search_result["projects"]:
                project["analytics"] = analytics_data.get(project["id"], {})

        # Calculate response time
        search_time_ms = (time.time() - start_time) * 1000

        result = {
            "status": "success",
            "projects": search_result.get("projects", []),
            "total_count": search_result.get("total_count", 0),
            "search_time_ms": search_time_ms,
            "from_cache": False
        }

        # Cache the response
        await set_cached_response(cache_key, result)

        # Track search analytics
        await track_search_event(user_id, request.query, len(result["projects"]))

        return ProjectSearchResponse(**result)

    except httpx.RequestError as e:
        logger.error(f"Search service error: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/documents/search")
async def search_documents(
    request: DocumentSearchRequest,
    user_id: str = Depends(get_user_id)
):
    """Search for documents within projects."""
    try:
        response = await http_client.post(
            f"{Config.SEARCH_SERVICE_URL}/documents",
            json=request.dict()
        )
        response.raise_for_status()

        result = response.json()

        # Track document search
        await track_document_search_event(user_id, request.project_id, request.doc_type)

        return result

    except httpx.RequestError as e:
        logger.error(f"Document search service error: {e}")
        raise HTTPException(status_code=503, detail="Document search service unavailable")

# Recommendation endpoints
@app.post("/api/v1/recommendations/projects")
async def get_project_recommendations(
    request: RecommendationRequest,
    user_id: str = Depends(get_user_id)
):
    """Get personalized project recommendations."""
    cache_key = f"recommendations:{user_id}:{hash(str(request.context))}"

    # Check cache
    cached_response = await get_cached_response(cache_key)
    if cached_response:
        return cached_response

    try:
        response = await http_client.post(
            f"{Config.RECOMMENDATION_SERVICE_URL}/projects",
            json=request.dict()
        )
        response.raise_for_status()

        result = response.json()

        # Cache for shorter time (recommendations change frequently)
        await set_cached_response(cache_key, result, ttl=60)

        return result

    except httpx.RequestError as e:
        logger.error(f"Recommendation service error: {e}")
        raise HTTPException(status_code=503, detail="Recommendation service unavailable")

# Analytics endpoints
@app.post("/api/v1/analytics/dashboard")
async def get_dashboard_analytics(
    request: AnalyticsRequest,
    user_id: str = Depends(get_user_id)
):
    """Get analytics data for dashboard."""
    try:
        response = await http_client.post(
            f"{Config.ANALYTICS_SERVICE_URL}/dashboard",
            json=request.dict()
        )
        response.raise_for_status()

        return response.json()

    except httpx.RequestError as e:
        logger.error(f"Analytics service error: {e}")
        raise HTTPException(status_code=503, detail="Analytics service unavailable")

@app.get("/api/v1/analytics/projects/{project_id}/metrics")
async def get_project_metrics(project_id: str):
    """Get metrics for a specific project."""
    cache_key = f"project_metrics:{project_id}"

    cached_response = await get_cached_response(cache_key)
    if cached_response:
        return cached_response

    try:
        response = await http_client.get(
            f"{Config.ANALYTICS_SERVICE_URL}/projects/{project_id}/metrics"
        )
        response.raise_for_status()

        result = response.json()
        await set_cached_response(cache_key, result, ttl=600)  # Cache for 10 minutes

        return result

    except httpx.RequestError as e:
        logger.error(f"Analytics service error: {e}")
        raise HTTPException(status_code=503, detail="Analytics service unavailable")

# Predictive navigation endpoints
@app.post("/api/v1/prediction/next-action")
async def predict_next_action(
    context: Dict[str, Any],
    user_id: str = Depends(get_user_id)
):
    """Predict user's next likely action."""
    try:
        response = await http_client.post(
            f"{Config.PREDICTION_SERVICE_URL}/next-action",
            json={"user_id": user_id, "context": context}
        )
        response.raise_for_status()

        return response.json()

    except httpx.RequestError as e:
        logger.error(f"Prediction service error: {e}")
        raise HTTPException(status_code=503, detail="Prediction service unavailable")

# ML model endpoints
@app.post("/api/v1/ml/classify-document")
async def classify_document(
    document_path: str,
    content: Optional[str] = None
):
    """Classify document type using ML model."""
    try:
        response = await http_client.post(
            f"{Config.ML_SERVICE_URL}/classify",
            json={"document_path": document_path, "content": content}
        )
        response.raise_for_status()

        return response.json()

    except httpx.RequestError as e:
        logger.error(f"ML service error: {e}")
        raise HTTPException(status_code=503, detail="ML service unavailable")

# Data ingestion endpoints
@app.post("/api/v1/ingest/project-update")
async def ingest_project_update(
    project_id: str,
    update_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Ingest project update for real-time processing."""
    background_tasks.add_task(process_project_update, project_id, update_data)

    return {"status": "accepted", "message": "Update queued for processing"}

@app.post("/api/v1/ingest/user-action")
async def ingest_user_action(
    action_data: Dict[str, Any],
    user_id: str = Depends(get_user_id),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Ingest user action for analytics and recommendations."""
    action_data["user_id"] = user_id
    action_data["timestamp"] = datetime.utcnow().isoformat()

    background_tasks.add_task(process_user_action, action_data)

    return {"status": "accepted"}

# Helper functions
async def get_project_analytics(project_ids: List[str]) -> Dict[str, Any]:
    """Get analytics data for multiple projects."""
    try:
        response = await http_client.post(
            f"{Config.ANALYTICS_SERVICE_URL}/projects/batch",
            json={"project_ids": project_ids}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get project analytics: {e}")
        return {}

async def track_search_event(user_id: str, query: str, result_count: int):
    """Track search event for analytics."""
    event_data = {
        "user_id": user_id,
        "event_type": "search",
        "query": query,
        "result_count": result_count,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        await http_client.post(
            f"{Config.ANALYTICS_SERVICE_URL}/events",
            json=event_data
        )
    except Exception as e:
        logger.error(f"Failed to track search event: {e}")

async def track_document_search_event(user_id: str, project_id: str, doc_type: Optional[str]):
    """Track document search event."""
    event_data = {
        "user_id": user_id,
        "event_type": "document_search",
        "project_id": project_id,
        "doc_type": doc_type,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        await http_client.post(
            f"{Config.ANALYTICS_SERVICE_URL}/events",
            json=event_data
        )
    except Exception as e:
        logger.error(f"Failed to track document search event: {e}")

async def process_project_update(project_id: str, update_data: Dict[str, Any]):
    """Process project update in background."""
    # Send to data ingestion service
    try:
        await http_client.post(
            f"{Config.SEARCH_SERVICE_URL}/index/update",
            json={"project_id": project_id, "data": update_data}
        )
    except Exception as e:
        logger.error(f"Failed to process project update: {e}")

async def process_user_action(action_data: Dict[str, Any]):
    """Process user action in background."""
    # Send to recommendation service for learning
    try:
        await http_client.post(
            f"{Config.RECOMMENDATION_SERVICE_URL}/learn",
            json=action_data
        )
    except Exception as e:
        logger.error(f"Failed to process user action: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)