"""
Analytics Service for Project QuickNav

Provides comprehensive analytics and metrics for:
- User behavior and navigation patterns
- Project access frequency and trends
- Document usage statistics
- Performance metrics and bottlenecks
- Dashboard data aggregation
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncpg
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class UserEvent(Base):
    __tablename__ = "user_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    event_type = Column(String, index=True)
    project_id = Column(String, index=True, nullable=True)
    document_path = Column(String, nullable=True)
    query = Column(String, nullable=True)
    result_count = Column(Integer, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata = Column(JSON, nullable=True)

class ProjectMetrics(Base):
    __tablename__ = "project_metrics"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, unique=True, index=True)
    project_name = Column(String)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    unique_users = Column(Integer, default=0)
    document_count = Column(Integer, default=0)
    total_size_mb = Column(Float, default=0.0)
    average_response_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DashboardMetrics(Base):
    __tablename__ = "dashboard_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, index=True)
    metric_value = Column(Float)
    metric_type = Column(String)  # daily, weekly, monthly
    date = Column(DateTime, index=True)
    metadata = Column(JSON, nullable=True)

# Pydantic models
class EventRequest(BaseModel):
    user_id: str
    event_type: str
    project_id: Optional[str] = None
    document_path: Optional[str] = None
    query: Optional[str] = None
    result_count: Optional[int] = None
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AnalyticsRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    metrics: List[str]
    granularity: str = "daily"
    filters: Optional[Dict[str, Any]] = None

class DashboardResponse(BaseModel):
    total_projects: int
    total_users: int
    total_searches: int
    avg_response_time_ms: float
    top_projects: List[Dict[str, Any]]
    search_trends: List[Dict[str, Any]]
    user_activity: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

# Configuration
DATABASE_URL = "postgresql://quicknav:password@localhost/quicknav_analytics"
REDIS_URL = "redis://localhost:6379"

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Analytics Service...")

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Start background tasks
    asyncio.create_task(aggregate_daily_metrics())

    yield

    logger.info("Shutting down Analytics Service...")

app = FastAPI(
    title="Project QuickNav Analytics Service",
    description="Analytics and metrics service for project navigation",
    version="1.0.0",
    lifespan=lifespan
)

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }

@app.post("/events")
async def track_event(
    event: EventRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Track user event for analytics."""
    try:
        # Store event
        db_event = UserEvent(
            user_id=event.user_id,
            event_type=event.event_type,
            project_id=event.project_id,
            document_path=event.document_path,
            query=event.query,
            result_count=event.result_count,
            response_time_ms=event.response_time_ms,
            metadata=event.metadata
        )

        db.add(db_event)
        db.commit()

        # Update project metrics asynchronously
        if event.project_id:
            background_tasks.add_task(update_project_metrics, event.project_id, event.user_id)

        # Update real-time cache
        await update_realtime_cache(event)

        return {"status": "success", "message": "Event tracked"}

    except Exception as e:
        logger.error(f"Failed to track event: {e}")
        raise HTTPException(status_code=500, detail="Failed to track event")

@app.post("/dashboard", response_model=DashboardResponse)
async def get_dashboard_metrics(
    request: AnalyticsRequest,
    db: Session = Depends(get_db)
):
    """Get comprehensive dashboard metrics."""
    try:
        end_date = request.end_date
        start_date = request.start_date

        # Total projects accessed
        total_projects = db.query(func.count(func.distinct(UserEvent.project_id))).filter(
            UserEvent.timestamp >= start_date,
            UserEvent.timestamp <= end_date,
            UserEvent.project_id.isnot(None)
        ).scalar() or 0

        # Total unique users
        total_users = db.query(func.count(func.distinct(UserEvent.user_id))).filter(
            UserEvent.timestamp >= start_date,
            UserEvent.timestamp <= end_date
        ).scalar() or 0

        # Total searches
        total_searches = db.query(func.count(UserEvent.id)).filter(
            UserEvent.event_type == "search",
            UserEvent.timestamp >= start_date,
            UserEvent.timestamp <= end_date
        ).scalar() or 0

        # Average response time
        avg_response_time = db.query(func.avg(UserEvent.response_time_ms)).filter(
            UserEvent.response_time_ms.isnot(None),
            UserEvent.timestamp >= start_date,
            UserEvent.timestamp <= end_date
        ).scalar() or 0.0

        # Top projects by access count
        top_projects_query = db.query(
            UserEvent.project_id,
            func.count(UserEvent.id).label('access_count'),
            func.count(func.distinct(UserEvent.user_id)).label('unique_users')
        ).filter(
            UserEvent.project_id.isnot(None),
            UserEvent.timestamp >= start_date,
            UserEvent.timestamp <= end_date
        ).group_by(UserEvent.project_id).order_by(
            func.count(UserEvent.id).desc()
        ).limit(10).all()

        top_projects = [
            {
                "project_id": row.project_id,
                "access_count": row.access_count,
                "unique_users": row.unique_users
            }
            for row in top_projects_query
        ]

        # Search trends (daily aggregation)
        search_trends = await get_search_trends(db, start_date, end_date, request.granularity)

        # User activity patterns
        user_activity = await get_user_activity_patterns(db, start_date, end_date)

        # Performance metrics
        performance_metrics = await get_performance_metrics(db, start_date, end_date)

        return DashboardResponse(
            total_projects=total_projects,
            total_users=total_users,
            total_searches=total_searches,
            avg_response_time_ms=float(avg_response_time),
            top_projects=top_projects,
            search_trends=search_trends,
            user_activity=user_activity,
            performance_metrics=performance_metrics
        )

    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard metrics")

@app.get("/projects/{project_id}/metrics")
async def get_project_metrics(project_id: str, db: Session = Depends(get_db)):
    """Get detailed metrics for a specific project."""
    try:
        # Get project metrics from database
        project_metrics = db.query(ProjectMetrics).filter(
            ProjectMetrics.project_id == project_id
        ).first()

        if not project_metrics:
            # Calculate metrics on the fly
            metrics = await calculate_project_metrics(db, project_id)
        else:
            metrics = {
                "project_id": project_metrics.project_id,
                "project_name": project_metrics.project_name,
                "access_count": project_metrics.access_count,
                "last_accessed": project_metrics.last_accessed.isoformat() if project_metrics.last_accessed else None,
                "unique_users": project_metrics.unique_users,
                "document_count": project_metrics.document_count,
                "total_size_mb": project_metrics.total_size_mb,
                "average_response_time_ms": project_metrics.average_response_time_ms
            }

        # Get recent activity
        recent_activity = db.query(UserEvent).filter(
            UserEvent.project_id == project_id
        ).order_by(UserEvent.timestamp.desc()).limit(10).all()

        metrics["recent_activity"] = [
            {
                "user_id": event.user_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata
            }
            for event in recent_activity
        ]

        return metrics

    except Exception as e:
        logger.error(f"Failed to get project metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get project metrics")

@app.post("/projects/batch")
async def get_batch_project_analytics(
    request: Dict[str, List[str]],
    db: Session = Depends(get_db)
):
    """Get analytics for multiple projects."""
    project_ids = request.get("project_ids", [])

    try:
        analytics = {}

        for project_id in project_ids:
            # Get cached analytics first
            cached_key = f"project_analytics:{project_id}"
            cached_data = redis_client.get(cached_key)

            if cached_data:
                analytics[project_id] = json.loads(cached_data)
            else:
                # Calculate fresh analytics
                project_analytics = await calculate_project_analytics(db, project_id)
                analytics[project_id] = project_analytics

                # Cache for 10 minutes
                redis_client.setex(cached_key, 600, json.dumps(project_analytics))

        return analytics

    except Exception as e:
        logger.error(f"Failed to get batch project analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get batch analytics")

@app.get("/trends/search")
async def get_search_trends(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get search trends over time."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    trends = await get_search_trends(db, start_date, end_date, "daily")
    return {"trends": trends, "period_days": days}

@app.get("/performance/summary")
async def get_performance_summary(db: Session = Depends(get_db)):
    """Get performance summary metrics."""
    try:
        # Last 24 hours performance
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)

        metrics = await get_performance_metrics(db, start_date, end_date)

        # Add real-time metrics from cache
        realtime_metrics = await get_realtime_metrics()
        metrics.update(realtime_metrics)

        return metrics

    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance summary")

# Helper functions
async def update_project_metrics(project_id: str, user_id: str):
    """Update project metrics in background."""
    try:
        db = SessionLocal()
        try:
            project_metrics = db.query(ProjectMetrics).filter(
                ProjectMetrics.project_id == project_id
            ).first()

            if project_metrics:
                project_metrics.access_count += 1
                project_metrics.last_accessed = datetime.utcnow()
                project_metrics.updated_at = datetime.utcnow()
            else:
                # Create new project metrics
                project_metrics = ProjectMetrics(
                    project_id=project_id,
                    access_count=1,
                    last_accessed=datetime.utcnow(),
                    unique_users=1
                )
                db.add(project_metrics)

            db.commit()

            # Update unique users count
            await update_unique_users_count(db, project_id)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to update project metrics: {e}")

async def update_unique_users_count(db: Session, project_id: str):
    """Update unique users count for a project."""
    unique_count = db.query(func.count(func.distinct(UserEvent.user_id))).filter(
        UserEvent.project_id == project_id
    ).scalar() or 0

    db.query(ProjectMetrics).filter(
        ProjectMetrics.project_id == project_id
    ).update({"unique_users": unique_count})

    db.commit()

async def calculate_project_metrics(db: Session, project_id: str) -> Dict[str, Any]:
    """Calculate project metrics on demand."""
    # Access count
    access_count = db.query(func.count(UserEvent.id)).filter(
        UserEvent.project_id == project_id
    ).scalar() or 0

    # Unique users
    unique_users = db.query(func.count(func.distinct(UserEvent.user_id))).filter(
        UserEvent.project_id == project_id
    ).scalar() or 0

    # Last accessed
    last_event = db.query(UserEvent).filter(
        UserEvent.project_id == project_id
    ).order_by(UserEvent.timestamp.desc()).first()

    # Average response time
    avg_response_time = db.query(func.avg(UserEvent.response_time_ms)).filter(
        UserEvent.project_id == project_id,
        UserEvent.response_time_ms.isnot(None)
    ).scalar() or 0.0

    return {
        "access_count": access_count,
        "unique_users": unique_users,
        "last_accessed": last_event.timestamp.isoformat() if last_event else None,
        "average_response_time_ms": float(avg_response_time)
    }

async def calculate_project_analytics(db: Session, project_id: str) -> Dict[str, Any]:
    """Calculate comprehensive analytics for a project."""
    # Basic metrics
    basic_metrics = await calculate_project_metrics(db, project_id)

    # Access patterns by hour
    hourly_access = db.query(
        func.extract('hour', UserEvent.timestamp).label('hour'),
        func.count(UserEvent.id).label('count')
    ).filter(
        UserEvent.project_id == project_id,
        UserEvent.timestamp >= datetime.utcnow() - timedelta(days=7)
    ).group_by(func.extract('hour', UserEvent.timestamp)).all()

    # Most common document types accessed
    doc_types = db.query(
        UserEvent.metadata['doc_type'].astext.label('doc_type'),
        func.count(UserEvent.id).label('count')
    ).filter(
        UserEvent.project_id == project_id,
        UserEvent.event_type == 'document_search'
    ).group_by(UserEvent.metadata['doc_type'].astext).limit(5).all()

    analytics = {
        **basic_metrics,
        "hourly_access_pattern": [
            {"hour": int(row.hour), "count": row.count}
            for row in hourly_access
        ],
        "popular_document_types": [
            {"doc_type": row.doc_type, "count": row.count}
            for row in doc_types if row.doc_type
        ]
    }

    return analytics

async def get_search_trends(db: Session, start_date: datetime, end_date: datetime, granularity: str) -> List[Dict]:
    """Get search trends over time."""
    if granularity == "daily":
        date_trunc = func.date_trunc('day', UserEvent.timestamp)
    elif granularity == "weekly":
        date_trunc = func.date_trunc('week', UserEvent.timestamp)
    else:  # monthly
        date_trunc = func.date_trunc('month', UserEvent.timestamp)

    trends = db.query(
        date_trunc.label('date'),
        func.count(UserEvent.id).label('count')
    ).filter(
        UserEvent.event_type == "search",
        UserEvent.timestamp >= start_date,
        UserEvent.timestamp <= end_date
    ).group_by(date_trunc).order_by(date_trunc).all()

    return [
        {
            "date": row.date.isoformat(),
            "count": row.count
        }
        for row in trends
    ]

async def get_user_activity_patterns(db: Session, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Get user activity patterns."""
    activity = db.query(
        func.extract('hour', UserEvent.timestamp).label('hour'),
        func.count(UserEvent.id).label('total_events'),
        func.count(func.distinct(UserEvent.user_id)).label('unique_users')
    ).filter(
        UserEvent.timestamp >= start_date,
        UserEvent.timestamp <= end_date
    ).group_by(func.extract('hour', UserEvent.timestamp)).order_by('hour').all()

    return [
        {
            "hour": int(row.hour),
            "total_events": row.total_events,
            "unique_users": row.unique_users
        }
        for row in activity
    ]

async def get_performance_metrics(db: Session, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Get performance metrics."""
    # Response time percentiles
    response_times = db.query(UserEvent.response_time_ms).filter(
        UserEvent.response_time_ms.isnot(None),
        UserEvent.timestamp >= start_date,
        UserEvent.timestamp <= end_date
    ).all()

    times = [float(rt[0]) for rt in response_times if rt[0] is not None]

    if times:
        times.sort()
        p50 = times[int(len(times) * 0.5)]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
    else:
        p50 = p95 = p99 = 0.0

    # Error rate
    total_events = db.query(func.count(UserEvent.id)).filter(
        UserEvent.timestamp >= start_date,
        UserEvent.timestamp <= end_date
    ).scalar() or 0

    error_events = db.query(func.count(UserEvent.id)).filter(
        UserEvent.metadata['error'].astext.isnot(None),
        UserEvent.timestamp >= start_date,
        UserEvent.timestamp <= end_date
    ).scalar() or 0

    error_rate = (error_events / total_events * 100) if total_events > 0 else 0.0

    return {
        "response_time_p50_ms": p50,
        "response_time_p95_ms": p95,
        "response_time_p99_ms": p99,
        "error_rate_percent": error_rate,
        "total_requests": total_events
    }

async def update_realtime_cache(event: EventRequest):
    """Update real-time metrics cache."""
    try:
        # Update counters
        redis_client.incr("realtime:total_events")
        redis_client.incr(f"realtime:events:{event.event_type}")

        if event.user_id:
            redis_client.sadd("realtime:active_users", event.user_id)
            redis_client.expire("realtime:active_users", 3600)  # 1 hour

        if event.response_time_ms:
            redis_client.lpush("realtime:response_times", event.response_time_ms)
            redis_client.ltrim("realtime:response_times", 0, 99)  # Keep last 100

    except Exception as e:
        logger.error(f"Failed to update realtime cache: {e}")

async def get_realtime_metrics() -> Dict[str, Any]:
    """Get real-time metrics from cache."""
    try:
        active_users = redis_client.scard("realtime:active_users") or 0
        total_events = int(redis_client.get("realtime:total_events") or 0)

        # Recent response times
        response_times = redis_client.lrange("realtime:response_times", 0, -1)
        avg_response_time = 0.0
        if response_times:
            times = [float(t) for t in response_times]
            avg_response_time = sum(times) / len(times)

        return {
            "active_users_last_hour": active_users,
            "total_events_today": total_events,
            "avg_response_time_recent_ms": avg_response_time
        }

    except Exception as e:
        logger.error(f"Failed to get realtime metrics: {e}")
        return {}

async def aggregate_daily_metrics():
    """Background task to aggregate daily metrics."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour

            db = SessionLocal()
            try:
                # Aggregate metrics for yesterday
                yesterday = datetime.utcnow().date() - timedelta(days=1)
                start_date = datetime.combine(yesterday, datetime.min.time())
                end_date = datetime.combine(yesterday, datetime.max.time())

                # Total searches
                total_searches = db.query(func.count(UserEvent.id)).filter(
                    UserEvent.event_type == "search",
                    UserEvent.timestamp >= start_date,
                    UserEvent.timestamp <= end_date
                ).scalar() or 0

                # Save to dashboard metrics
                metric = DashboardMetrics(
                    metric_name="daily_searches",
                    metric_value=total_searches,
                    metric_type="daily",
                    date=start_date
                )
                db.add(metric)
                db.commit()

                logger.info(f"Aggregated daily metrics for {yesterday}")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to aggregate daily metrics: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)