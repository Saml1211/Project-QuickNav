"""
Recommendation Service for Project QuickNav

Provides intelligent project and document recommendations using:
- Collaborative filtering (user-based and item-based)
- Content-based filtering (project similarity)
- Hybrid recommendation algorithms
- Real-time learning from user interactions
- Context-aware recommendations
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import asyncio
import json
import logging
from pydantic import BaseModel
from contextlib import asynccontextmanager
import redis
import pickle
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    preferences = Column(JSON, default=dict)
    interaction_count = Column(Integer, default=0)
    favorite_projects = Column(JSON, default=list)
    preferred_doc_types = Column(JSON, default=list)
    work_hours = Column(JSON, default=dict)  # Peak activity hours
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProjectProfile(Base):
    __tablename__ = "project_profiles"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, unique=True, index=True)
    project_name = Column(String)
    keywords = Column(JSON, default=list)
    categories = Column(JSON, default=list)
    popularity_score = Column(Float, default=0.0)
    content_vector = Column(JSON, default=list)  # TF-IDF features
    similar_projects = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserProjectInteraction(Base):
    __tablename__ = "user_project_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    project_id = Column(String, index=True)
    interaction_type = Column(String)  # view, search, document_access
    interaction_score = Column(Float, default=1.0)
    context = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class RecommendationCache(Base):
    __tablename__ = "recommendation_cache"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    recommendation_type = Column(String)
    recommendations = Column(JSON, default=list)
    context_hash = Column(String)
    expires_at = Column(DateTime, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: str
    context: Dict[str, Any] = {}
    limit: int = 5
    recommendation_type: str = "projects"
    filters: Optional[Dict[str, Any]] = None

class LearnRequest(BaseModel):
    user_id: str
    event_type: str
    project_id: Optional[str] = None
    document_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    algorithm_used: str
    confidence_score: float
    context_considered: List[str]
    cached: bool = False

# Configuration
DATABASE_URL = "postgresql://quicknav:password@localhost/quicknav_recommendations"
REDIS_URL = "redis://localhost:6379"

# Database and cache setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Recommendation Service...")

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Load or initialize recommendation models
    await initialize_recommendation_models()

    # Start background tasks
    asyncio.create_task(retrain_models_periodically())
    asyncio.create_task(update_popularity_scores())

    yield

    logger.info("Shutting down Recommendation Service...")

app = FastAPI(
    title="Project QuickNav Recommendation Service",
    description="Intelligent recommendation service for projects and documents",
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

# Global variables for recommendation models
tfidf_vectorizer = None
project_similarity_matrix = None
user_item_matrix = None
collaborative_model = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": tfidf_vectorizer is not None
    }

@app.post("/projects", response_model=RecommendationResponse)
async def recommend_projects(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get project recommendations for a user."""
    try:
        # Check cache first
        cached_recommendations = await get_cached_recommendations(
            request.user_id, "projects", request.context, request.limit
        )

        if cached_recommendations:
            return RecommendationResponse(**cached_recommendations)

        # Get user profile
        user_profile = await get_or_create_user_profile(db, request.user_id)

        # Generate recommendations using hybrid approach
        recommendations = await generate_hybrid_recommendations(
            db, user_profile, request.context, request.limit, request.filters
        )

        # Cache recommendations
        await cache_recommendations(
            request.user_id, "projects", recommendations, request.context
        )

        return recommendations

    except Exception as e:
        logger.error(f"Failed to generate project recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

@app.post("/documents")
async def recommend_documents(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get document recommendations within a project."""
    try:
        project_id = request.context.get("project_id")
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id required in context")

        # Get user's document preferences
        user_profile = await get_or_create_user_profile(db, request.user_id)
        preferred_doc_types = user_profile.preferred_doc_types or []

        # Get document access patterns for this project
        recent_docs = await get_user_document_patterns(db, request.user_id, project_id)

        # Generate document recommendations
        recommendations = await generate_document_recommendations(
            db, request.user_id, project_id, preferred_doc_types, recent_docs, request.limit
        )

        return {
            "recommendations": recommendations,
            "algorithm_used": "document_patterns",
            "confidence_score": 0.8,
            "context_considered": ["user_preferences", "access_patterns", "project_context"]
        }

    except Exception as e:
        logger.error(f"Failed to generate document recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate document recommendations")

@app.post("/similar-projects")
async def get_similar_projects(
    project_id: str,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get projects similar to the given project."""
    try:
        # Get project profile
        project_profile = db.query(ProjectProfile).filter(
            ProjectProfile.project_id == project_id
        ).first()

        if not project_profile:
            # Calculate similarity on the fly
            similar_projects = await calculate_project_similarity_onfly(db, project_id, limit)
        else:
            similar_projects = project_profile.similar_projects[:limit]

        return {
            "project_id": project_id,
            "similar_projects": similar_projects,
            "algorithm_used": "content_similarity"
        }

    except Exception as e:
        logger.error(f"Failed to get similar projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to get similar projects")

@app.post("/learn")
async def learn_from_interaction(
    request: LearnRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Learn from user interactions for improved recommendations."""
    try:
        # Record interaction
        interaction = UserProjectInteraction(
            user_id=request.user_id,
            project_id=request.project_id,
            interaction_type=request.event_type,
            interaction_score=calculate_interaction_score(request.event_type),
            context=request.metadata or {}
        )

        db.add(interaction)
        db.commit()

        # Update user profile asynchronously
        background_tasks.add_task(update_user_profile, request.user_id, request)

        # Invalidate cached recommendations
        await invalidate_user_cache(request.user_id)

        return {"status": "success", "message": "Interaction recorded"}

    except Exception as e:
        logger.error(f"Failed to learn from interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to record interaction")

@app.get("/users/{user_id}/profile")
async def get_user_profile_endpoint(user_id: str, db: Session = Depends(get_db)):
    """Get user profile and preferences."""
    try:
        user_profile = await get_or_create_user_profile(db, user_id)

        # Get interaction statistics
        interaction_stats = await get_user_interaction_stats(db, user_id)

        # Get recent activity
        recent_interactions = db.query(UserProjectInteraction).filter(
            UserProjectInteraction.user_id == user_id
        ).order_by(UserProjectInteraction.timestamp.desc()).limit(10).all()

        profile_data = {
            "user_id": user_profile.user_id,
            "preferences": user_profile.preferences,
            "interaction_count": user_profile.interaction_count,
            "favorite_projects": user_profile.favorite_projects,
            "preferred_doc_types": user_profile.preferred_doc_types,
            "work_hours": user_profile.work_hours,
            "statistics": interaction_stats,
            "recent_activity": [
                {
                    "project_id": interaction.project_id,
                    "interaction_type": interaction.interaction_type,
                    "timestamp": interaction.timestamp.isoformat()
                }
                for interaction in recent_interactions
            ]
        }

        return profile_data

    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user profile")

@app.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Manually trigger model retraining."""
    background_tasks.add_task(retrain_recommendation_models)
    return {"status": "success", "message": "Model retraining initiated"}

# Helper functions
async def initialize_recommendation_models():
    """Initialize or load recommendation models."""
    global tfidf_vectorizer, project_similarity_matrix, collaborative_model

    try:
        # Try to load from cache
        tfidf_data = redis_client.get("model:tfidf_vectorizer")
        if tfidf_data:
            tfidf_vectorizer = pickle.loads(tfidf_data)
            logger.info("Loaded TF-IDF vectorizer from cache")

        similarity_data = redis_client.get("model:similarity_matrix")
        if similarity_data:
            project_similarity_matrix = pickle.loads(similarity_data)
            logger.info("Loaded similarity matrix from cache")

        if not tfidf_vectorizer or not project_similarity_matrix:
            # Train models from scratch
            await retrain_recommendation_models()

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Initialize empty models
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

async def retrain_recommendation_models():
    """Retrain recommendation models with latest data."""
    global tfidf_vectorizer, project_similarity_matrix, collaborative_model

    try:
        db = SessionLocal()
        try:
            logger.info("Starting model retraining...")

            # Get all project profiles
            projects = db.query(ProjectProfile).all()

            if len(projects) < 2:
                logger.warning("Not enough projects for training")
                return

            # Prepare content data for TF-IDF
            project_texts = []
            project_ids = []

            for project in projects:
                # Combine project name and keywords for content analysis
                text_content = f"{project.project_name} {' '.join(project.keywords or [])}"
                project_texts.append(text_content)
                project_ids.append(project.project_id)

            # Train TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            tfidf_matrix = tfidf_vectorizer.fit_transform(project_texts)

            # Calculate project similarity matrix
            project_similarity_matrix = cosine_similarity(tfidf_matrix)

            # Update project profiles with similarity data
            for i, project in enumerate(projects):
                similarities = project_similarity_matrix[i]
                similar_indices = np.argsort(similarities)[::-1][1:6]  # Top 5 similar (excluding self)

                similar_projects = [
                    {
                        "project_id": project_ids[idx],
                        "similarity_score": float(similarities[idx])
                    }
                    for idx in similar_indices if similarities[idx] > 0.1
                ]

                project.similar_projects = similar_projects
                project.content_vector = tfidf_matrix[i].toarray().flatten().tolist()

            db.commit()

            # Train collaborative filtering model
            await train_collaborative_filtering_model(db)

            # Cache models
            redis_client.setex("model:tfidf_vectorizer", 3600, pickle.dumps(tfidf_vectorizer))
            redis_client.setex("model:similarity_matrix", 3600, pickle.dumps(project_similarity_matrix))

            logger.info("Model retraining completed successfully")

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to retrain models: {e}")

async def train_collaborative_filtering_model(db: Session):
    """Train collaborative filtering model."""
    global user_item_matrix, collaborative_model

    try:
        # Get interaction data
        interactions = db.query(UserProjectInteraction).all()

        if len(interactions) < 10:
            logger.warning("Not enough interactions for collaborative filtering")
            return

        # Create user-item matrix
        user_project_scores = defaultdict(lambda: defaultdict(float))

        for interaction in interactions:
            user_project_scores[interaction.user_id][interaction.project_id] += interaction.interaction_score

        # Convert to matrix format
        users = list(user_project_scores.keys())
        projects = list(set(
            project_id for user_scores in user_project_scores.values()
            for project_id in user_scores.keys()
        ))

        user_item_matrix = np.zeros((len(users), len(projects)))

        for i, user_id in enumerate(users):
            for j, project_id in enumerate(projects):
                user_item_matrix[i, j] = user_project_scores[user_id].get(project_id, 0)

        # Apply matrix factorization (SVD)
        if user_item_matrix.shape[0] > 5 and user_item_matrix.shape[1] > 5:
            collaborative_model = TruncatedSVD(n_components=min(50, min(user_item_matrix.shape) - 1))
            collaborative_model.fit(user_item_matrix)

            # Cache the model
            model_data = {
                "model": collaborative_model,
                "users": users,
                "projects": projects,
                "matrix": user_item_matrix
            }
            redis_client.setex("model:collaborative", 3600, pickle.dumps(model_data))

            logger.info("Collaborative filtering model trained successfully")

    except Exception as e:
        logger.error(f"Failed to train collaborative filtering model: {e}")

async def get_or_create_user_profile(db: Session, user_id: str) -> UserProfile:
    """Get or create user profile."""
    user_profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

    if not user_profile:
        user_profile = UserProfile(user_id=user_id)
        db.add(user_profile)
        db.commit()
        db.refresh(user_profile)

    return user_profile

async def generate_hybrid_recommendations(
    db: Session,
    user_profile: UserProfile,
    context: Dict[str, Any],
    limit: int,
    filters: Optional[Dict[str, Any]]
) -> RecommendationResponse:
    """Generate recommendations using hybrid approach."""
    try:
        # Get recommendations from different algorithms
        content_recs = await get_content_based_recommendations(db, user_profile, limit * 2)
        collaborative_recs = await get_collaborative_recommendations(db, user_profile, limit * 2)
        popularity_recs = await get_popularity_based_recommendations(db, limit)

        # Combine and score recommendations
        recommendation_scores = defaultdict(float)
        algorithm_weights = {"content": 0.4, "collaborative": 0.4, "popularity": 0.2}

        # Content-based recommendations
        for i, rec in enumerate(content_recs):
            score = algorithm_weights["content"] * (1.0 - i / len(content_recs))
            recommendation_scores[rec["project_id"]] += score

        # Collaborative recommendations
        for i, rec in enumerate(collaborative_recs):
            score = algorithm_weights["collaborative"] * (1.0 - i / len(collaborative_recs))
            recommendation_scores[rec["project_id"]] += score

        # Popularity recommendations
        for i, rec in enumerate(popularity_recs):
            score = algorithm_weights["popularity"] * (1.0 - i / len(popularity_recs))
            recommendation_scores[rec["project_id"]] += score

        # Apply context and filters
        recommendation_scores = await apply_context_filters(
            db, recommendation_scores, context, filters
        )

        # Sort and limit recommendations
        sorted_recommendations = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        # Get project details
        final_recommendations = []
        for project_id, score in sorted_recommendations:
            project_details = await get_project_details(db, project_id)
            if project_details:
                project_details["recommendation_score"] = float(score)
                final_recommendations.append(project_details)

        # Calculate confidence score
        confidence_score = min(1.0, len(final_recommendations) / limit)

        return RecommendationResponse(
            recommendations=final_recommendations,
            algorithm_used="hybrid",
            confidence_score=confidence_score,
            context_considered=list(context.keys()) + ["user_profile", "popularity"]
        )

    except Exception as e:
        logger.error(f"Failed to generate hybrid recommendations: {e}")
        # Fallback to popularity-based recommendations
        popularity_recs = await get_popularity_based_recommendations(db, limit)
        return RecommendationResponse(
            recommendations=popularity_recs,
            algorithm_used="popularity_fallback",
            confidence_score=0.5,
            context_considered=["popularity"]
        )

async def get_content_based_recommendations(
    db: Session,
    user_profile: UserProfile,
    limit: int
) -> List[Dict[str, Any]]:
    """Get content-based recommendations."""
    try:
        # Get user's favorite projects
        favorite_projects = user_profile.favorite_projects or []

        if not favorite_projects:
            return []

        # Find similar projects to user's favorites
        similar_projects = []

        for fav_project_id in favorite_projects:
            project_profile = db.query(ProjectProfile).filter(
                ProjectProfile.project_id == fav_project_id
            ).first()

            if project_profile and project_profile.similar_projects:
                similar_projects.extend(project_profile.similar_projects)

        # Sort by similarity score and remove duplicates
        seen_projects = set(favorite_projects)
        recommendations = []

        for similar_project in sorted(similar_projects, key=lambda x: x["similarity_score"], reverse=True):
            if similar_project["project_id"] not in seen_projects:
                recommendations.append(similar_project)
                seen_projects.add(similar_project["project_id"])

                if len(recommendations) >= limit:
                    break

        return recommendations

    except Exception as e:
        logger.error(f"Failed to get content-based recommendations: {e}")
        return []

async def get_collaborative_recommendations(
    db: Session,
    user_profile: UserProfile,
    limit: int
) -> List[Dict[str, Any]]:
    """Get collaborative filtering recommendations."""
    try:
        # Load collaborative model from cache
        model_data = redis_client.get("model:collaborative")
        if not model_data:
            return []

        model_info = pickle.loads(model_data)
        model = model_info["model"]
        users = model_info["users"]
        projects = model_info["projects"]
        user_item_matrix = model_info["matrix"]

        # Find user index
        if user_profile.user_id not in users:
            return []

        user_idx = users.index(user_profile.user_id)

        # Get user's latent factors
        user_factors = model.transform(user_item_matrix[user_idx:user_idx+1])

        # Get project factors
        project_factors = model.components_

        # Calculate scores for all projects
        scores = user_factors.dot(project_factors)

        # Get top recommendations
        top_indices = np.argsort(scores[0])[::-1]

        recommendations = []
        user_interacted_projects = set()

        # Get projects user has already interacted with
        interactions = db.query(UserProjectInteraction).filter(
            UserProjectInteraction.user_id == user_profile.user_id
        ).all()

        for interaction in interactions:
            user_interacted_projects.add(interaction.project_id)

        for idx in top_indices:
            project_id = projects[idx]
            if project_id not in user_interacted_projects:
                recommendations.append({
                    "project_id": project_id,
                    "collaborative_score": float(scores[0][idx])
                })

                if len(recommendations) >= limit:
                    break

        return recommendations

    except Exception as e:
        logger.error(f"Failed to get collaborative recommendations: {e}")
        return []

async def get_popularity_based_recommendations(db: Session, limit: int) -> List[Dict[str, Any]]:
    """Get popularity-based recommendations."""
    try:
        popular_projects = db.query(ProjectProfile).order_by(
            ProjectProfile.popularity_score.desc()
        ).limit(limit).all()

        recommendations = []
        for project in popular_projects:
            project_details = await get_project_details(db, project.project_id)
            if project_details:
                project_details["popularity_score"] = project.popularity_score
                recommendations.append(project_details)

        return recommendations

    except Exception as e:
        logger.error(f"Failed to get popularity recommendations: {e}")
        return []

async def get_project_details(db: Session, project_id: str) -> Optional[Dict[str, Any]]:
    """Get project details."""
    try:
        project_profile = db.query(ProjectProfile).filter(
            ProjectProfile.project_id == project_id
        ).first()

        if project_profile:
            return {
                "project_id": project_profile.project_id,
                "project_name": project_profile.project_name,
                "categories": project_profile.categories,
                "keywords": project_profile.keywords,
                "popularity_score": project_profile.popularity_score
            }

        return None

    except Exception as e:
        logger.error(f"Failed to get project details: {e}")
        return None

async def apply_context_filters(
    db: Session,
    recommendations: Dict[str, float],
    context: Dict[str, Any],
    filters: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    """Apply context and filters to recommendations."""
    try:
        filtered_recommendations = recommendations.copy()

        # Apply time-based context
        current_hour = datetime.utcnow().hour
        if "work_hours" in context:
            # Boost recommendations based on work hours
            for project_id in filtered_recommendations:
                # This is a placeholder - implement work hours logic
                pass

        # Apply category filters
        if filters and "categories" in filters:
            allowed_categories = filters["categories"]
            projects_to_remove = []

            for project_id in filtered_recommendations:
                project_profile = db.query(ProjectProfile).filter(
                    ProjectProfile.project_id == project_id
                ).first()

                if project_profile and project_profile.categories:
                    if not any(cat in allowed_categories for cat in project_profile.categories):
                        projects_to_remove.append(project_id)

            for project_id in projects_to_remove:
                del filtered_recommendations[project_id]

        return filtered_recommendations

    except Exception as e:
        logger.error(f"Failed to apply context filters: {e}")
        return recommendations

def calculate_interaction_score(event_type: str) -> float:
    """Calculate interaction score based on event type."""
    scores = {
        "view": 1.0,
        "search": 1.5,
        "document_access": 2.0,
        "favorite": 3.0,
        "share": 2.5
    }
    return scores.get(event_type, 1.0)

async def update_user_profile(user_id: str, interaction: LearnRequest):
    """Update user profile based on interaction."""
    try:
        db = SessionLocal()
        try:
            user_profile = await get_or_create_user_profile(db, user_id)

            # Update interaction count
            user_profile.interaction_count += 1

            # Update favorite projects if applicable
            if interaction.event_type == "favorite" and interaction.project_id:
                favorites = user_profile.favorite_projects or []
                if interaction.project_id not in favorites:
                    favorites.append(interaction.project_id)
                    user_profile.favorite_projects = favorites[-10:]  # Keep last 10

            # Update preferred document types
            if interaction.event_type == "document_access" and interaction.metadata:
                doc_type = interaction.metadata.get("doc_type")
                if doc_type:
                    preferred_types = user_profile.preferred_doc_types or []
                    if doc_type not in preferred_types:
                        preferred_types.append(doc_type)
                        user_profile.preferred_doc_types = preferred_types

            # Update work hours pattern
            current_hour = datetime.utcnow().hour
            work_hours = user_profile.work_hours or {}
            work_hours[str(current_hour)] = work_hours.get(str(current_hour), 0) + 1
            user_profile.work_hours = work_hours

            user_profile.updated_at = datetime.utcnow()
            db.commit()

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to update user profile: {e}")

async def get_cached_recommendations(
    user_id: str,
    rec_type: str,
    context: Dict[str, Any],
    limit: int
) -> Optional[Dict[str, Any]]:
    """Get cached recommendations if available."""
    try:
        context_hash = str(hash(str(sorted(context.items()))))
        cache_key = f"recommendations:{user_id}:{rec_type}:{context_hash}:{limit}"

        cached_data = redis_client.get(cache_key)
        if cached_data:
            recommendations = pickle.loads(cached_data)
            recommendations["cached"] = True
            return recommendations

        return None

    except Exception as e:
        logger.error(f"Failed to get cached recommendations: {e}")
        return None

async def cache_recommendations(
    user_id: str,
    rec_type: str,
    recommendations: RecommendationResponse,
    context: Dict[str, Any]
):
    """Cache recommendations."""
    try:
        context_hash = str(hash(str(sorted(context.items()))))
        cache_key = f"recommendations:{user_id}:{rec_type}:{context_hash}:{len(recommendations.recommendations)}"

        # Cache for 30 minutes
        redis_client.setex(cache_key, 1800, pickle.dumps(recommendations.dict()))

    except Exception as e:
        logger.error(f"Failed to cache recommendations: {e}")

async def invalidate_user_cache(user_id: str):
    """Invalidate all cached recommendations for a user."""
    try:
        pattern = f"recommendations:{user_id}:*"
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)

    except Exception as e:
        logger.error(f"Failed to invalidate user cache: {e}")

async def get_user_interaction_stats(db: Session, user_id: str) -> Dict[str, Any]:
    """Get user interaction statistics."""
    try:
        # Total interactions
        total_interactions = db.query(func.count(UserProjectInteraction.id)).filter(
            UserProjectInteraction.user_id == user_id
        ).scalar() or 0

        # Interactions by type
        interactions_by_type = db.query(
            UserProjectInteraction.interaction_type,
            func.count(UserProjectInteraction.id)
        ).filter(
            UserProjectInteraction.user_id == user_id
        ).group_by(UserProjectInteraction.interaction_type).all()

        # Most accessed projects
        top_projects = db.query(
            UserProjectInteraction.project_id,
            func.count(UserProjectInteraction.id).label('count')
        ).filter(
            UserProjectInteraction.user_id == user_id,
            UserProjectInteraction.project_id.isnot(None)
        ).group_by(UserProjectInteraction.project_id).order_by(
            func.count(UserProjectInteraction.id).desc()
        ).limit(5).all()

        return {
            "total_interactions": total_interactions,
            "interactions_by_type": {
                interaction_type: count for interaction_type, count in interactions_by_type
            },
            "top_projects": [
                {"project_id": project_id, "access_count": count}
                for project_id, count in top_projects
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get user interaction stats: {e}")
        return {}

async def get_user_document_patterns(
    db: Session,
    user_id: str,
    project_id: str
) -> List[Dict[str, Any]]:
    """Get user's document access patterns for a project."""
    try:
        interactions = db.query(UserProjectInteraction).filter(
            UserProjectInteraction.user_id == user_id,
            UserProjectInteraction.project_id == project_id,
            UserProjectInteraction.interaction_type == "document_access"
        ).order_by(UserProjectInteraction.timestamp.desc()).limit(20).all()

        patterns = []
        for interaction in interactions:
            if interaction.context:
                patterns.append({
                    "doc_type": interaction.context.get("doc_type"),
                    "document_path": interaction.context.get("document_path"),
                    "timestamp": interaction.timestamp.isoformat()
                })

        return patterns

    except Exception as e:
        logger.error(f"Failed to get user document patterns: {e}")
        return []

async def generate_document_recommendations(
    db: Session,
    user_id: str,
    project_id: str,
    preferred_doc_types: List[str],
    recent_docs: List[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """Generate document recommendations for a project."""
    try:
        # This is a simplified implementation
        # In practice, you would integrate with the document navigator service
        recommendations = []

        # Recommend based on preferred document types
        for doc_type in preferred_doc_types[:limit]:
            recommendations.append({
                "doc_type": doc_type,
                "reason": "preferred_type",
                "confidence": 0.8
            })

        # Add recommendations based on recent patterns
        doc_type_counts = Counter(doc["doc_type"] for doc in recent_docs if doc.get("doc_type"))
        for doc_type, count in doc_type_counts.most_common(limit - len(recommendations)):
            if doc_type not in [r["doc_type"] for r in recommendations]:
                recommendations.append({
                    "doc_type": doc_type,
                    "reason": "recent_pattern",
                    "confidence": min(0.9, count / 10.0)
                })

        return recommendations[:limit]

    except Exception as e:
        logger.error(f"Failed to generate document recommendations: {e}")
        return []

async def calculate_project_similarity_onfly(
    db: Session,
    project_id: str,
    limit: int
) -> List[Dict[str, Any]]:
    """Calculate project similarity on the fly."""
    try:
        # This is a simplified version - in practice, use the trained models
        # Get all projects and calculate similarity based on keywords/categories
        target_project = db.query(ProjectProfile).filter(
            ProjectProfile.project_id == project_id
        ).first()

        if not target_project:
            return []

        all_projects = db.query(ProjectProfile).filter(
            ProjectProfile.project_id != project_id
        ).all()

        similarities = []
        target_keywords = set(target_project.keywords or [])
        target_categories = set(target_project.categories or [])

        for project in all_projects:
            project_keywords = set(project.keywords or [])
            project_categories = set(project.categories or [])

            # Simple Jaccard similarity
            keyword_similarity = len(target_keywords & project_keywords) / len(target_keywords | project_keywords) if target_keywords | project_keywords else 0
            category_similarity = len(target_categories & project_categories) / len(target_categories | project_categories) if target_categories | project_categories else 0

            overall_similarity = (keyword_similarity + category_similarity) / 2

            if overall_similarity > 0.1:
                similarities.append({
                    "project_id": project.project_id,
                    "similarity_score": overall_similarity
                })

        # Sort and return top similar projects
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:limit]

    except Exception as e:
        logger.error(f"Failed to calculate project similarity: {e}")
        return []

async def retrain_models_periodically():
    """Periodically retrain recommendation models."""
    while True:
        try:
            # Retrain every 6 hours
            await asyncio.sleep(6 * 3600)
            await retrain_recommendation_models()
            logger.info("Periodic model retraining completed")

        except Exception as e:
            logger.error(f"Periodic retraining failed: {e}")

async def update_popularity_scores():
    """Update popularity scores for projects."""
    while True:
        try:
            await asyncio.sleep(3600)  # Update every hour

            db = SessionLocal()
            try:
                # Calculate popularity based on recent interactions
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)  # Last 30 days

                project_scores = db.query(
                    UserProjectInteraction.project_id,
                    func.count(UserProjectInteraction.id).label('interaction_count'),
                    func.count(func.distinct(UserProjectInteraction.user_id)).label('unique_users')
                ).filter(
                    UserProjectInteraction.timestamp >= start_date,
                    UserProjectInteraction.project_id.isnot(None)
                ).group_by(UserProjectInteraction.project_id).all()

                for project_id, interaction_count, unique_users in project_scores:
                    # Calculate popularity score (weighted by interactions and unique users)
                    popularity_score = (interaction_count * 0.7) + (unique_users * 1.5)

                    # Update project profile
                    project_profile = db.query(ProjectProfile).filter(
                        ProjectProfile.project_id == project_id
                    ).first()

                    if project_profile:
                        project_profile.popularity_score = popularity_score
                        project_profile.updated_at = datetime.utcnow()

                db.commit()

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to update popularity scores: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)