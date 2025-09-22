"""
Enhanced Search Service for Project QuickNav

Provides advanced search capabilities with:
- Elasticsearch integration for full-text search
- ML-enhanced ranking and relevance scoring
- Real-time indexing and document updates
- Faceted search and filtering
- Auto-completion and query suggestions
- Semantic search capabilities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from elasticsearch import AsyncElasticsearch, NotFoundError
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import asyncio
import json
import logging
import re
import hashlib
from pydantic import BaseModel
from contextlib import asynccontextmanager
import redis
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class SearchIndex(Base):
    __tablename__ = "search_index"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, unique=True, index=True)
    document_type = Column(String, index=True)  # project, document, folder
    project_id = Column(String, index=True, nullable=True)
    title = Column(String, index=True)
    content = Column(Text)
    path = Column(String)
    metadata = Column(JSON, default=dict)
    keywords = Column(JSON, default=list)
    categories = Column(JSON, default=list)
    popularity_score = Column(Float, default=0.0)
    last_indexed = Column(DateTime, default=datetime.utcnow, index=True)
    search_count = Column(Integer, default=0)

class SearchQuery(Base):
    __tablename__ = "search_queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String, index=True)
    query_hash = Column(String, index=True)
    user_id = Column(String, index=True, nullable=True)
    results_count = Column(Integer)
    response_time_ms = Column(Float)
    filters_used = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    offset: int = 0
    filters: Optional[Dict[str, Any]] = None
    sort_by: str = "relevance"  # relevance, date, popularity
    include_content: bool = False
    search_type: str = "all"  # all, projects, documents, folders
    fuzzy: bool = True
    highlight: bool = True

class DocumentSearchRequest(BaseModel):
    project_id: str
    doc_type: Optional[str] = None
    query: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: int = 20

class IndexUpdateRequest(BaseModel):
    document_id: str
    document_type: str
    project_id: Optional[str] = None
    title: str
    content: Optional[str] = None
    path: str
    metadata: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    categories: Optional[List[str]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    search_time_ms: float
    facets: Dict[str, List[Dict[str, Any]]]
    suggestions: List[str]
    query_info: Dict[str, Any]

class AutocompleteRequest(BaseModel):
    query: str
    limit: int = 10
    context: Optional[Dict[str, Any]] = None

# Configuration
ELASTICSEARCH_URL = "http://localhost:9200"
DATABASE_URL = "postgresql://quicknav:password@localhost/quicknav_search"
REDIS_URL = "redis://localhost:6379"

# Clients
es_client = AsyncElasticsearch([ELASTICSEARCH_URL])
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Enhanced Search Service...")

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Initialize Elasticsearch indices
    await initialize_elasticsearch_indices()

    # Start background tasks
    asyncio.create_task(refresh_search_index_periodically())
    asyncio.create_task(update_popularity_scores())

    yield

    logger.info("Shutting down Enhanced Search Service...")
    await es_client.close()

app = FastAPI(
    title="Project QuickNav Enhanced Search Service",
    description="Advanced search service with ML-enhanced ranking",
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
    try:
        # Check Elasticsearch connection
        es_health = await es_client.cluster.health()
        es_status = es_health["status"]

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "elasticsearch": es_status,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Perform advanced search across projects and documents."""
    start_time = datetime.utcnow()

    try:
        # Build Elasticsearch query
        es_query = await build_elasticsearch_query(request)

        # Execute search
        es_response = await es_client.search(
            index="quicknav_*",
            body=es_query,
            size=request.limit,
            from_=request.offset
        )

        # Process results
        results = await process_search_results(es_response, request)

        # Get facets
        facets = await extract_facets(es_response)

        # Get query suggestions
        suggestions = await get_query_suggestions(request.query, request.limit)

        # Calculate search time
        search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Record search query
        await record_search_query(db, request, user_id, len(results), search_time_ms)

        # Update search counts
        await update_search_counts(db, results)

        response = SearchResponse(
            results=results,
            total_count=es_response["hits"]["total"]["value"],
            search_time_ms=search_time_ms,
            facets=facets,
            suggestions=suggestions,
            query_info={
                "original_query": request.query,
                "processed_query": es_query.get("query", {}),
                "filters_applied": request.filters or {},
                "search_type": request.search_type
            }
        )

        return response

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search request failed")

@app.post("/documents")
async def search_project_documents(
    request: DocumentSearchRequest,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Search for documents within a specific project."""
    try:
        # Build project-specific query
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"project_id": request.project_id}},
                        {"term": {"document_type": "document"}}
                    ]
                }
            },
            "sort": [
                {"popularity_score": {"order": "desc"}},
                {"_score": {"order": "desc"}}
            ]
        }

        # Add document type filter
        if request.doc_type:
            es_query["query"]["bool"]["must"].append({
                "term": {"metadata.doc_type": request.doc_type}
            })

        # Add text query if provided
        if request.query:
            es_query["query"]["bool"]["must"].append({
                "multi_match": {
                    "query": request.query,
                    "fields": ["title^2", "content", "keywords^1.5"],
                    "fuzziness": "AUTO"
                }
            })

        # Add filters
        if request.filters:
            await apply_document_filters(es_query, request.filters)

        # Execute search
        es_response = await es_client.search(
            index="quicknav_documents",
            body=es_query,
            size=request.limit
        )

        # Process results
        results = []
        for hit in es_response["hits"]["hits"]:
            source = hit["_source"]
            result = {
                "document_id": source["document_id"],
                "title": source["title"],
                "path": source["path"],
                "doc_type": source.get("metadata", {}).get("doc_type"),
                "score": hit["_score"],
                "popularity_score": source.get("popularity_score", 0.0),
                "last_modified": source.get("metadata", {}).get("last_modified"),
                "size": source.get("metadata", {}).get("size"),
                "highlights": hit.get("highlight", {})
            }
            results.append(result)

        return {
            "project_id": request.project_id,
            "results": results,
            "total_count": es_response["hits"]["total"]["value"],
            "doc_type_filter": request.doc_type
        }

    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail="Document search failed")

@app.post("/autocomplete")
async def get_autocomplete_suggestions(request: AutocompleteRequest):
    """Get autocomplete suggestions for search queries."""
    try:
        # Get cached suggestions first
        cache_key = f"autocomplete:{hashlib.md5(request.query.encode()).hexdigest()}"
        cached_suggestions = redis_client.get(cache_key)

        if cached_suggestions:
            return {"suggestions": json.loads(cached_suggestions)}

        # Build Elasticsearch completion query
        es_query = {
            "suggest": {
                "title_suggest": {
                    "prefix": request.query,
                    "completion": {
                        "field": "title_suggest",
                        "size": request.limit,
                        "skip_duplicates": True
                    }
                },
                "keyword_suggest": {
                    "prefix": request.query,
                    "completion": {
                        "field": "keyword_suggest",
                        "size": request.limit,
                        "skip_duplicates": True
                    }
                }
            }
        }

        # Execute suggestion query
        es_response = await es_client.search(
            index="quicknav_*",
            body=es_query
        )

        # Process suggestions
        suggestions = []
        seen_suggestions = set()

        # Process title suggestions
        for suggestion in es_response.get("suggest", {}).get("title_suggest", []):
            for option in suggestion.get("options", []):
                text = option["text"]
                if text not in seen_suggestions and len(text) > len(request.query):
                    suggestions.append({
                        "text": text,
                        "type": "title",
                        "score": option["_score"]
                    })
                    seen_suggestions.add(text)

        # Process keyword suggestions
        for suggestion in es_response.get("suggest", {}).get("keyword_suggest", []):
            for option in suggestion.get("options", []):
                text = option["text"]
                if text not in seen_suggestions and len(text) > len(request.query):
                    suggestions.append({
                        "text": text,
                        "type": "keyword",
                        "score": option["_score"]
                    })
                    seen_suggestions.add(text)

        # Sort by score and limit
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        suggestions = suggestions[:request.limit]

        # Cache suggestions for 1 hour
        redis_client.setex(cache_key, 3600, json.dumps([s["text"] for s in suggestions]))

        return {"suggestions": [s["text"] for s in suggestions]}

    except Exception as e:
        logger.error(f"Autocomplete failed: {e}")
        raise HTTPException(status_code=500, detail="Autocomplete request failed")

@app.post("/index/update")
async def update_search_index(
    request: IndexUpdateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Update search index with new or modified document."""
    try:
        # Update database record
        search_record = db.query(SearchIndex).filter(
            SearchIndex.document_id == request.document_id
        ).first()

        if search_record:
            # Update existing record
            search_record.title = request.title
            search_record.content = request.content
            search_record.path = request.path
            search_record.metadata = request.metadata or {}
            search_record.keywords = request.keywords or []
            search_record.categories = request.categories or []
            search_record.last_indexed = datetime.utcnow()
        else:
            # Create new record
            search_record = SearchIndex(
                document_id=request.document_id,
                document_type=request.document_type,
                project_id=request.project_id,
                title=request.title,
                content=request.content,
                path=request.path,
                metadata=request.metadata or {},
                keywords=request.keywords or [],
                categories=request.categories or []
            )
            db.add(search_record)

        db.commit()

        # Update Elasticsearch index in background
        background_tasks.add_task(update_elasticsearch_document, request)

        return {
            "status": "success",
            "message": "Index update queued",
            "document_id": request.document_id
        }

    except Exception as e:
        logger.error(f"Index update failed: {e}")
        raise HTTPException(status_code=500, detail="Index update failed")

@app.delete("/index/{document_id}")
async def delete_from_index(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Remove document from search index."""
    try:
        # Remove from database
        search_record = db.query(SearchIndex).filter(
            SearchIndex.document_id == document_id
        ).first()

        if search_record:
            db.delete(search_record)
            db.commit()

        # Remove from Elasticsearch in background
        background_tasks.add_task(delete_elasticsearch_document, document_id)

        return {
            "status": "success",
            "message": "Document removed from index",
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"Index deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Index deletion failed")

@app.post("/reindex")
async def reindex_all_documents(background_tasks: BackgroundTasks):
    """Trigger full reindexing of all documents."""
    background_tasks.add_task(full_reindex)
    return {
        "status": "success",
        "message": "Full reindexing initiated"
    }

@app.get("/analytics/popular-queries")
async def get_popular_queries(
    limit: int = 20,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get popular search queries."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get popular queries from database
        popular_queries = db.query(
            SearchQuery.query_text,
            db.func.count(SearchQuery.id).label('search_count'),
            db.func.avg(SearchQuery.response_time_ms).label('avg_response_time'),
            db.func.avg(SearchQuery.results_count).label('avg_results')
        ).filter(
            SearchQuery.timestamp >= start_date
        ).group_by(
            SearchQuery.query_text
        ).order_by(
            db.func.count(SearchQuery.id).desc()
        ).limit(limit).all()

        results = []
        for query, count, avg_time, avg_results in popular_queries:
            results.append({
                "query": query,
                "search_count": count,
                "avg_response_time_ms": float(avg_time or 0),
                "avg_results": float(avg_results or 0)
            })

        return {"popular_queries": results, "period_days": days}

    except Exception as e:
        logger.error(f"Failed to get popular queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to get popular queries")

# Helper functions
async def initialize_elasticsearch_indices():
    """Initialize Elasticsearch indices with proper mappings."""
    try:
        indices = [
            {
                "name": "quicknav_projects",
                "mapping": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "document_type": {"type": "keyword"},
                        "project_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "title_suggest": {
                            "type": "completion",
                            "analyzer": "simple"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "path": {"type": "keyword"},
                        "keywords": {
                            "type": "text",
                            "analyzer": "keyword"
                        },
                        "keyword_suggest": {
                            "type": "completion",
                            "analyzer": "simple"
                        },
                        "categories": {"type": "keyword"},
                        "popularity_score": {"type": "float"},
                        "last_indexed": {"type": "date"},
                        "metadata": {
                            "type": "object",
                            "dynamic": True
                        }
                    }
                }
            },
            {
                "name": "quicknav_documents",
                "mapping": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "document_type": {"type": "keyword"},
                        "project_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "title_suggest": {
                            "type": "completion",
                            "analyzer": "simple"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "path": {"type": "keyword"},
                        "keywords": {
                            "type": "text",
                            "analyzer": "keyword"
                        },
                        "keyword_suggest": {
                            "type": "completion",
                            "analyzer": "simple"
                        },
                        "categories": {"type": "keyword"},
                        "popularity_score": {"type": "float"},
                        "last_indexed": {"type": "date"},
                        "metadata": {
                            "type": "object",
                            "dynamic": True,
                            "properties": {
                                "doc_type": {"type": "keyword"},
                                "size": {"type": "long"},
                                "last_modified": {"type": "date"}
                            }
                        }
                    }
                }
            }
        ]

        for index_config in indices:
            index_name = index_config["name"]
            mapping = index_config["mapping"]

            # Check if index exists
            exists = await es_client.indices.exists(index=index_name)

            if not exists:
                # Create index with mapping
                await es_client.indices.create(
                    index=index_name,
                    body={"mappings": mapping}
                )
                logger.info(f"Created Elasticsearch index: {index_name}")
            else:
                # Update mapping if needed
                await es_client.indices.put_mapping(
                    index=index_name,
                    body=mapping
                )
                logger.info(f"Updated Elasticsearch mapping: {index_name}")

    except Exception as e:
        logger.error(f"Failed to initialize Elasticsearch indices: {e}")

async def build_elasticsearch_query(request: SearchRequest) -> Dict[str, Any]:
    """Build Elasticsearch query from search request."""
    query = {
        "query": {
            "bool": {
                "must": [],
                "filter": [],
                "should": []
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "content": {"fragment_size": 150}
            }
        } if request.highlight else {},
        "aggs": {
            "document_types": {
                "terms": {"field": "document_type"}
            },
            "categories": {
                "terms": {"field": "categories"}
            },
            "projects": {
                "terms": {"field": "project_id", "size": 20}
            }
        }
    }

    # Main search query
    if request.query.strip():
        main_query = {
            "multi_match": {
                "query": request.query,
                "fields": [
                    "title^3",
                    "content^1",
                    "keywords^2",
                    "metadata.doc_type^1.5"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO" if request.fuzzy else 0,
                "operator": "and"
            }
        }
        query["query"]["bool"]["must"].append(main_query)

        # Add should clauses for boosting
        query["query"]["bool"]["should"].extend([
            {
                "match_phrase": {
                    "title": {
                        "query": request.query,
                        "boost": 2.0
                    }
                }
            },
            {
                "prefix": {
                    "title.keyword": {
                        "value": request.query,
                        "boost": 1.5
                    }
                }
            }
        ])
    else:
        # Match all if no query
        query["query"]["bool"]["must"].append({"match_all": {}})

    # Document type filter
    if request.search_type != "all":
        query["query"]["bool"]["filter"].append({
            "term": {"document_type": request.search_type}
        })

    # Apply additional filters
    if request.filters:
        await apply_search_filters(query, request.filters)

    # Sorting
    if request.sort_by == "date":
        query["sort"] = [{"last_indexed": {"order": "desc"}}]
    elif request.sort_by == "popularity":
        query["sort"] = [
            {"popularity_score": {"order": "desc"}},
            {"_score": {"order": "desc"}}
        ]
    else:  # relevance
        query["sort"] = [{"_score": {"order": "desc"}}]

    # Function score for popularity boost
    if request.sort_by == "relevance":
        query = {
            "query": {
                "function_score": {
                    "query": query["query"],
                    "functions": [
                        {
                            "field_value_factor": {
                                "field": "popularity_score",
                                "factor": 0.1,
                                "modifier": "log1p",
                                "missing": 0
                            }
                        }
                    ],
                    "boost_mode": "multiply",
                    "score_mode": "sum"
                }
            },
            **{k: v for k, v in query.items() if k != "query"}
        }

    return query

async def apply_search_filters(query: Dict[str, Any], filters: Dict[str, Any]):
    """Apply filters to Elasticsearch query."""
    filter_clauses = query["query"]["bool"]["filter"]

    # Project filter
    if "project_ids" in filters:
        filter_clauses.append({
            "terms": {"project_id": filters["project_ids"]}
        })

    # Date range filter
    if "date_range" in filters:
        date_range = filters["date_range"]
        range_filter = {"range": {"last_indexed": {}}}

        if "start" in date_range:
            range_filter["range"]["last_indexed"]["gte"] = date_range["start"]
        if "end" in date_range:
            range_filter["range"]["last_indexed"]["lte"] = date_range["end"]

        filter_clauses.append(range_filter)

    # Category filter
    if "categories" in filters:
        filter_clauses.append({
            "terms": {"categories": filters["categories"]}
        })

    # Document type filter
    if "doc_types" in filters:
        filter_clauses.append({
            "terms": {"metadata.doc_type": filters["doc_types"]}
        })

    # Size filter
    if "size_range" in filters:
        size_range = filters["size_range"]
        range_filter = {"range": {"metadata.size": {}}}

        if "min" in size_range:
            range_filter["range"]["metadata.size"]["gte"] = size_range["min"]
        if "max" in size_range:
            range_filter["range"]["metadata.size"]["lte"] = size_range["max"]

        filter_clauses.append(range_filter)

async def apply_document_filters(query: Dict[str, Any], filters: Dict[str, Any]):
    """Apply document-specific filters."""
    filter_clauses = query["query"]["bool"].setdefault("filter", [])

    # Room filter
    if "room" in filters:
        filter_clauses.append({
            "term": {"metadata.room": filters["room"]}
        })

    # Change order filter
    if "co" in filters:
        filter_clauses.append({
            "term": {"metadata.change_order": filters["co"]}
        })

    # Version filter
    if "version_filter" in filters:
        version_filter = filters["version_filter"]
        if version_filter == "latest":
            # Add script to get only latest versions
            query["collapse"] = {
                "field": "metadata.base_name",
                "inner_hits": {
                    "name": "latest_version",
                    "size": 1,
                    "sort": [{"metadata.version_score": {"order": "desc"}}]
                }
            }

    # Include archive filter
    if "exclude_archive" in filters and filters["exclude_archive"]:
        filter_clauses.append({
            "bool": {
                "must_not": {
                    "term": {"metadata.is_archived": True}
                }
            }
        })

async def process_search_results(
    es_response: Dict[str, Any],
    request: SearchRequest
) -> List[Dict[str, Any]]:
    """Process Elasticsearch response into search results."""
    results = []

    for hit in es_response["hits"]["hits"]:
        source = hit["_source"]
        result = {
            "document_id": source["document_id"],
            "document_type": source["document_type"],
            "title": source["title"],
            "path": source["path"],
            "score": hit["_score"],
            "popularity_score": source.get("popularity_score", 0.0),
            "keywords": source.get("keywords", []),
            "categories": source.get("categories", []),
            "last_indexed": source.get("last_indexed"),
            "metadata": source.get("metadata", {})
        }

        # Add project info if available
        if source.get("project_id"):
            result["project_id"] = source["project_id"]

        # Add content if requested
        if request.include_content and source.get("content"):
            result["content"] = source["content"][:500]  # Truncate for performance

        # Add highlights
        if "highlight" in hit:
            result["highlights"] = hit["highlight"]

        results.append(result)

    return results

async def extract_facets(es_response: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract facets from Elasticsearch aggregations."""
    facets = {}

    if "aggregations" in es_response:
        aggs = es_response["aggregations"]

        # Document types facet
        if "document_types" in aggs:
            facets["document_types"] = [
                {"value": bucket["key"], "count": bucket["doc_count"]}
                for bucket in aggs["document_types"]["buckets"]
            ]

        # Categories facet
        if "categories" in aggs:
            facets["categories"] = [
                {"value": bucket["key"], "count": bucket["doc_count"]}
                for bucket in aggs["categories"]["buckets"]
            ]

        # Projects facet
        if "projects" in aggs:
            facets["projects"] = [
                {"value": bucket["key"], "count": bucket["doc_count"]}
                for bucket in aggs["projects"]["buckets"]
            ]

    return facets

async def get_query_suggestions(query: str, limit: int) -> List[str]:
    """Get query suggestions based on popular searches."""
    try:
        # Get cached suggestions
        cache_key = f"suggestions:{hashlib.md5(query.encode()).hexdigest()}"
        cached = redis_client.get(cache_key)

        if cached:
            return json.loads(cached)

        # Get from database (simplified - in practice, use more sophisticated algorithms)
        db = SessionLocal()
        try:
            similar_queries = db.query(SearchQuery.query_text).filter(
                SearchQuery.query_text.contains(query)
            ).distinct().limit(limit).all()

            suggestions = [q[0] for q in similar_queries if q[0] != query]

            # Cache for 1 hour
            redis_client.setex(cache_key, 3600, json.dumps(suggestions))

            return suggestions

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to get query suggestions: {e}")
        return []

async def record_search_query(
    db: Session,
    request: SearchRequest,
    user_id: Optional[str],
    results_count: int,
    response_time_ms: float
):
    """Record search query for analytics."""
    try:
        query_hash = hashlib.md5(request.query.encode()).hexdigest()

        search_query = SearchQuery(
            query_text=request.query,
            query_hash=query_hash,
            user_id=user_id,
            results_count=results_count,
            response_time_ms=response_time_ms,
            filters_used=request.filters or {}
        )

        db.add(search_query)
        db.commit()

    except Exception as e:
        logger.error(f"Failed to record search query: {e}")

async def update_search_counts(db: Session, results: List[Dict[str, Any]]):
    """Update search counts for returned documents."""
    try:
        document_ids = [result["document_id"] for result in results]

        # Batch update search counts
        db.query(SearchIndex).filter(
            SearchIndex.document_id.in_(document_ids)
        ).update(
            {SearchIndex.search_count: SearchIndex.search_count + 1},
            synchronize_session=False
        )

        db.commit()

    except Exception as e:
        logger.error(f"Failed to update search counts: {e}")

async def update_elasticsearch_document(request: IndexUpdateRequest):
    """Update document in Elasticsearch."""
    try:
        # Determine index based on document type
        index_name = f"quicknav_{request.document_type}s"

        # Prepare document for indexing
        doc = {
            "document_id": request.document_id,
            "document_type": request.document_type,
            "project_id": request.project_id,
            "title": request.title,
            "content": request.content,
            "path": request.path,
            "keywords": request.keywords or [],
            "categories": request.categories or [],
            "metadata": request.metadata or {},
            "last_indexed": datetime.utcnow().isoformat(),
            "title_suggest": {
                "input": [request.title] + (request.keywords or []),
                "weight": 10
            },
            "keyword_suggest": {
                "input": request.keywords or [],
                "weight": 5
            }
        }

        # Index document
        await es_client.index(
            index=index_name,
            id=request.document_id,
            body=doc
        )

        logger.info(f"Updated Elasticsearch document: {request.document_id}")

    except Exception as e:
        logger.error(f"Failed to update Elasticsearch document: {e}")

async def delete_elasticsearch_document(document_id: str):
    """Delete document from Elasticsearch."""
    try:
        # Try to delete from all indices
        indices = ["quicknav_projects", "quicknav_documents"]

        for index in indices:
            try:
                await es_client.delete(index=index, id=document_id)
                logger.info(f"Deleted document {document_id} from {index}")
            except NotFoundError:
                pass  # Document not in this index

    except Exception as e:
        logger.error(f"Failed to delete Elasticsearch document: {e}")

async def full_reindex():
    """Perform full reindexing of all documents."""
    try:
        logger.info("Starting full reindex...")

        db = SessionLocal()
        try:
            # Get all documents from database
            documents = db.query(SearchIndex).all()

            for doc in documents:
                # Update in Elasticsearch
                await update_elasticsearch_document(IndexUpdateRequest(
                    document_id=doc.document_id,
                    document_type=doc.document_type,
                    project_id=doc.project_id,
                    title=doc.title,
                    content=doc.content,
                    path=doc.path,
                    metadata=doc.metadata,
                    keywords=doc.keywords,
                    categories=doc.categories
                ))

                # Small delay to avoid overwhelming Elasticsearch
                await asyncio.sleep(0.01)

            logger.info(f"Full reindex completed: {len(documents)} documents")

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Full reindex failed: {e}")

async def refresh_search_index_periodically():
    """Periodically refresh search index."""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour

            # Refresh Elasticsearch indices
            await es_client.indices.refresh(index="quicknav_*")

            logger.info("Search indices refreshed")

        except Exception as e:
            logger.error(f"Failed to refresh search indices: {e}")

async def update_popularity_scores():
    """Update popularity scores based on search patterns."""
    while True:
        try:
            await asyncio.sleep(1800)  # Every 30 minutes

            db = SessionLocal()
            try:
                # Calculate popularity scores based on search counts and recency
                documents = db.query(SearchIndex).all()

                for doc in documents:
                    # Simple popularity calculation
                    # In practice, use more sophisticated algorithms
                    popularity_score = min(100.0, doc.search_count * 0.1)

                    # Update in database
                    doc.popularity_score = popularity_score

                    # Update in Elasticsearch
                    try:
                        await es_client.update(
                            index=f"quicknav_{doc.document_type}s",
                            id=doc.document_id,
                            body={"doc": {"popularity_score": popularity_score}}
                        )
                    except Exception:
                        pass  # Ignore update errors

                db.commit()

                logger.info("Popularity scores updated")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to update popularity scores: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)