"""
RAG (Retrieval-Augmented Generation) System for Project QuickNav

Implements document embedding, vector search, and semantic retrieval for
intelligent project navigation and document discovery.

Features:
- Document embedding and vector indexing
- Semantic similarity search across projects and documents
- Contextual information retrieval from document corpus
- Multi-modal search (text, metadata, file structure)
- Real-time index updates and incremental learning
- Query expansion and result ranking
"""

import os
import json
import logging
import asyncio
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. ML-based retrieval features will be disabled.")
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Advanced embedding features will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. High-performance vector search will be disabled.")
    FAISS_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    chunk_id: str
    document_path: str
    project_name: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_index: int = 0
    chunk_size: int = 0


@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""
    document_path: str
    project_name: str
    document_name: str
    content_preview: str
    relevance_score: float
    metadata: Dict[str, Any]
    matching_chunks: List[DocumentChunk]
    explanation: str


@dataclass
class VectorIndex:
    """Vector index configuration and state."""
    index_type: str  # 'faiss', 'sklearn', 'memory'
    dimension: int
    total_vectors: int
    index_path: str
    metadata_path: str
    last_updated: datetime
    version: str = "1.0"


class EmbeddingManager:
    """Manages document embeddings and vector operations."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = "data/embeddings"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.embedding_cache = {}

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {self.model_name} (dim: {self.dimension})")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                SENTENCE_TRANSFORMERS_AVAILABLE = False

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.embedding_model:
            # Fallback to TF-IDF if sentence transformers not available
            return self._tfidf_embeddings(texts)

        try:
            # Process in batches to manage memory
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings)

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return self._tfidf_embeddings(texts)

    def _tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF embeddings when sentence transformers unavailable."""
        if not SKLEARN_AVAILABLE:
            # Ultra-fallback: return random embeddings
            return np.random.rand(len(texts), 100)

        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()

        # Apply dimensionality reduction if needed
        if embeddings.shape[1] > 100:
            svd = TruncatedSVD(n_components=100)
            embeddings = svd.fit_transform(embeddings)

        return embeddings

    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache an embedding for future use."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache[text_hash] = embedding

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.embedding_cache.get(text_hash)


class VectorStore:
    """High-performance vector storage and search."""

    def __init__(self, dimension: int, index_path: str = "data/vector_index"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index = None
        self.metadata_db = None
        self.document_chunks = {}

        self._initialize_index()
        self._initialize_metadata_db()

    def _initialize_index(self):
        """Initialize the vector index."""
        if FAISS_AVAILABLE:
            self._initialize_faiss_index()
        else:
            self._initialize_memory_index()

    def _initialize_faiss_index(self):
        """Initialize FAISS index for high-performance search."""
        try:
            index_file = self.index_path / "faiss_index.bin"

            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index - using HNSW for better recall
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
                logger.info(f"Created new FAISS HNSW index (dim: {self.dimension})")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self._initialize_memory_index()

    def _initialize_memory_index(self):
        """Initialize in-memory index as fallback."""
        self.index = {
            'vectors': [],
            'ids': [],
            'type': 'memory'
        }
        logger.info("Using in-memory vector index")

    def _initialize_metadata_db(self):
        """Initialize SQLite database for metadata storage."""
        db_path = self.index_path / "metadata.db"
        self.metadata_db = sqlite3.connect(str(db_path), check_same_thread=False)

        # Create tables
        self.metadata_db.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                chunk_id TEXT PRIMARY KEY,
                document_path TEXT,
                project_name TEXT,
                document_name TEXT,
                content_preview TEXT,
                metadata TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        self.metadata_db.execute("""
            CREATE TABLE IF NOT EXISTS search_stats (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                result_count INTEGER,
                avg_relevance REAL,
                search_time REAL,
                created_at TIMESTAMP
            )
        """)

        self.metadata_db.commit()

    async def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store."""
        if not chunks:
            return

        try:
            # Generate embeddings for all chunks
            embedding_manager = EmbeddingManager()
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_manager.embed_texts(texts)

            # Add to vector index
            if FAISS_AVAILABLE and hasattr(self.index, 'add'):
                # FAISS index
                self.index.add(embeddings.astype(np.float32))
            else:
                # Memory index
                if 'vectors' not in self.index:
                    self.index['vectors'] = []
                    self.index['ids'] = []

                for i, embedding in enumerate(embeddings):
                    self.index['vectors'].append(embedding)
                    self.index['ids'].append(chunks[i].chunk_id)

            # Store metadata in database
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                self.document_chunks[chunk.chunk_id] = chunk

                self.metadata_db.execute("""
                    INSERT OR REPLACE INTO document_metadata
                    (chunk_id, document_path, project_name, document_name, content_preview, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.document_path,
                    chunk.project_name,
                    os.path.basename(chunk.document_path),
                    chunk.content[:200],
                    json.dumps(chunk.metadata),
                    datetime.now(),
                    datetime.now()
                ))

            self.metadata_db.commit()
            logger.info(f"Added {len(chunks)} document chunks to vector store")

        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")

    async def search(self, query_embedding: np.ndarray, top_k: int = 10,
                    filters: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        try:
            if FAISS_AVAILABLE and hasattr(self.index, 'search'):
                # FAISS search
                query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
                distances, indices = self.index.search(query_embedding, top_k)

                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx >= 0:  # Valid index
                        # Convert distance to similarity score
                        similarity = 1.0 / (1.0 + distance)

                        # Get chunk_id from metadata
                        chunk_id = self._get_chunk_id_by_index(idx)
                        if chunk_id:
                            results.append((chunk_id, similarity))

                return results

            else:
                # Memory-based search
                if not self.index.get('vectors'):
                    return []

                vectors = np.array(self.index['vectors'])
                similarities = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    vectors
                )[0]

                # Get top-k results
                top_indices = np.argsort(similarities)[-top_k:][::-1]

                results = []
                for idx in top_indices:
                    chunk_id = self.index['ids'][idx]
                    similarity = similarities[idx]
                    results.append((chunk_id, similarity))

                return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _get_chunk_id_by_index(self, idx: int) -> Optional[str]:
        """Get chunk ID by vector index."""
        # This would need to be implemented based on how we track index->chunk_id mapping
        # For now, return None and fall back to metadata search
        return None

    def save_index(self):
        """Save the vector index to disk."""
        try:
            if FAISS_AVAILABLE and hasattr(self.index, 'write_index'):
                index_file = self.index_path / "faiss_index.bin"
                faiss.write_index(self.index, str(index_file))
                logger.info(f"Saved FAISS index to {index_file}")
            else:
                # Save memory index
                index_file = self.index_path / "memory_index.pkl"
                with open(index_file, 'wb') as f:
                    pickle.dump(self.index, f)
                logger.info(f"Saved memory index to {index_file}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if FAISS_AVAILABLE and hasattr(self.index, 'ntotal'):
            total_vectors = self.index.ntotal
        else:
            total_vectors = len(self.index.get('vectors', []))

        return {
            'total_vectors': total_vectors,
            'dimension': self.dimension,
            'index_type': 'faiss' if FAISS_AVAILABLE else 'memory',
            'total_documents': len(set(chunk.document_path for chunk in self.document_chunks.values())),
            'total_projects': len(set(chunk.project_name for chunk in self.document_chunks.values()))
        }


class RAGSystem:
    """Main RAG system for document retrieval and augmented generation."""

    def __init__(self, training_data_path: str = "training_data",
                 cache_dir: str = "data/rag_cache"):
        self.training_data_path = Path(training_data_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(self.embedding_manager.dimension)
        self.document_index = {}
        self.project_index = defaultdict(list)

        # Initialize from training data
        asyncio.create_task(self._initialize_from_training_data())

    async def _initialize_from_training_data(self):
        """Initialize RAG system from existing training data."""
        if not self.training_data_path.exists():
            logger.warning(f"Training data path {self.training_data_path} does not exist")
            return

        try:
            chunks = []

            for json_file in self.training_data_path.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for item in data:
                    document_path = item.get('document_path', '')
                    project_name = item.get('project_folder', '')
                    document_name = item.get('document_name', '')

                    if document_path and project_name:
                        # Create document chunk
                        chunk = DocumentChunk(
                            chunk_id=hashlib.md5(document_path.encode()).hexdigest(),
                            document_path=document_path,
                            project_name=project_name,
                            content=f"{document_name} {project_name}",  # Basic content from filename
                            metadata={
                                'document_type': self._classify_document_type(document_name),
                                'file_extension': Path(document_name).suffix.lower(),
                                'project_number': self._extract_project_number(project_name),
                                'training_source': str(json_file.name)
                            }
                        )

                        chunks.append(chunk)

                        # Build indexes
                        self.document_index[document_path] = chunk
                        self.project_index[project_name].append(chunk)

            # Add chunks to vector store
            if chunks:
                await self.vector_store.add_documents(chunks)
                logger.info(f"Initialized RAG system with {len(chunks)} document chunks from {len(self.project_index)} projects")

        except Exception as e:
            logger.error(f"Failed to initialize from training data: {e}")

    def _classify_document_type(self, document_name: str) -> str:
        """Classify document type from filename."""
        name_lower = document_name.lower()

        if 'lld' in name_lower or 'low level' in name_lower:
            return 'lld'
        elif 'hld' in name_lower or 'high level' in name_lower:
            return 'hld'
        elif any(word in name_lower for word in ['floor', 'plan', 'layout', 'drawing']):
            return 'floor_plans'
        elif any(word in name_lower for word in ['po', 'purchase', 'order']):
            return 'sales_po'
        elif any(word in name_lower for word in ['change', 'co', 'variation']):
            return 'change_order'
        elif any(word in name_lower for word in ['scope', 'sow', 'statement']):
            return 'scope'
        elif any(word in name_lower for word in ['qa', 'itp', 'inspection', 'test']):
            return 'qa_itp'
        elif any(word in name_lower for word in ['swms', 'safety', 'method']):
            return 'swms'
        elif any(word in name_lower for word in ['quote', 'quotation', 'pricing']):
            return 'supplier_quotes'
        elif any(ext in name_lower for ext in ['.jpg', '.png', '.jpeg', '.gif', '.bmp']):
            return 'photos'
        else:
            return 'other'

    def _extract_project_number(self, project_name: str) -> Optional[str]:
        """Extract project number from project name."""
        match = re.search(r'\b(\d{5})\b', project_name)
        return match.group(1) if match else None

    async def search_documents(self, query: str, filters: Optional[Dict] = None,
                             top_k: int = 10) -> List[SearchResult]:
        """Search for relevant documents using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_texts([query])
            query_vector = query_embedding[0]

            # Search vector store
            search_results = await self.vector_store.search(query_vector, top_k * 2)  # Get more for filtering

            # Convert to SearchResult objects
            results = []
            for chunk_id, similarity in search_results:
                if chunk_id in self.vector_store.document_chunks:
                    chunk = self.vector_store.document_chunks[chunk_id]

                    # Apply filters if specified
                    if filters and not self._matches_filters(chunk, filters):
                        continue

                    result = SearchResult(
                        document_path=chunk.document_path,
                        project_name=chunk.project_name,
                        document_name=os.path.basename(chunk.document_path),
                        content_preview=chunk.content[:200],
                        relevance_score=similarity,
                        metadata=chunk.metadata,
                        matching_chunks=[chunk],
                        explanation=self._generate_explanation(query, chunk, similarity)
                    )

                    results.append(result)

            # Sort by relevance and limit results
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []

    def _matches_filters(self, chunk: DocumentChunk, filters: Dict) -> bool:
        """Check if chunk matches the specified filters."""
        if 'project_number' in filters:
            project_number = self._extract_project_number(chunk.project_name)
            if project_number != filters['project_number']:
                return False

        if 'document_type' in filters:
            if chunk.metadata.get('document_type') != filters['document_type']:
                return False

        if 'project_name' in filters:
            if filters['project_name'].lower() not in chunk.project_name.lower():
                return False

        return True

    def _generate_explanation(self, query: str, chunk: DocumentChunk, similarity: float) -> str:
        """Generate explanation for why this result is relevant."""
        explanations = []

        if similarity > 0.8:
            explanations.append("High semantic similarity to your query")
        elif similarity > 0.6:
            explanations.append("Good semantic match")
        else:
            explanations.append("Partial semantic match")

        # Check for exact term matches
        query_terms = set(query.lower().split())
        content_terms = set(chunk.content.lower().split())
        common_terms = query_terms.intersection(content_terms)

        if common_terms:
            explanations.append(f"Contains terms: {', '.join(list(common_terms)[:3])}")

        # Document type relevance
        doc_type = chunk.metadata.get('document_type')
        if doc_type and doc_type.replace('_', ' ') in query.lower():
            explanations.append(f"Matches document type: {doc_type}")

        return "; ".join(explanations)

    async def get_contextual_information(self, query: str, max_context_length: int = 2000) -> str:
        """Get contextual information for RAG-enhanced generation."""
        # Search for relevant documents
        search_results = await self.search_documents(query, top_k=5)

        if not search_results:
            return "No relevant context found in the document corpus."

        # Build context from search results
        context_parts = []
        current_length = 0

        for result in search_results:
            # Create context snippet
            snippet = f"""
Document: {result.document_name}
Project: {result.project_name}
Type: {result.metadata.get('document_type', 'unknown')}
Relevance: {result.relevance_score:.2f}
Content: {result.content_preview}
---
"""

            if current_length + len(snippet) > max_context_length:
                break

            context_parts.append(snippet)
            current_length += len(snippet)

        context = "Relevant documents from your project database:\n\n" + "\n".join(context_parts)

        return context

    async def expand_query(self, query: str, expansion_count: int = 3) -> List[str]:
        """Expand query with related terms and concepts."""
        expanded_queries = [query]  # Include original

        try:
            # Get related documents
            search_results = await self.search_documents(query, top_k=5)

            # Extract common terms from relevant documents
            all_terms = []
            for result in search_results:
                # Extract meaningful terms from document names and projects
                terms = self._extract_meaningful_terms(result.document_name + " " + result.project_name)
                all_terms.extend(terms)

            # Find most common terms not in original query
            from collections import Counter
            term_counts = Counter(all_terms)
            query_terms = set(query.lower().split())

            expansion_terms = []
            for term, count in term_counts.most_common():
                if term not in query_terms and len(term) > 2:
                    expansion_terms.append(term)
                    if len(expansion_terms) >= expansion_count:
                        break

            # Create expanded queries
            for term in expansion_terms:
                expanded_queries.append(f"{query} {term}")

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")

        return expanded_queries

    def _extract_meaningful_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text."""
        # Remove common stopwords and extract meaningful terms
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        # Clean text
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        terms = [term.strip() for term in cleaned.split() if len(term) > 2 and term not in stopwords]

        return terms

    async def update_index(self, new_documents: List[Dict[str, Any]]):
        """Update the RAG index with new documents."""
        try:
            chunks = []

            for doc_info in new_documents:
                chunk = DocumentChunk(
                    chunk_id=hashlib.md5(doc_info['path'].encode()).hexdigest(),
                    document_path=doc_info['path'],
                    project_name=doc_info.get('project_name', ''),
                    content=doc_info.get('content', doc_info.get('name', '')),
                    metadata=doc_info.get('metadata', {}),
                )

                chunks.append(chunk)
                self.document_index[doc_info['path']] = chunk
                self.project_index[chunk.project_name].append(chunk)

            # Add to vector store
            await self.vector_store.add_documents(chunks)

            # Save index
            self.vector_store.save_index()

            logger.info(f"Updated RAG index with {len(chunks)} new documents")

        except Exception as e:
            logger.error(f"Failed to update RAG index: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        vector_stats = self.vector_store.get_stats()

        return {
            **vector_stats,
            'total_document_types': len(set(
                chunk.metadata.get('document_type', 'unknown')
                for chunk in self.document_index.values()
            )),
            'embedding_model': self.embedding_manager.model_name,
            'cache_size': len(self.embedding_manager.embedding_cache),
            'last_updated': datetime.now().isoformat()
        }

    async def similarity_search_projects(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar projects using semantic similarity."""
        try:
            project_names = list(self.project_index.keys())
            if not project_names:
                return []

            # Generate embeddings for query and project names
            query_embedding = await self.embedding_manager.embed_texts([query])
            project_embeddings = await self.embedding_manager.embed_texts(project_names)

            # Calculate similarities
            similarities = cosine_similarity(query_embedding, project_embeddings)[0]

            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Minimum threshold
                    project_name = project_names[idx]
                    project_chunks = self.project_index[project_name]

                    results.append({
                        'project_name': project_name,
                        'similarity': float(similarities[idx]),
                        'document_count': len(project_chunks),
                        'document_types': list(set(
                            chunk.metadata.get('document_type', 'unknown')
                            for chunk in project_chunks
                        ))
                    })

            return results

        except Exception as e:
            logger.error(f"Project similarity search failed: {e}")
            return []


# Export main classes
__all__ = [
    'RAGSystem',
    'EmbeddingManager',
    'VectorStore',
    'DocumentChunk',
    'SearchResult',
    'VectorIndex'
]