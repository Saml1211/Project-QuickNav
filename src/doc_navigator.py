#!/usr/bin/env python3
"""
Advanced Document Navigator for Project QuickNav

Provides enterprise-grade document search and filtering capabilities including:
- Multi-criteria filtering (date, type, status, size, tags)
- Full-text content search
- Metadata extraction and indexing
- Bulk operations support
- Search query persistence
- Performance optimization for large datasets

Integration:
- Works with find_project_path.py for project resolution
- Provides backend for GUI advanced search features
- Supports CLI and MCP server usage
"""

import os
import re
import json
import sqlite3
import hashlib
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional dependencies for enhanced functionality
try:
    import docx2txt
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type enumeration."""
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
    CAD_DRAWINGS = "cad_drawings"
    SPECIFICATIONS = "specifications"
    MANUALS = "manuals"
    CERTIFICATES = "certificates"
    OTHER = "other"


class DocumentStatus(Enum):
    """Document status enumeration."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"


@dataclass
class DocumentInfo:
    """Document information structure."""
    path: str
    name: str
    type: DocumentType
    status: DocumentStatus
    size: int
    created_date: datetime
    modified_date: datetime
    accessed_date: datetime
    mime_type: str
    project_number: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    content_preview: str = ""
    checksum: str = ""
    folder_category: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchCriteria:
    """Search criteria structure for advanced filtering."""
    query: str = ""
    document_types: List[DocumentType] = None
    document_statuses: List[DocumentStatus] = None
    project_numbers: List[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    size_min: Optional[int] = None
    size_max: Optional[int] = None
    tags: List[str] = None
    folder_categories: List[str] = None
    mime_types: List[str] = None
    content_search: bool = False
    fuzzy_search: bool = True
    case_sensitive: bool = False
    
    def __post_init__(self):
        if self.document_types is None:
            self.document_types = []
        if self.document_statuses is None:
            self.document_statuses = []
        if self.project_numbers is None:
            self.project_numbers = []
        if self.tags is None:
            self.tags = []
        if self.folder_categories is None:
            self.folder_categories = []
        if self.mime_types is None:
            self.mime_types = []


@dataclass
class SearchResult:
    """Search result structure."""
    documents: List[DocumentInfo]
    total_count: int
    search_time: float
    query_used: SearchCriteria
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class DocumentNavigator:
    """Advanced document navigator with enterprise-grade search capabilities."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the document navigator.
        
        Args:
            cache_dir: Directory for caching document metadata and search indices
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".quicknav" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / "document_index.db"
        self.saved_searches_path = self.cache_dir / "saved_searches.json"
        
        # Document type patterns for classification
        self.type_patterns = {
            DocumentType.LLD: [r'\blld\b', r'low.?level.?design', r'detailed.?design'],
            DocumentType.HLD: [r'\bhld\b', r'high.?level.?design', r'system.?design'],
            DocumentType.CHANGE_ORDER: [r'change.?order', r'\bco\b', r'variation'],
            DocumentType.SALES_PO: [r'purchase.?order', r'\bpo\b', r'sales.?order'],
            DocumentType.FLOOR_PLANS: [r'floor.?plan', r'layout', r'\bfp\b'],
            DocumentType.SCOPE: [r'scope', r'sow', r'statement.?of.?work'],
            DocumentType.QA_ITP: [r'\bqa\b', r'\bitp\b', r'inspection', r'test.?plan'],
            DocumentType.SWMS: [r'\bswms\b', r'safe.?work', r'method.?statement'],
            DocumentType.SUPPLIER_QUOTES: [r'quote', r'quotation', r'supplier'],
            DocumentType.PHOTOS: [r'photo', r'image', r'picture'],
            DocumentType.CAD_DRAWINGS: [r'\bcad\b', r'drawing', r'dwg', r'dxf'],
            DocumentType.SPECIFICATIONS: [r'spec', r'specification', r'requirement'],
            DocumentType.MANUALS: [r'manual', r'guide', r'documentation'],
            DocumentType.CERTIFICATES: [r'certificate', r'cert', r'compliance']
        }
        
        # Status patterns for classification
        self.status_patterns = {
            DocumentStatus.DRAFT: [r'draft', r'wip', r'work.?in.?progress'],
            DocumentStatus.REVIEW: [r'review', r'pending', r'check'],
            DocumentStatus.APPROVED: [r'approved', r'final', r'signed'],
            DocumentStatus.SUPERSEDED: [r'superseded', r'obsolete', r'old'],
            DocumentStatus.ARCHIVED: [r'archived', r'backup', r'historical']
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._init_database()
        
        # Performance settings
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.batch_size = 100
        
    def _init_database(self):
        """Initialize the document metadata database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_date TEXT NOT NULL,
                    modified_date TEXT NOT NULL,
                    accessed_date TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    project_number TEXT,
                    tags TEXT,
                    metadata TEXT,
                    content_preview TEXT,
                    checksum TEXT,
                    folder_category TEXT,
                    indexed_date TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_number)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_modified ON documents(modified_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(name)
            """)
            
            # Full-text search table
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    path, name, content_preview, tags, metadata,
                    content='documents',
                    content_rowid='id'
                )
            """)
            
            conn.commit()
    
    def classify_document_type(self, file_path: str, content_preview: str = "") -> DocumentType:
        """Classify document type based on filename and content.
        
        Args:
            file_path: Path to the document
            content_preview: Preview of document content
            
        Returns:
            Classified document type
        """
        filename = os.path.basename(file_path).lower()
        content_lower = content_preview.lower()
        combined_text = f"{filename} {content_lower}"
        
        # Score each type based on pattern matches
        type_scores = {}
        for doc_type, patterns in self.type_patterns.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, combined_text, re.IGNORECASE))
            type_scores[doc_type] = score
        
        # Return type with highest score, or OTHER if no matches
        best_type = max(type_scores, key=type_scores.get)
        return best_type if type_scores[best_type] > 0 else DocumentType.OTHER
    
    def classify_document_status(self, file_path: str, content_preview: str = "") -> DocumentStatus:
        """Classify document status based on filename and content.
        
        Args:
            file_path: Path to the document
            content_preview: Preview of document content
            
        Returns:
            Classified document status
        """
        filename = os.path.basename(file_path).lower()
        content_lower = content_preview.lower()
        combined_text = f"{filename} {content_lower}"
        
        # Score each status based on pattern matches
        status_scores = {}
        for status, patterns in self.status_patterns.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, combined_text, re.IGNORECASE))
            status_scores[status] = score
        
        # Return status with highest score, or UNKNOWN if no matches
        best_status = max(status_scores, key=status_scores.get)
        return best_status if status_scores[best_status] > 0 else DocumentStatus.UNKNOWN
    
    def extract_content_preview(self, file_path: str, max_length: int = 1000) -> str:
        """Extract content preview from document.
        
        Args:
            file_path: Path to the document
            max_length: Maximum length of preview text
            
        Returns:
            Content preview text
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(max_length)
                    
            elif file_ext == '.docx' and HAS_DOCX:
                text = docx2txt.process(file_path)
                return text[:max_length] if text else ""
                
            elif file_ext == '.pdf' and HAS_PDF:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages[:3]:  # First 3 pages only
                        text += page.extract_text()
                        if len(text) >= max_length:
                            break
                    return text[:max_length]
                    
            return ""  # No content extraction available
            
        except Exception as e:
            logger.debug(f"Content extraction failed for {file_path}: {e}")
            return ""
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document metadata dictionary
        """
        metadata = {}
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Image metadata
            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff'] and HAS_PIL:
                with Image.open(file_path) as img:
                    metadata['dimensions'] = img.size
                    metadata['mode'] = img.mode
                    
                    # EXIF data
                    exif = img.getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            if isinstance(tag, str) and tag in ['DateTime', 'Make', 'Model']:
                                metadata[f'exif_{tag}'] = str(value)
            
            # PDF metadata
            elif file_ext == '.pdf' and HAS_PDF:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    if reader.metadata:
                        for key, value in reader.metadata.items():
                            if key.startswith('/'):
                                metadata[key[1:]] = str(value)
                    metadata['page_count'] = len(reader.pages)
            
            # Common file attributes
            stat = os.stat(file_path)
            metadata['permissions'] = oct(stat.st_mode)[-3:]
            
        except Exception as e:
            logger.debug(f"Metadata extraction failed for {file_path}: {e}")
        
        return metadata
    
    def calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum for change detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 checksum of file
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.debug(f"Checksum calculation failed for {file_path}: {e}")
            return ""
    
    def index_document(self, file_path: str, project_number: Optional[str] = None, 
                      folder_category: str = "") -> DocumentInfo:
        """Index a single document.
        
        Args:
            file_path: Path to the document
            project_number: Associated project number
            folder_category: Folder category (e.g., 'System Designs', 'Sales Handover')
            
        Returns:
            Indexed document information
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = os.stat(file_path)
            
            # Extract content preview and metadata
            content_preview = self.extract_content_preview(file_path)
            metadata = self.extract_metadata(file_path)
            
            # Classify document
            doc_type = self.classify_document_type(file_path, content_preview)
            doc_status = self.classify_document_status(file_path, content_preview)
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # Generate tags
            tags = self._generate_tags(file_path, content_preview, doc_type)
            
            # Create document info
            doc_info = DocumentInfo(
                path=file_path,
                name=os.path.basename(file_path),
                type=doc_type,
                status=doc_status,
                size=stat.st_size,
                created_date=datetime.fromtimestamp(stat.st_ctime),
                modified_date=datetime.fromtimestamp(stat.st_mtime),
                accessed_date=datetime.fromtimestamp(stat.st_atime),
                mime_type=mime_type,
                project_number=project_number,
                tags=tags,
                metadata=metadata,
                content_preview=content_preview,
                checksum=self.calculate_checksum(file_path),
                folder_category=folder_category
            )
            
            # Store in database
            self._store_document(doc_info)
            
            return doc_info
            
        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}")
            raise
    
    def _generate_tags(self, file_path: str, content_preview: str, doc_type: DocumentType) -> List[str]:
        """Generate tags for document based on analysis."""
        tags = []
        
        # File extension tag
        ext = os.path.splitext(file_path)[1].lower()
        if ext:
            tags.append(f"ext{ext}")
        
        # Document type tag
        tags.append(doc_type.value)
        
        # Common keywords
        text = f"{os.path.basename(file_path)} {content_preview}".lower()
        keywords = {
            'urgent': ['urgent', 'asap', 'priority'],
            'revision': ['rev', 'revision', 'version', 'v1', 'v2'],
            'final': ['final', 'approved', 'signed'],
            'draft': ['draft', 'preliminary', 'wip'],
            'technical': ['technical', 'engineering', 'specification'],
            'commercial': ['commercial', 'cost', 'price', 'budget']
        }
        
        for tag, patterns in keywords.items():
            if any(pattern in text for pattern in patterns):
                tags.append(tag)
        
        return list(set(tags))  # Remove duplicates
    
    def _store_document(self, doc_info: DocumentInfo):
        """Store document information in database."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents (
                        path, name, type, status, size, created_date, modified_date,
                        accessed_date, mime_type, project_number, tags, metadata,
                        content_preview, checksum, folder_category, indexed_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_info.path,
                    doc_info.name,
                    doc_info.type.value,
                    doc_info.status.value,
                    doc_info.size,
                    doc_info.created_date.isoformat(),
                    doc_info.modified_date.isoformat(),
                    doc_info.accessed_date.isoformat(),
                    doc_info.mime_type,
                    doc_info.project_number,
                    json.dumps(doc_info.tags),
                    json.dumps(doc_info.metadata),
                    doc_info.content_preview,
                    doc_info.checksum,
                    doc_info.folder_category,
                    datetime.now().isoformat()
                ))
                
                # Update FTS index
                conn.execute("""
                    INSERT OR REPLACE INTO documents_fts (
                        rowid, path, name, content_preview, tags, metadata
                    ) SELECT id, path, name, content_preview, tags, metadata 
                    FROM documents WHERE path = ?
                """, (doc_info.path,))
                
                conn.commit()
    
    def bulk_index_directory(self, directory: str, project_number: Optional[str] = None,
                           recursive: bool = True, progress_callback: Optional[Callable] = None) -> List[DocumentInfo]:
        """Index all documents in a directory.
        
        Args:
            directory: Directory to index
            project_number: Associated project number
            recursive: Whether to index subdirectories
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of indexed documents
        """
        indexed_docs = []
        
        # Collect all files
        files_to_index = []
        for root, dirs, files in os.walk(directory):
            if not recursive and root != directory:
                continue
                
            # Determine folder category
            folder_category = os.path.relpath(root, directory)
            if folder_category == '.':
                folder_category = ''
            
            for file in files:
                file_path = os.path.join(root, file)
                # Skip hidden files and common non-document files
                if not file.startswith('.') and not file.lower().endswith((
                    '.tmp', '.log', '.bak', '.cache', '.lock'
                )):
                    files_to_index.append((file_path, folder_category))
        
        # Index files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for file_path, folder_category in files_to_index:
                future = executor.submit(
                    self._safe_index_document, 
                    file_path, 
                    project_number, 
                    folder_category
                )
                futures.append(future)
            
            # Process completed futures
            for i, future in enumerate(as_completed(futures)):
                try:
                    doc_info = future.result()
                    if doc_info:
                        indexed_docs.append(doc_info)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(futures))
                        
                except Exception as e:
                    logger.error(f"Error indexing document: {e}")
        
        logger.info(f"Indexed {len(indexed_docs)} documents from {directory}")
        return indexed_docs
    
    def _safe_index_document(self, file_path: str, project_number: Optional[str], 
                           folder_category: str) -> Optional[DocumentInfo]:
        """Safely index a document with error handling."""
        try:
            return self.index_document(file_path, project_number, folder_category)
        except Exception as e:
            logger.warning(f"Failed to index {file_path}: {e}")
            return None
    
    def search_documents(self, criteria: SearchCriteria, 
                        limit: Optional[int] = None) -> SearchResult:
        """Search documents using advanced criteria.
        
        Args:
            criteria: Search criteria
            limit: Maximum number of results to return
            
        Returns:
            Search results with matching documents
        """
        start_time = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build query
            query_parts = []
            params = []
            
            # Text search
            if criteria.query.strip():
                if criteria.content_search:
                    # Use FTS for content search
                    query_parts.append("""
                        documents.id IN (
                            SELECT rowid FROM documents_fts 
                            WHERE documents_fts MATCH ?
                        )
                    """)
                    params.append(criteria.query)
                else:
                    # Simple name search
                    if criteria.fuzzy_search:
                        query_parts.append("documents.name LIKE ?")
                        params.append(f"%{criteria.query}%")
                    else:
                        if criteria.case_sensitive:
                            query_parts.append("documents.name GLOB ?")
                            params.append(f"*{criteria.query}*")
                        else:
                            query_parts.append("LOWER(documents.name) LIKE LOWER(?)")
                            params.append(f"%{criteria.query}%")
            
            # Document types
            if criteria.document_types:
                placeholders = ','.join(['?' for _ in criteria.document_types])
                query_parts.append(f"documents.type IN ({placeholders})")
                params.extend([dt.value for dt in criteria.document_types])
            
            # Document statuses
            if criteria.document_statuses:
                placeholders = ','.join(['?' for _ in criteria.document_statuses])
                query_parts.append(f"documents.status IN ({placeholders})")
                params.extend([ds.value for ds in criteria.document_statuses])
            
            # Project numbers
            if criteria.project_numbers:
                placeholders = ','.join(['?' for _ in criteria.project_numbers])
                query_parts.append(f"documents.project_number IN ({placeholders})")
                params.extend(criteria.project_numbers)
            
            # Date range
            if criteria.date_from:
                query_parts.append("documents.modified_date >= ?")
                params.append(criteria.date_from.isoformat())
            
            if criteria.date_to:
                query_parts.append("documents.modified_date <= ?")
                params.append(criteria.date_to.isoformat())
            
            # Size range
            if criteria.size_min is not None:
                query_parts.append("documents.size >= ?")
                params.append(criteria.size_min)
            
            if criteria.size_max is not None:
                query_parts.append("documents.size <= ?")
                params.append(criteria.size_max)
            
            # Tags
            if criteria.tags:
                for tag in criteria.tags:
                    query_parts.append("documents.tags LIKE ?")
                    params.append(f"%{tag}%")
            
            # Folder categories
            if criteria.folder_categories:
                placeholders = ','.join(['?' for _ in criteria.folder_categories])
                query_parts.append(f"documents.folder_category IN ({placeholders})")
                params.extend(criteria.folder_categories)
            
            # MIME types
            if criteria.mime_types:
                placeholders = ','.join(['?' for _ in criteria.mime_types])
                query_parts.append(f"documents.mime_type IN ({placeholders})")
                params.extend(criteria.mime_types)
            
            # Build final query
            base_query = "SELECT * FROM documents"
            if query_parts:
                base_query += " WHERE " + " AND ".join(query_parts)
            
            # Add ordering
            base_query += " ORDER BY documents.modified_date DESC"
            
            # Add limit
            if limit:
                base_query += " LIMIT ?"
                params.append(limit)
            
            # Execute query
            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()
            
            # Convert to DocumentInfo objects
            documents = []
            for row in rows:
                doc_info = DocumentInfo(
                    path=row['path'],
                    name=row['name'],
                    type=DocumentType(row['type']),
                    status=DocumentStatus(row['status']),
                    size=row['size'],
                    created_date=datetime.fromisoformat(row['created_date']),
                    modified_date=datetime.fromisoformat(row['modified_date']),
                    accessed_date=datetime.fromisoformat(row['accessed_date']),
                    mime_type=row['mime_type'],
                    project_number=row['project_number'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    content_preview=row['content_preview'] or "",
                    checksum=row['checksum'] or "",
                    folder_category=row['folder_category'] or ""
                )
                documents.append(doc_info)
            
            # Get total count without limit
            if limit and len(documents) == limit:
                count_query = "SELECT COUNT(*) FROM documents"
                if query_parts:
                    count_query += " WHERE " + " AND ".join(query_parts)
                
                cursor = conn.execute(count_query, params[:-1] if limit else params)
                total_count = cursor.fetchone()[0]
            else:
                total_count = len(documents)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Generate suggestions
        suggestions = self._generate_search_suggestions(criteria, documents)
        
        return SearchResult(
            documents=documents,
            total_count=total_count,
            search_time=search_time,
            query_used=criteria,
            suggestions=suggestions
        )
    
    def _generate_search_suggestions(self, criteria: SearchCriteria, 
                                   results: List[DocumentInfo]) -> List[str]:
        """Generate search suggestions based on current results."""
        suggestions = []
        
        if not results and criteria.query:
            # Suggest spelling corrections or alternatives
            suggestions.append(f"Try searching for '{criteria.query}' with fuzzy matching enabled")
            suggestions.append("Check if the document type filter is too restrictive")
            suggestions.append("Expand the date range filter")
        
        if len(results) > 100:
            suggestions.append("Consider adding more specific filters to narrow results")
            suggestions.append("Try searching within a specific project or folder")
        
        # Suggest common document types found
        if results:
            common_types = {}
            for doc in results[:50]:  # Sample first 50
                common_types[doc.type] = common_types.get(doc.type, 0) + 1
            
            if len(common_types) > 1:
                most_common = max(common_types, key=common_types.get)
                suggestions.append(f"Filter by '{most_common.value}' documents to focus results")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def get_document_by_path(self, path: str) -> Optional[DocumentInfo]:
        """Get document information by path.
        
        Args:
            path: Document path
            
        Returns:
            Document information if found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM documents WHERE path = ?", (path,))
            row = cursor.fetchone()
            
            if row:
                return DocumentInfo(
                    path=row['path'],
                    name=row['name'],
                    type=DocumentType(row['type']),
                    status=DocumentStatus(row['status']),
                    size=row['size'],
                    created_date=datetime.fromisoformat(row['created_date']),
                    modified_date=datetime.fromisoformat(row['modified_date']),
                    accessed_date=datetime.fromisoformat(row['accessed_date']),
                    mime_type=row['mime_type'],
                    project_number=row['project_number'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    content_preview=row['content_preview'] or "",
                    checksum=row['checksum'] or "",
                    folder_category=row['folder_category'] or ""
                )
        
        return None
    
    def save_search_query(self, name: str, criteria: SearchCriteria) -> bool:
        """Save a search query for later use.
        
        Args:
            name: Name for the saved search
            criteria: Search criteria to save
            
        Returns:
            True if saved successfully
        """
        try:
            saved_searches = self.load_saved_searches()
            saved_searches[name] = asdict(criteria)
            
            with open(self.saved_searches_path, 'w') as f:
                json.dump(saved_searches, f, indent=2, default=str)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save search query: {e}")
            return False
    
    def load_saved_searches(self) -> Dict[str, Dict[str, Any]]:
        """Load saved search queries.
        
        Returns:
            Dictionary of saved searches
        """
        try:
            if self.saved_searches_path.exists():
                with open(self.saved_searches_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load saved searches: {e}")
        
        return {}
    
    def delete_saved_search(self, name: str) -> bool:
        """Delete a saved search query.
        
        Args:
            name: Name of the search to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            saved_searches = self.load_saved_searches()
            if name in saved_searches:
                del saved_searches[name]
                
                with open(self.saved_searches_path, 'w') as f:
                    json.dump(saved_searches, f, indent=2, default=str)
                
                return True
        except Exception as e:
            logger.error(f"Failed to delete saved search: {e}")
        
        return False
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search and indexing statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_documents': 0,
            'total_size': 0,
            'document_types': {},
            'document_statuses': {},
            'project_count': 0,
            'last_indexed': None,
            'index_size_mb': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total documents and size
                cursor = conn.execute("SELECT COUNT(*), SUM(size) FROM documents")
                row = cursor.fetchone()
                stats['total_documents'] = row[0] or 0
                stats['total_size'] = row[1] or 0
                
                # Document types
                cursor = conn.execute("""
                    SELECT type, COUNT(*) FROM documents GROUP BY type
                """)
                for row in cursor.fetchall():
                    stats['document_types'][row[0]] = row[1]
                
                # Document statuses
                cursor = conn.execute("""
                    SELECT status, COUNT(*) FROM documents GROUP BY status
                """)
                for row in cursor.fetchall():
                    stats['document_statuses'][row[0]] = row[1]
                
                # Project count
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT project_number) FROM documents 
                    WHERE project_number IS NOT NULL
                """)
                stats['project_count'] = cursor.fetchone()[0] or 0
                
                # Last indexed
                cursor = conn.execute("""
                    SELECT MAX(indexed_date) FROM documents
                """)
                last_indexed = cursor.fetchone()[0]
                if last_indexed:
                    stats['last_indexed'] = datetime.fromisoformat(last_indexed)
            
            # Database size
            if self.db_path.exists():
                stats['index_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
        
        return stats
    
    def cleanup_index(self, remove_missing: bool = True) -> int:
        """Clean up the document index.
        
        Args:
            remove_missing: Whether to remove entries for missing files
            
        Returns:
            Number of entries removed
        """
        removed_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if remove_missing:
                    # Get all document paths
                    cursor = conn.execute("SELECT id, path FROM documents")
                    documents = cursor.fetchall()
                    
                    # Check which files no longer exist
                    to_remove = []
                    for doc_id, path in documents:
                        if not os.path.exists(path):
                            to_remove.append(doc_id)
                    
                    # Remove missing files
                    if to_remove:
                        placeholders = ','.join(['?' for _ in to_remove])
                        conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", to_remove)
                        conn.execute(f"DELETE FROM documents_fts WHERE rowid IN ({placeholders})", to_remove)
                        removed_count = len(to_remove)
                
                # Vacuum database
                conn.execute("VACUUM")
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to cleanup index: {e}")
        
        return removed_count


# Bulk operation functions
class BulkOperations:
    """Bulk operations for document management."""
    
    @staticmethod
    def copy_documents(documents: List[DocumentInfo], destination: str, 
                      progress_callback: Optional[Callable] = None) -> List[str]:
        """Copy multiple documents to a destination.
        
        Args:
            documents: List of documents to copy
            destination: Destination directory
            progress_callback: Optional progress callback
            
        Returns:
            List of successfully copied file paths
        """
        import shutil
        
        os.makedirs(destination, exist_ok=True)
        copied_files = []
        
        for i, doc in enumerate(documents):
            try:
                dest_path = os.path.join(destination, doc.name)
                # Handle name conflicts
                counter = 1
                base_name, ext = os.path.splitext(doc.name)
                while os.path.exists(dest_path):
                    new_name = f"{base_name}_{counter}{ext}"
                    dest_path = os.path.join(destination, new_name)
                    counter += 1
                
                shutil.copy2(doc.path, dest_path)
                copied_files.append(dest_path)
                
                if progress_callback:
                    progress_callback(i + 1, len(documents))
                    
            except Exception as e:
                logger.error(f"Failed to copy {doc.path}: {e}")
        
        return copied_files
    
    @staticmethod
    def export_document_list(documents: List[DocumentInfo], format: str = 'csv') -> str:
        """Export document list to various formats.
        
        Args:
            documents: List of documents to export
            format: Export format ('csv', 'json', 'xlsx')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            import csv
            
            filename = f"document_export_{timestamp}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Name', 'Path', 'Type', 'Status', 'Size', 'Modified', 
                    'Project', 'Folder', 'Tags'
                ])
                
                for doc in documents:
                    writer.writerow([
                        doc.name,
                        doc.path,
                        doc.type.value,
                        doc.status.value,
                        doc.size,
                        doc.modified_date.isoformat(),
                        doc.project_number or '',
                        doc.folder_category,
                        ', '.join(doc.tags)
                    ])
            
            return filename
        
        elif format == 'json':
            filename = f"document_export_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                export_data = []
                for doc in documents:
                    export_data.append({
                        'name': doc.name,
                        'path': doc.path,
                        'type': doc.type.value,
                        'status': doc.status.value,
                        'size': doc.size,
                        'modified_date': doc.modified_date.isoformat(),
                        'project_number': doc.project_number,
                        'folder_category': doc.folder_category,
                        'tags': doc.tags,
                        'metadata': doc.metadata
                    })
                
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return filename
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Document Navigator")
    parser.add_argument('action', choices=['index', 'search', 'stats', 'cleanup'])
    parser.add_argument('--directory', help='Directory to index')
    parser.add_argument('--project', help='Project number')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--type', help='Document type filter')
    parser.add_argument('--limit', type=int, help='Limit results')
    
    args = parser.parse_args()
    
    navigator = DocumentNavigator()
    
    if args.action == 'index':
        if not args.directory:
            print("Error: --directory required for indexing")
            sys.exit(1)
        
        print(f"Indexing directory: {args.directory}")
        docs = navigator.bulk_index_directory(args.directory, args.project)
        print(f"Indexed {len(docs)} documents")
    
    elif args.action == 'search':
        criteria = SearchCriteria()
        if args.query:
            criteria.query = args.query
        if args.type:
            criteria.document_types = [DocumentType(args.type)]
        
        results = navigator.search_documents(criteria, args.limit)
        print(f"Found {results.total_count} documents in {results.search_time:.2f}s")
        
        for doc in results.documents[:10]:  # Show first 10
            print(f"- {doc.name} ({doc.type.value}) - {doc.path}")
    
    elif args.action == 'stats':
        stats = navigator.get_search_statistics()
        print("Document Index Statistics:")
        print(f"Total Documents: {stats['total_documents']:,}")
        print(f"Total Size: {stats['total_size']:,} bytes")
        print(f"Projects: {stats['project_count']}")
        print(f"Index Size: {stats['index_size_mb']:.1f} MB")
        
        if stats['document_types']:
            print("\nDocument Types:")
            for doc_type, count in stats['document_types'].items():
                print(f"  {doc_type}: {count}")
    
    elif args.action == 'cleanup':
        removed = navigator.cleanup_index()
        print(f"Removed {removed} missing document entries")
