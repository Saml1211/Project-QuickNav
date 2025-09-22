"""
Feature Store for Project QuickNav

Implements ML-ready feature engineering, storage, and serving for enhanced
search ranking, document recommendations, and user behavior analysis.
"""

import json
import time
import asyncio
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import logging
import threading
from pathlib import Path
import hashlib

from .config import get_config
from .etl import DuckDBLoader
from .streaming import Event

logger = logging.getLogger(__name__)


@dataclass
class Feature:
    """Individual feature definition"""
    name: str
    value: Union[float, int, str, bool, List, Dict]
    feature_type: str  # 'numeric', 'categorical', 'text', 'boolean', 'vector'
    timestamp: datetime
    ttl_hours: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def is_expired(self) -> bool:
        """Check if feature has expired"""
        if self.ttl_hours is None:
            return False
        expiry_time = self.timestamp + timedelta(hours=self.ttl_hours)
        return datetime.utcnow() > expiry_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feature':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class FeatureEngineer:
    """Generates features from raw data and user behavior"""

    def __init__(self, db_loader: DuckDBLoader = None):
        self.config = get_config()
        self.db = db_loader or DuckDBLoader()

    def compute_project_features(self, project_id: str) -> Dict[str, Feature]:
        """Compute comprehensive project-level features"""
        features = {}
        timestamp = datetime.utcnow()

        try:
            # Basic project statistics
            basic_stats = self._get_project_basic_stats(project_id)
            features.update(self._create_basic_features(basic_stats, timestamp))

            # Document diversity features
            doc_features = self._compute_document_diversity_features(project_id, timestamp)
            features.update(doc_features)

            # Temporal features
            temporal_features = self._compute_temporal_features(project_id, timestamp)
            features.update(temporal_features)

            # User behavior features
            behavior_features = self._compute_user_behavior_features(project_id, timestamp)
            features.update(behavior_features)

            # File system structure features
            structure_features = self._compute_structure_features(project_id, timestamp)
            features.update(structure_features)

        except Exception as e:
            logger.error(f"Error computing project features for {project_id}: {e}")

        return features

    def compute_document_features(self, document_id: str) -> Dict[str, Feature]:
        """Compute document-level features"""
        features = {}
        timestamp = datetime.utcnow()

        try:
            # Get document metadata
            doc_data = self._get_document_data(document_id)
            if not doc_data:
                return features

            # Version features
            version_features = self._compute_version_features(doc_data, timestamp)
            features.update(version_features)

            # Content features
            content_features = self._compute_content_features(doc_data, timestamp)
            features.update(content_features)

            # Access pattern features
            access_features = self._compute_access_pattern_features(document_id, timestamp)
            features.update(access_features)

            # Similarity features
            similarity_features = self._compute_similarity_features(doc_data, timestamp)
            features.update(similarity_features)

        except Exception as e:
            logger.error(f"Error computing document features for {document_id}: {e}")

        return features

    def compute_user_features(self, user_id: str, session_id: str = None) -> Dict[str, Feature]:
        """Compute user behavior features"""
        features = {}
        timestamp = datetime.utcnow()

        try:
            # User activity patterns
            activity_features = self._compute_user_activity_features(user_id, timestamp)
            features.update(activity_features)

            # Preference features
            preference_features = self._compute_user_preferences(user_id, timestamp)
            features.update(preference_features)

            # Session-specific features
            if session_id:
                session_features = self._compute_session_features(session_id, timestamp)
                features.update(session_features)

        except Exception as e:
            logger.error(f"Error computing user features for {user_id}: {e}")

        return features

    def _get_project_basic_stats(self, project_id: str) -> Dict[str, Any]:
        """Get basic project statistics"""
        result = self.db.db.execute("""
            SELECT
                p.project_name,
                p.total_files,
                p.total_size_bytes,
                p.document_count,
                p.has_standard_structure,
                p.folder_depth,
                p.created_date,
                p.modified_date,
                p.last_activity,
                COUNT(d.document_id) as indexed_documents,
                COUNT(DISTINCT d.doc_type) as doc_types_count,
                AVG(d.file_size) as avg_file_size,
                MAX(d.modified_date) as latest_doc_date
            FROM quicknav.projects p
            LEFT JOIN quicknav.documents d ON p.project_id = d.project_id
            WHERE p.project_id = ?
            GROUP BY p.project_id
        """, [project_id]).fetchone()

        if not result:
            return {}

        return {
            'project_name': result[0],
            'total_files': result[1] or 0,
            'total_size_bytes': result[2] or 0,
            'document_count': result[3] or 0,
            'has_standard_structure': result[4] or False,
            'folder_depth': result[5] or 0,
            'created_date': result[6],
            'modified_date': result[7],
            'last_activity': result[8],
            'indexed_documents': result[9] or 0,
            'doc_types_count': result[10] or 0,
            'avg_file_size': result[11] or 0,
            'latest_doc_date': result[12]
        }

    def _create_basic_features(self, stats: Dict[str, Any], timestamp: datetime) -> Dict[str, Feature]:
        """Create basic numerical features from project stats"""
        features = {}

        # Size and count features
        features['total_files'] = Feature(
            name='total_files',
            value=float(stats.get('total_files', 0)),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        features['total_size_mb'] = Feature(
            name='total_size_mb',
            value=float(stats.get('total_size_bytes', 0)) / (1024 * 1024),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        features['document_count'] = Feature(
            name='document_count',
            value=float(stats.get('document_count', 0)),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        features['doc_types_diversity'] = Feature(
            name='doc_types_diversity',
            value=float(stats.get('doc_types_count', 0)),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        # Structure features
        features['has_standard_structure'] = Feature(
            name='has_standard_structure',
            value=bool(stats.get('has_standard_structure', False)),
            feature_type='boolean',
            timestamp=timestamp,
            ttl_hours=24
        )

        features['folder_depth'] = Feature(
            name='folder_depth',
            value=float(stats.get('folder_depth', 0)),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        # Derived features
        if stats.get('document_count', 0) > 0:
            features['avg_file_size_mb'] = Feature(
                name='avg_file_size_mb',
                value=float(stats.get('avg_file_size', 0)) / (1024 * 1024),
                feature_type='numeric',
                timestamp=timestamp,
                ttl_hours=24
            )

            features['files_per_document_ratio'] = Feature(
                name='files_per_document_ratio',
                value=float(stats.get('total_files', 0)) / float(stats.get('document_count', 1)),
                feature_type='numeric',
                timestamp=timestamp,
                ttl_hours=24
            )

        return features

    def _compute_document_diversity_features(self, project_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute document type diversity features"""
        features = {}

        try:
            # Get document type distribution
            doc_types = self.db.db.execute("""
                SELECT doc_type, COUNT(*) as count, AVG(file_size) as avg_size
                FROM quicknav.documents
                WHERE project_id = ? AND doc_type != 'unknown'
                GROUP BY doc_type
            """, [project_id]).fetchall()

            if doc_types:
                type_counts = {row[0]: row[1] for row in doc_types}
                total_docs = sum(type_counts.values())

                # Shannon diversity index
                shannon_entropy = -sum(
                    (count / total_docs) * np.log2(count / total_docs)
                    for count in type_counts.values()
                    if count > 0
                )

                features['doc_type_entropy'] = Feature(
                    name='doc_type_entropy',
                    value=float(shannon_entropy),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=12
                )

                # Most common document type ratio
                max_count = max(type_counts.values())
                features['dominant_doc_type_ratio'] = Feature(
                    name='dominant_doc_type_ratio',
                    value=float(max_count) / float(total_docs),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=12
                )

                # Document type vector (for similarity calculations)
                type_vector = [type_counts.get(doc_type, 0) for doc_type in [
                    'lld', 'hld', 'floor_plans', 'change_order', 'sales_po', 'scope', 'photos'
                ]]
                features['doc_type_vector'] = Feature(
                    name='doc_type_vector',
                    value=type_vector,
                    feature_type='vector',
                    timestamp=timestamp,
                    ttl_hours=12
                )

        except Exception as e:
            logger.error(f"Error computing document diversity features: {e}")

        return features

    def _compute_temporal_features(self, project_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute temporal features based on document dates"""
        features = {}

        try:
            # Get temporal statistics
            temporal_stats = self.db.db.execute("""
                SELECT
                    MIN(created_date) as earliest_date,
                    MAX(modified_date) as latest_date,
                    AVG(EXTRACT(EPOCH FROM (modified_date - created_date))) as avg_edit_duration,
                    COUNT(CASE WHEN modified_date > created_date THEN 1 END) as edited_docs_count,
                    COUNT(*) as total_docs
                FROM quicknav.documents
                WHERE project_id = ?
            """, [project_id]).fetchone()

            if temporal_stats and temporal_stats[0]:
                earliest_date = temporal_stats[0]
                latest_date = temporal_stats[1]
                avg_edit_duration = temporal_stats[2] or 0
                edited_docs_count = temporal_stats[3] or 0
                total_docs = temporal_stats[4] or 0

                # Project age in days
                if isinstance(earliest_date, str):
                    earliest_date = datetime.fromisoformat(earliest_date)
                project_age_days = (timestamp - earliest_date).days

                features['project_age_days'] = Feature(
                    name='project_age_days',
                    value=float(project_age_days),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=24
                )

                # Days since last activity
                if isinstance(latest_date, str):
                    latest_date = datetime.fromisoformat(latest_date)
                days_since_activity = (timestamp - latest_date).days

                features['days_since_last_activity'] = Feature(
                    name='days_since_last_activity',
                    value=float(days_since_activity),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=1
                )

                # Average document edit duration (hours)
                features['avg_edit_duration_hours'] = Feature(
                    name='avg_edit_duration_hours',
                    value=float(avg_edit_duration) / 3600.0,
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=12
                )

                # Document edit ratio
                if total_docs > 0:
                    features['document_edit_ratio'] = Feature(
                        name='document_edit_ratio',
                        value=float(edited_docs_count) / float(total_docs),
                        feature_type='numeric',
                        timestamp=timestamp,
                        ttl_hours=12
                    )

                # Activity recency score (exponential decay)
                recency_score = np.exp(-days_since_activity / 30.0)  # 30-day half-life
                features['activity_recency_score'] = Feature(
                    name='activity_recency_score',
                    value=float(recency_score),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=1
                )

        except Exception as e:
            logger.error(f"Error computing temporal features: {e}")

        return features

    def _compute_user_behavior_features(self, project_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute features based on user behavior patterns"""
        features = {}

        try:
            # Get user interaction statistics
            behavior_stats = self.db.db.execute("""
                SELECT
                    COUNT(DISTINCT session_id) as unique_sessions,
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(CASE WHEN event_type = 'project_search' THEN 1 END) as search_count,
                    COUNT(CASE WHEN event_type = 'document_open' THEN 1 END) as document_opens,
                    MIN(timestamp) as first_access,
                    MAX(timestamp) as last_access
                FROM quicknav.user_interactions
                WHERE project_id = ? AND timestamp > ?
            """, [project_id, timestamp - timedelta(days=30)]).fetchone()

            if behavior_stats:
                unique_sessions = behavior_stats[0] or 0
                total_interactions = behavior_stats[1] or 0
                unique_users = behavior_stats[2] or 0
                avg_response_time = behavior_stats[3] or 0
                search_count = behavior_stats[4] or 0
                document_opens = behavior_stats[5] or 0

                # Usage frequency
                features['monthly_sessions'] = Feature(
                    name='monthly_sessions',
                    value=float(unique_sessions),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=6
                )

                features['monthly_users'] = Feature(
                    name='monthly_users',
                    value=float(unique_users),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=6
                )

                # Interaction patterns
                if total_interactions > 0:
                    features['search_to_interaction_ratio'] = Feature(
                        name='search_to_interaction_ratio',
                        value=float(search_count) / float(total_interactions),
                        feature_type='numeric',
                        timestamp=timestamp,
                        ttl_hours=6
                    )

                    features['document_open_ratio'] = Feature(
                        name='document_open_ratio',
                        value=float(document_opens) / float(total_interactions),
                        feature_type='numeric',
                        timestamp=timestamp,
                        ttl_hours=6
                    )

                # Performance metrics
                features['avg_response_time_ms'] = Feature(
                    name='avg_response_time_ms',
                    value=float(avg_response_time),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=1
                )

                # Popularity score (based on unique users and sessions)
                popularity_score = np.log1p(unique_users) * np.log1p(unique_sessions)
                features['popularity_score'] = Feature(
                    name='popularity_score',
                    value=float(popularity_score),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=6
                )

        except Exception as e:
            logger.error(f"Error computing user behavior features: {e}")

        return features

    def _compute_structure_features(self, project_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute features based on file system structure"""
        features = {}

        try:
            # Get folder structure analysis
            folder_stats = self.db.db.execute("""
                SELECT
                    folder_name,
                    COUNT(*) as file_count,
                    AVG(file_size) as avg_file_size,
                    COUNT(DISTINCT doc_type) as doc_types_in_folder
                FROM quicknav.documents
                WHERE project_id = ?
                GROUP BY folder_name
            """, [project_id]).fetchall()

            if folder_stats:
                folder_data = {row[0]: {'count': row[1], 'avg_size': row[2], 'types': row[3]}
                              for row in folder_stats}

                # Standard folder compliance
                standard_folders = [
                    '1. Sales Handover',
                    '2. BOM & Orders',
                    '3. PMO',
                    '4. System Designs',
                    '5. Floor Plans',
                    '6. Site Photos'
                ]

                standard_folder_count = sum(1 for folder in standard_folders
                                          if any(std in folder_name for folder_name in folder_data.keys()
                                                for std in [folder]))

                features['standard_folder_compliance'] = Feature(
                    name='standard_folder_compliance',
                    value=float(standard_folder_count) / len(standard_folders),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=24
                )

                # Folder distribution entropy
                total_files = sum(data['count'] for data in folder_data.values())
                if total_files > 0:
                    folder_entropy = -sum(
                        (data['count'] / total_files) * np.log2(data['count'] / total_files)
                        for data in folder_data.values()
                        if data['count'] > 0
                    )

                    features['folder_distribution_entropy'] = Feature(
                        name='folder_distribution_entropy',
                        value=float(folder_entropy),
                        feature_type='numeric',
                        timestamp=timestamp,
                        ttl_hours=12
                    )

                # Organization score (lower is better organized)
                max_files_in_folder = max(data['count'] for data in folder_data.values())
                organization_score = 1.0 - (max_files_in_folder / total_files)

                features['organization_score'] = Feature(
                    name='organization_score',
                    value=float(organization_score),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=12
                )

        except Exception as e:
            logger.error(f"Error computing structure features: {e}")

        return features

    def _get_document_data(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document data from database"""
        result = self.db.db.execute("""
            SELECT *
            FROM quicknav.documents
            WHERE document_id = ?
        """, [document_id]).fetchone()

        if not result:
            return None

        # Convert to dictionary (assuming column order)
        columns = [desc[0] for desc in self.db.db.description]
        return dict(zip(columns, result))

    def _compute_version_features(self, doc_data: Dict[str, Any], timestamp: datetime) -> Dict[str, Feature]:
        """Compute version-related features"""
        features = {}

        version_numeric = doc_data.get('version_numeric')
        if version_numeric is not None:
            features['version_numeric'] = Feature(
                name='version_numeric',
                value=float(version_numeric),
                feature_type='numeric',
                timestamp=timestamp,
                ttl_hours=24
            )

            # Version maturity score (higher versions are more mature)
            maturity_score = min(1.0, version_numeric / 200.0)  # Normalize to 0-1
            features['version_maturity_score'] = Feature(
                name='version_maturity_score',
                value=float(maturity_score),
                feature_type='numeric',
                timestamp=timestamp,
                ttl_hours=24
            )

        # As-built status
        features['is_as_built'] = Feature(
            name='is_as_built',
            value=bool(doc_data.get('is_as_built', False)),
            feature_type='boolean',
            timestamp=timestamp,
            ttl_hours=24
        )

        # Document type priority
        features['doc_type_priority'] = Feature(
            name='doc_type_priority',
            value=float(doc_data.get('doc_type_priority', 0)),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        return features

    def _compute_content_features(self, doc_data: Dict[str, Any], timestamp: datetime) -> Dict[str, Feature]:
        """Compute content-based features"""
        features = {}

        # File size features
        file_size = doc_data.get('file_size', 0)
        features['file_size_mb'] = Feature(
            name='file_size_mb',
            value=float(file_size) / (1024 * 1024),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        # Size category
        if file_size < 1024 * 1024:  # < 1MB
            size_category = 'small'
        elif file_size < 10 * 1024 * 1024:  # < 10MB
            size_category = 'medium'
        else:
            size_category = 'large'

        features['size_category'] = Feature(
            name='size_category',
            value=size_category,
            feature_type='categorical',
            timestamp=timestamp,
            ttl_hours=24
        )

        # Filename complexity
        filename = doc_data.get('file_name', '')
        complexity_score = len(filename) / 100.0  # Normalize
        features['filename_complexity'] = Feature(
            name='filename_complexity',
            value=float(complexity_score),
            feature_type='numeric',
            timestamp=timestamp,
            ttl_hours=24
        )

        return features

    def _compute_access_pattern_features(self, document_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute access pattern features"""
        features = {}

        try:
            # Get access statistics
            access_stats = self.db.db.execute("""
                SELECT
                    COUNT(*) as access_count,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    COUNT(DISTINCT user_id) as unique_users,
                    MIN(timestamp) as first_access,
                    MAX(timestamp) as last_access,
                    AVG(response_time_ms) as avg_response_time
                FROM quicknav.user_interactions
                WHERE document_path LIKE '%' || ? || '%'
                AND timestamp > ?
            """, [document_id, timestamp - timedelta(days=30)]).fetchone()

            if access_stats:
                access_count = access_stats[0] or 0
                unique_sessions = access_stats[1] or 0
                unique_users = access_stats[2] or 0

                features['monthly_access_count'] = Feature(
                    name='monthly_access_count',
                    value=float(access_count),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=6
                )

                features['monthly_unique_users'] = Feature(
                    name='monthly_unique_users',
                    value=float(unique_users),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=6
                )

                # Access frequency score
                if access_count > 0:
                    frequency_score = np.log1p(access_count)
                    features['access_frequency_score'] = Feature(
                        name='access_frequency_score',
                        value=float(frequency_score),
                        feature_type='numeric',
                        timestamp=timestamp,
                        ttl_hours=6
                    )

        except Exception as e:
            logger.error(f"Error computing access pattern features: {e}")

        return features

    def _compute_similarity_features(self, doc_data: Dict[str, Any], timestamp: datetime) -> Dict[str, Feature]:
        """Compute document similarity features"""
        features = {}

        try:
            project_id = doc_data.get('project_id')
            doc_type = doc_data.get('doc_type')

            if project_id and doc_type:
                # Count similar documents in same project
                similar_count = self.db.db.execute("""
                    SELECT COUNT(*)
                    FROM quicknav.documents
                    WHERE project_id = ? AND doc_type = ? AND document_id != ?
                """, [project_id, doc_type, doc_data.get('document_id')]).fetchone()[0]

                features['similar_docs_in_project'] = Feature(
                    name='similar_docs_in_project',
                    value=float(similar_count),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=12
                )

                # Uniqueness score (inverse of similarity)
                uniqueness_score = 1.0 / (1.0 + similar_count)
                features['uniqueness_score'] = Feature(
                    name='uniqueness_score',
                    value=float(uniqueness_score),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=12
                )

        except Exception as e:
            logger.error(f"Error computing similarity features: {e}")

        return features

    def _compute_user_activity_features(self, user_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute user activity features"""
        features = {}

        try:
            # Get user activity statistics
            activity_stats = self.db.db.execute("""
                SELECT
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(DISTINCT project_id) as projects_accessed,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(CASE WHEN event_type = 'project_search' THEN 1 END) as search_count,
                    MIN(timestamp) as first_activity,
                    MAX(timestamp) as last_activity
                FROM quicknav.user_interactions
                WHERE user_id = ? AND timestamp > ?
            """, [user_id, timestamp - timedelta(days=30)]).fetchone()

            if activity_stats:
                total_interactions = activity_stats[0] or 0
                total_sessions = activity_stats[1] or 0
                projects_accessed = activity_stats[2] or 0

                features['user_activity_level'] = Feature(
                    name='user_activity_level',
                    value=float(total_interactions),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=6
                )

                features['user_project_diversity'] = Feature(
                    name='user_project_diversity',
                    value=float(projects_accessed),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=6
                )

                # User expertise score (based on activity and diversity)
                if total_sessions > 0:
                    expertise_score = np.log1p(total_interactions) * np.log1p(projects_accessed)
                    features['user_expertise_score'] = Feature(
                        name='user_expertise_score',
                        value=float(expertise_score),
                        feature_type='numeric',
                        timestamp=timestamp,
                        ttl_hours=6
                    )

        except Exception as e:
            logger.error(f"Error computing user activity features: {e}")

        return features

    def _compute_user_preferences(self, user_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute user preference features"""
        features = {}

        try:
            # Get user's preferred document types
            doc_preferences = self.db.db.execute("""
                SELECT
                    JSON_EXTRACT(event_data, '$.doc_type') as doc_type,
                    COUNT(*) as access_count
                FROM quicknav.user_interactions
                WHERE user_id = ?
                AND event_type = 'document_open'
                AND timestamp > ?
                GROUP BY JSON_EXTRACT(event_data, '$.doc_type')
                ORDER BY access_count DESC
            """, [user_id, timestamp - timedelta(days=90)]).fetchall()

            if doc_preferences:
                # Most preferred document type
                preferred_doc_type = doc_preferences[0][0]
                features['preferred_doc_type'] = Feature(
                    name='preferred_doc_type',
                    value=preferred_doc_type,
                    feature_type='categorical',
                    timestamp=timestamp,
                    ttl_hours=24
                )

                # Document type preference vector
                total_accesses = sum(row[1] for row in doc_preferences)
                preference_vector = {}
                for doc_type, count in doc_preferences:
                    if doc_type:
                        preference_vector[doc_type] = count / total_accesses

                features['doc_type_preferences'] = Feature(
                    name='doc_type_preferences',
                    value=preference_vector,
                    feature_type='vector',
                    timestamp=timestamp,
                    ttl_hours=24
                )

        except Exception as e:
            logger.error(f"Error computing user preferences: {e}")

        return features

    def _compute_session_features(self, session_id: str, timestamp: datetime) -> Dict[str, Feature]:
        """Compute session-specific features"""
        features = {}

        try:
            # Get session statistics
            session_stats = self.db.db.execute("""
                SELECT
                    COUNT(*) as interaction_count,
                    COUNT(DISTINCT project_id) as projects_in_session,
                    AVG(response_time_ms) as avg_response_time,
                    MIN(timestamp) as session_start,
                    MAX(timestamp) as session_end
                FROM quicknav.user_interactions
                WHERE session_id = ?
            """, [session_id]).fetchone()

            if session_stats:
                interaction_count = session_stats[0] or 0
                projects_in_session = session_stats[1] or 0
                session_start = session_stats[3]
                session_end = session_stats[4]

                features['session_interaction_count'] = Feature(
                    name='session_interaction_count',
                    value=float(interaction_count),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=1
                )

                features['session_project_diversity'] = Feature(
                    name='session_project_diversity',
                    value=float(projects_in_session),
                    feature_type='numeric',
                    timestamp=timestamp,
                    ttl_hours=1
                )

                # Session intensity (interactions per minute)
                if session_start and session_end:
                    if isinstance(session_start, str):
                        session_start = datetime.fromisoformat(session_start)
                        session_end = datetime.fromisoformat(session_end)

                    duration_minutes = max(1, (session_end - session_start).total_seconds() / 60)
                    intensity = interaction_count / duration_minutes

                    features['session_intensity'] = Feature(
                        name='session_intensity',
                        value=float(intensity),
                        feature_type='numeric',
                        timestamp=timestamp,
                        ttl_hours=1
                    )

        except Exception as e:
            logger.error(f"Error computing session features: {e}")

        return features


class FeatureStore:
    """Centralized feature storage and retrieval system"""

    def __init__(self, db_loader: DuckDBLoader = None):
        self.config = get_config()
        self.db = db_loader or DuckDBLoader()
        self.cache = {}
        self.cache_lock = threading.RLock()
        self._setup_feature_tables()

    def _setup_feature_tables(self):
        """Setup feature storage tables"""
        schema_sql = """
        -- Feature store table
        CREATE TABLE IF NOT EXISTS quicknav.features (
            entity_id VARCHAR NOT NULL,
            entity_type VARCHAR NOT NULL, -- 'project', 'document', 'user', 'session'
            feature_name VARCHAR NOT NULL,
            feature_value TEXT NOT NULL, -- JSON serialized value
            feature_type VARCHAR NOT NULL,
            computed_at TIMESTAMP NOT NULL,
            ttl_hours INTEGER,
            expires_at TIMESTAMP,
            metadata TEXT,
            PRIMARY KEY (entity_id, entity_type, feature_name)
        );

        -- Feature computation log
        CREATE TABLE IF NOT EXISTS quicknav.feature_computation_log (
            log_id VARCHAR PRIMARY KEY,
            entity_id VARCHAR NOT NULL,
            entity_type VARCHAR NOT NULL,
            feature_count INTEGER,
            computation_time_ms DOUBLE,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indices
        CREATE INDEX IF NOT EXISTS idx_features_entity ON quicknav.features(entity_id, entity_type);
        CREATE INDEX IF NOT EXISTS idx_features_computed ON quicknav.features(computed_at);
        CREATE INDEX IF NOT EXISTS idx_features_expires ON quicknav.features(expires_at);
        CREATE INDEX IF NOT EXISTS idx_feature_log_entity ON quicknav.feature_computation_log(entity_id, entity_type);
        """

        self.db.db.executescript(schema_sql)

    def store_features(self, entity_id: str, entity_type: str, features: Dict[str, Feature]) -> bool:
        """Store features for an entity"""
        try:
            start_time = time.time()

            # Prepare records for bulk insert
            records = []
            for feature_name, feature in features.items():
                expires_at = None
                if feature.ttl_hours:
                    expires_at = feature.timestamp + timedelta(hours=feature.ttl_hours)

                records.append({
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'feature_name': feature_name,
                    'feature_value': json.dumps(feature.value),
                    'feature_type': feature.feature_type,
                    'computed_at': feature.timestamp,
                    'ttl_hours': feature.ttl_hours,
                    'expires_at': expires_at,
                    'metadata': json.dumps(feature.metadata) if feature.metadata else None
                })

            # Bulk insert
            if records:
                df = pd.DataFrame(records)
                self.db.db.execute("""
                    INSERT OR REPLACE INTO quicknav.features
                    SELECT * FROM df
                """)

                # Update cache
                with self.cache_lock:
                    cache_key = f"{entity_type}:{entity_id}"
                    self.cache[cache_key] = features

                # Log computation
                computation_time = (time.time() - start_time) * 1000
                self._log_computation(entity_id, entity_type, len(features), computation_time, True)

                logger.debug(f"Stored {len(features)} features for {entity_type}:{entity_id}")
                return True

        except Exception as e:
            logger.error(f"Error storing features for {entity_type}:{entity_id}: {e}")
            computation_time = (time.time() - start_time) * 1000
            self._log_computation(entity_id, entity_type, 0, computation_time, False, str(e))
            return False

        return False

    def get_features(self, entity_id: str, entity_type: str,
                    feature_names: Optional[List[str]] = None) -> Dict[str, Feature]:
        """Get features for an entity"""
        cache_key = f"{entity_type}:{entity_id}"

        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                cached_features = self.cache[cache_key]
                # Filter expired features
                valid_features = {name: feature for name, feature in cached_features.items()
                                if not feature.is_expired()}
                if valid_features:
                    if feature_names:
                        return {name: feature for name, feature in valid_features.items()
                               if name in feature_names}
                    return valid_features

        # Load from database
        try:
            where_clause = "WHERE entity_id = ? AND entity_type = ?"
            params = [entity_id, entity_type]

            if feature_names:
                placeholders = ','.join(['?' for _ in feature_names])
                where_clause += f" AND feature_name IN ({placeholders})"
                params.extend(feature_names)

            # Filter out expired features
            where_clause += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(datetime.utcnow())

            results = self.db.db.execute(f"""
                SELECT feature_name, feature_value, feature_type, computed_at, ttl_hours, metadata
                FROM quicknav.features
                {where_clause}
            """, params).fetchall()

            features = {}
            for row in results:
                feature_name = row[0]
                feature_value = json.loads(row[1])
                feature_type = row[2]
                computed_at = datetime.fromisoformat(row[3]) if isinstance(row[3], str) else row[3]
                ttl_hours = row[4]
                metadata = json.loads(row[5]) if row[5] else {}

                features[feature_name] = Feature(
                    name=feature_name,
                    value=feature_value,
                    feature_type=feature_type,
                    timestamp=computed_at,
                    ttl_hours=ttl_hours,
                    metadata=metadata
                )

            # Update cache
            with self.cache_lock:
                self.cache[cache_key] = features

            return features

        except Exception as e:
            logger.error(f"Error loading features for {entity_type}:{entity_id}: {e}")
            return {}

    def get_feature_vector(self, entity_id: str, entity_type: str,
                          feature_names: List[str]) -> np.ndarray:
        """Get feature vector as numpy array"""
        features = self.get_features(entity_id, entity_type, feature_names)

        vector = []
        for feature_name in feature_names:
            if feature_name in features:
                feature = features[feature_name]
                if feature.feature_type == 'numeric':
                    vector.append(float(feature.value))
                elif feature.feature_type == 'boolean':
                    vector.append(1.0 if feature.value else 0.0)
                elif feature.feature_type == 'categorical':
                    # Simple hash-based encoding for categorical features
                    vector.append(float(hash(str(feature.value)) % 1000) / 1000.0)
                else:
                    vector.append(0.0)
            else:
                vector.append(0.0)  # Missing value

        return np.array(vector, dtype=np.float32)

    def compute_and_store_features(self, entity_id: str, entity_type: str,
                                  engineer: FeatureEngineer) -> bool:
        """Compute and store features for an entity"""
        try:
            if entity_type == 'project':
                features = engineer.compute_project_features(entity_id)
            elif entity_type == 'document':
                features = engineer.compute_document_features(entity_id)
            elif entity_type == 'user':
                features = engineer.compute_user_features(entity_id)
            else:
                logger.warning(f"Unknown entity type: {entity_type}")
                return False

            return self.store_features(entity_id, entity_type, features)

        except Exception as e:
            logger.error(f"Error computing features for {entity_type}:{entity_id}: {e}")
            return False

    def cleanup_expired_features(self) -> int:
        """Clean up expired features from storage"""
        try:
            result = self.db.db.execute("""
                DELETE FROM quicknav.features
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, [datetime.utcnow()])

            # Clear cache for safety
            with self.cache_lock:
                self.cache.clear()

            deleted_count = result.rowcount if hasattr(result, 'rowcount') else 0
            logger.info(f"Cleaned up {deleted_count} expired features")
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up expired features: {e}")
            return 0

    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        try:
            stats = self.db.db.execute("""
                SELECT
                    entity_type,
                    COUNT(*) as feature_count,
                    COUNT(DISTINCT entity_id) as entity_count,
                    COUNT(DISTINCT feature_name) as unique_features,
                    AVG(CASE WHEN expires_at IS NULL THEN 1 ELSE 0 END) as non_expiring_ratio
                FROM quicknav.features
                GROUP BY entity_type
            """).fetchall()

            statistics = {}
            for row in stats:
                entity_type = row[0]
                statistics[entity_type] = {
                    'feature_count': row[1],
                    'entity_count': row[2],
                    'unique_features': row[3],
                    'non_expiring_ratio': row[4]
                }

            # Overall statistics
            overall = self.db.db.execute("""
                SELECT
                    COUNT(*) as total_features,
                    COUNT(DISTINCT entity_id) as total_entities,
                    COUNT(CASE WHEN expires_at < ? THEN 1 END) as expired_features
                FROM quicknav.features
            """, [datetime.utcnow()]).fetchone()

            statistics['overall'] = {
                'total_features': overall[0],
                'total_entities': overall[1],
                'expired_features': overall[2],
                'cache_size': len(self.cache)
            }

            return statistics

        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return {}

    def _log_computation(self, entity_id: str, entity_type: str, feature_count: int,
                        computation_time_ms: float, success: bool, error_message: str = None):
        """Log feature computation"""
        try:
            log_id = str(uuid.uuid4())
            self.db.db.execute("""
                INSERT INTO quicknav.feature_computation_log
                (log_id, entity_id, entity_type, feature_count, computation_time_ms, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [log_id, entity_id, entity_type, feature_count, computation_time_ms, success, error_message])

        except Exception as e:
            logger.error(f"Error logging feature computation: {e}")


class FeatureServer:
    """Serves features for real-time ML inference"""

    def __init__(self, feature_store: FeatureStore, engineer: FeatureEngineer):
        self.feature_store = feature_store
        self.engineer = engineer
        self.config = get_config()

        # Pre-defined feature sets for different use cases
        self.feature_sets = {
            'project_ranking': [
                'total_files', 'document_count', 'doc_types_diversity',
                'activity_recency_score', 'popularity_score', 'has_standard_structure'
            ],
            'document_ranking': [
                'version_maturity_score', 'is_as_built', 'doc_type_priority',
                'access_frequency_score', 'file_size_mb', 'uniqueness_score'
            ],
            'user_personalization': [
                'user_activity_level', 'user_expertise_score', 'user_project_diversity'
            ]
        }

    async def get_features_for_ranking(self, entity_ids: List[str], entity_type: str,
                                     feature_set: str = None) -> Dict[str, np.ndarray]:
        """Get feature vectors for ranking multiple entities"""
        if feature_set is None:
            feature_set = f"{entity_type}_ranking"

        if feature_set not in self.feature_sets:
            raise ValueError(f"Unknown feature set: {feature_set}")

        feature_names = self.feature_sets[feature_set]
        result = {}

        # Process entities in parallel
        tasks = []
        for entity_id in entity_ids:
            task = asyncio.create_task(
                self._get_entity_features_async(entity_id, entity_type, feature_names)
            )
            tasks.append((entity_id, task))

        # Collect results
        for entity_id, task in tasks:
            try:
                features = await task
                if features is not None:
                    result[entity_id] = features
            except Exception as e:
                logger.error(f"Error getting features for {entity_id}: {e}")

        return result

    async def _get_entity_features_async(self, entity_id: str, entity_type: str,
                                       feature_names: List[str]) -> Optional[np.ndarray]:
        """Get features for a single entity asynchronously"""
        # Check if features exist and are fresh
        features = self.feature_store.get_features(entity_id, entity_type, feature_names)

        # Determine if we need to compute features
        missing_features = set(feature_names) - set(features.keys())
        expired_features = {name for name, feature in features.items() if feature.is_expired()}

        if missing_features or expired_features:
            # Compute fresh features
            success = await asyncio.get_event_loop().run_in_executor(
                None, self.feature_store.compute_and_store_features,
                entity_id, entity_type, self.engineer
            )

            if success:
                # Reload features
                features = self.feature_store.get_features(entity_id, entity_type, feature_names)

        # Convert to feature vector
        return self.feature_store.get_feature_vector(entity_id, entity_type, feature_names)

    def get_similarity_score(self, entity1_id: str, entity2_id: str,
                           entity_type: str, feature_set: str = None) -> float:
        """Calculate similarity score between two entities"""
        if feature_set is None:
            feature_set = f"{entity_type}_ranking"

        if feature_set not in self.feature_sets:
            return 0.0

        feature_names = self.feature_sets[feature_set]

        # Get feature vectors
        vector1 = self.feature_store.get_feature_vector(entity1_id, entity_type, feature_names)
        vector2 = self.feature_store.get_feature_vector(entity2_id, entity_type, feature_names)

        if len(vector1) == 0 or len(vector2) == 0:
            return 0.0

        # Calculate cosine similarity
        try:
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def get_recommendation_scores(self, user_id: str, candidate_ids: List[str],
                                entity_type: str) -> Dict[str, float]:
        """Get recommendation scores for candidates based on user preferences"""
        scores = {}

        try:
            # Get user features
            user_features = self.feature_store.get_features(user_id, 'user')
            if not user_features:
                return scores

            # Get user preferences
            preferred_doc_type = user_features.get('preferred_doc_type')
            doc_type_preferences = user_features.get('doc_type_preferences')

            # Score each candidate
            for candidate_id in candidate_ids:
                score = 0.0

                if entity_type == 'document':
                    # Get document features
                    doc_features = self.feature_store.get_features(candidate_id, 'document')

                    # Boost score based on document type preference
                    if preferred_doc_type and 'doc_type' in doc_features:
                        doc_type = doc_features['doc_type'].value
                        if doc_type == preferred_doc_type.value:
                            score += 2.0

                    # Add preference-based scoring
                    if doc_type_preferences and 'doc_type' in doc_features:
                        doc_type = doc_features['doc_type'].value
                        preference_score = doc_type_preferences.value.get(doc_type, 0)
                        score += preference_score * 3.0

                    # Add quality-based scoring
                    if 'is_as_built' in doc_features and doc_features['is_as_built'].value:
                        score += 1.0

                    if 'version_maturity_score' in doc_features:
                        score += doc_features['version_maturity_score'].value

                elif entity_type == 'project':
                    # Project recommendation scoring
                    project_features = self.feature_store.get_features(candidate_id, 'project')

                    if 'popularity_score' in project_features:
                        score += project_features['popularity_score'].value * 0.5

                    if 'activity_recency_score' in project_features:
                        score += project_features['activity_recency_score'].value

                scores[candidate_id] = score

        except Exception as e:
            logger.error(f"Error computing recommendation scores: {e}")

        return scores