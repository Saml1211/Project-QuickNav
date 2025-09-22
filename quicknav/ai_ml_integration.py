"""
ML Pipeline Integration for Enhanced AI Features

Integrates the existing data pipeline with AI capabilities to provide
predictive recommendations, usage forecasting, and intelligent suggestions
based on machine learning models.

Features:
- Integration with data pipeline for real-time predictions
- Document ranking and recommendation models
- Usage pattern prediction and forecasting
- Intelligent project and document suggestions
- Feature store integration for ML model serving
- Model performance monitoring and retraining
"""

import os
import json
import logging
import asyncio
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. ML model features will be disabled.")
    SKLEARN_AVAILABLE = False

try:
    from ..src.data_pipeline.config import get_config
    from ..src.data_pipeline.feature_store import FeatureStore
    from ..src.data_pipeline.cache import CacheManager
    PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("Data pipeline not available. Using mock implementations.")
    PIPELINE_AVAILABLE = False


class DocumentRankingModel:
    """ML model for ranking documents based on user preferences and context."""

    def __init__(self, model_path: str = "data/models/document_ranking.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False

        self._load_model()

    def _load_model(self):
        """Load trained model from disk if available."""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.feature_names = model_data['feature_names']
                self.is_trained = True

                logger.info(f"Loaded document ranking model from {self.model_path}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    def _extract_features(self, documents: List[Dict], user_context: Dict) -> np.ndarray:
        """Extract features for document ranking."""
        features = []

        for doc in documents:
            doc_features = []

            # Document type features
            doc_type = doc.get('document_type', 'unknown')
            doc_features.append(1 if doc_type == 'lld' else 0)
            doc_features.append(1 if doc_type == 'hld' else 0)
            doc_features.append(1 if doc_type == 'floor_plans' else 0)
            doc_features.append(1 if doc_type == 'photos' else 0)

            # File extension features
            file_ext = doc.get('file_extension', '').lower()
            doc_features.append(1 if file_ext == '.pdf' else 0)
            doc_features.append(1 if file_ext in ['.jpg', '.png', '.jpeg'] else 0)
            doc_features.append(1 if file_ext in ['.doc', '.docx'] else 0)

            # Project context features
            project_number = doc.get('project_number', '')
            is_recent_project = project_number in user_context.get('recent_projects', [])
            doc_features.append(1 if is_recent_project else 0)

            # Time-based features
            doc_path = doc.get('document_path', '')
            try:
                # Try to get file modification time
                if os.path.exists(doc_path):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(doc_path))
                    days_old = (datetime.now() - mod_time).days
                    doc_features.append(max(0, 365 - days_old) / 365)  # Recency score
                else:
                    doc_features.append(0.5)  # Default for non-existent files
            except:
                doc_features.append(0.5)

            # Text similarity features (if query provided)
            query = user_context.get('query', '')
            if query:
                doc_text = f"{doc.get('document_name', '')} {doc.get('project_name', '')}"
                similarity = self._calculate_text_similarity(query, doc_text)
                doc_features.append(similarity)
            else:
                doc_features.append(0.0)

            # Document name features
            doc_name = doc.get('document_name', '').lower()
            doc_features.append(1 if 'design' in doc_name else 0)
            doc_features.append(1 if 'final' in doc_name else 0)
            doc_features.append(1 if 'draft' in doc_name else 0)
            doc_features.append(1 if 'v' in doc_name or 'version' in doc_name else 0)

            features.append(doc_features)

        return np.array(features)

    def _calculate_text_similarity(self, query: str, document_text: str) -> float:
        """Calculate simple text similarity."""
        if not query or not document_text:
            return 0.0

        query_terms = set(query.lower().split())
        doc_terms = set(document_text.lower().split())

        if not query_terms:
            return 0.0

        intersection = query_terms.intersection(doc_terms)
        return len(intersection) / len(query_terms)

    def train(self, training_data: List[Dict], user_interactions: List[Dict]):
        """Train the document ranking model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train model: scikit-learn not available")
            return

        try:
            # Prepare training data
            features = []
            labels = []

            for interaction in user_interactions:
                doc = interaction.get('document')
                user_context = interaction.get('user_context', {})
                relevance_score = interaction.get('relevance_score', 0.5)

                if doc:
                    doc_features = self._extract_features([doc], user_context)[0]
                    features.append(doc_features)
                    labels.append(relevance_score)

            if len(features) < 10:
                logger.warning("Insufficient training data for document ranking model")
                return

            X = np.array(features)
            y = np.array(labels)

            # Store feature names
            self.feature_names = [
                'is_lld', 'is_hld', 'is_floor_plan', 'is_photo',
                'is_pdf', 'is_image', 'is_document',
                'is_recent_project', 'recency_score', 'text_similarity',
                'has_design', 'is_final', 'is_draft', 'has_version'
            ]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Convert regression to classification (high/medium/low relevance)
            y_train_class = np.digitize(y_train, bins=[0.33, 0.67]) - 1
            y_test_class = np.digitize(y_test, bins=[0.33, 0.67]) - 1

            self.model.fit(X_train, y_train_class)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test_class, y_pred)

            logger.info(f"Document ranking model trained with accuracy: {accuracy:.3f}")

            self.is_trained = True
            self._save_model()

        except Exception as e:
            logger.error(f"Failed to train document ranking model: {e}")

    def predict_relevance(self, documents: List[Dict], user_context: Dict) -> List[float]:
        """Predict relevance scores for documents."""
        if not self.is_trained or not self.model:
            # Return default scores if model not trained
            return [0.5] * len(documents)

        try:
            features = self._extract_features(documents, user_context)
            features_scaled = self.scaler.transform(features)

            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)

            # Convert class probabilities to relevance scores
            relevance_scores = []
            for probs in probabilities:
                # Weighted average: low=0.2, medium=0.5, high=0.8
                score = 0.2 * probs[0] + 0.5 * probs[1] + 0.8 * probs[2]
                relevance_scores.append(score)

            return relevance_scores

        except Exception as e:
            logger.error(f"Failed to predict relevance: {e}")
            return [0.5] * len(documents)

    def _save_model(self):
        """Save trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'trained_at': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved document ranking model to {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")


class UsagePredictionModel:
    """ML model for predicting user usage patterns and next actions."""

    def __init__(self, model_path: str = "data/models/usage_prediction.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

        self._load_model()

    def _load_model(self):
        """Load trained model from disk if available."""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True

                logger.info(f"Loaded usage prediction model from {self.model_path}")

            except Exception as e:
                logger.error(f"Failed to load usage model: {e}")

    def _extract_usage_features(self, user_history: List[Dict], current_context: Dict) -> np.ndarray:
        """Extract features for usage prediction."""
        features = []

        # Time-based features
        now = datetime.now()
        features.append(now.hour)  # Hour of day
        features.append(now.weekday())  # Day of week
        features.append(now.day)  # Day of month

        # Historical usage patterns
        if user_history:
            project_accesses = [h for h in user_history if h.get('event_type') == 'project_access']
            search_queries = [h for h in user_history if h.get('event_type') == 'search_query']

            features.append(len(project_accesses))  # Total project accesses
            features.append(len(search_queries))  # Total search queries

            # Recent activity (last 7 days)
            recent_cutoff = now - timedelta(days=7)
            recent_activity = [h for h in user_history if h.get('timestamp', now) > recent_cutoff]
            features.append(len(recent_activity))

            # Average session duration
            durations = [h.get('duration_seconds', 0) for h in user_history]
            features.append(np.mean(durations) if durations else 0)

            # Success rate
            successes = [h.get('success', False) for h in user_history]
            features.append(np.mean(successes) if successes else 0.5)

            # Most common project types
            project_ids = [h.get('project_id', '') for h in project_accesses if h.get('project_id')]
            unique_projects = len(set(project_ids))
            features.append(unique_projects)

        else:
            # Default values for new users
            features.extend([0, 0, 0, 0, 0.5, 0])

        # Current context features
        features.append(1 if current_context.get('has_active_project') else 0)
        features.append(len(current_context.get('recent_projects', [])))
        features.append(1 if current_context.get('in_search_session') else 0)

        return np.array(features).reshape(1, -1)

    def train(self, user_sessions: List[Dict]):
        """Train the usage prediction model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train usage model: scikit-learn not available")
            return

        try:
            features = []
            labels = []

            for session in user_sessions:
                user_history = session.get('history', [])
                current_context = session.get('context', {})
                next_action = session.get('next_action', 'unknown')

                session_features = self._extract_usage_features(user_history, current_context)[0]
                features.append(session_features)

                # Encode next action as label
                if next_action == 'project_access':
                    labels.append(0)
                elif next_action == 'search_query':
                    labels.append(1)
                elif next_action == 'document_view':
                    labels.append(2)
                else:
                    labels.append(3)  # other/unknown

            if len(features) < 20:
                logger.warning("Insufficient training data for usage prediction model")
                return

            X = np.array(features)
            y = np.array(labels)

            # Store feature names
            self.feature_names = [
                'hour_of_day', 'day_of_week', 'day_of_month',
                'total_project_accesses', 'total_searches', 'recent_activity',
                'avg_duration', 'success_rate', 'unique_projects',
                'has_active_project', 'recent_projects_count', 'in_search_session'
            ]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                max_depth=8
            )

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Usage prediction model trained with accuracy: {accuracy:.3f}")

            self.is_trained = True
            self._save_model()

        except Exception as e:
            logger.error(f"Failed to train usage prediction model: {e}")

    def predict_next_action(self, user_history: List[Dict], current_context: Dict) -> Dict[str, float]:
        """Predict the likelihood of different next actions."""
        if not self.is_trained or not self.model:
            # Return default probabilities
            return {
                'project_access': 0.25,
                'search_query': 0.25,
                'document_view': 0.25,
                'other': 0.25
            }

        try:
            features = self._extract_usage_features(user_history, current_context)
            features_scaled = self.scaler.transform(features)

            probabilities = self.model.predict_proba(features_scaled)[0]

            return {
                'project_access': probabilities[0],
                'search_query': probabilities[1],
                'document_view': probabilities[2],
                'other': probabilities[3]
            }

        except Exception as e:
            logger.error(f"Failed to predict next action: {e}")
            return {
                'project_access': 0.25,
                'search_query': 0.25,
                'document_view': 0.25,
                'other': 0.25
            }

    def _save_model(self):
        """Save trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'trained_at': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved usage prediction model to {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to save usage model: {e}")


class RecommendationEngine:
    """ML-powered recommendation engine for projects and documents."""

    def __init__(self):
        self.document_ranking_model = DocumentRankingModel()
        self.usage_prediction_model = UsagePredictionModel()
        self.feature_store = None
        self.cache_manager = None

        # Initialize data pipeline components if available
        if PIPELINE_AVAILABLE:
            try:
                config = get_config()
                self.feature_store = FeatureStore(config.feature_store)
                self.cache_manager = CacheManager(config.cache)
            except Exception as e:
                logger.warning(f"Failed to initialize pipeline components: {e}")

    async def get_document_recommendations(self, user_context: Dict,
                                         available_documents: List[Dict],
                                         limit: int = 10) -> List[Dict]:
        """Get ML-powered document recommendations."""
        try:
            # Get relevance scores from ML model
            relevance_scores = self.document_ranking_model.predict_relevance(
                available_documents, user_context
            )

            # Combine documents with scores
            scored_documents = []
            for doc, score in zip(available_documents, relevance_scores):
                scored_doc = doc.copy()
                scored_doc['ml_relevance_score'] = score
                scored_doc['recommendation_reason'] = self._generate_recommendation_reason(doc, score, user_context)
                scored_documents.append(scored_doc)

            # Sort by relevance and return top results
            scored_documents.sort(key=lambda x: x['ml_relevance_score'], reverse=True)
            return scored_documents[:limit]

        except Exception as e:
            logger.error(f"Failed to get document recommendations: {e}")
            return available_documents[:limit]

    async def get_project_recommendations(self, user_context: Dict,
                                        available_projects: List[Dict],
                                        limit: int = 5) -> List[Dict]:
        """Get ML-powered project recommendations."""
        try:
            # Use usage prediction to influence project recommendations
            next_action_probs = self.usage_prediction_model.predict_next_action(
                user_context.get('user_history', []),
                user_context
            )

            # Score projects based on context and predictions
            scored_projects = []
            for project in available_projects:
                score = self._calculate_project_score(project, user_context, next_action_probs)
                scored_project = project.copy()
                scored_project['ml_recommendation_score'] = score
                scored_project['recommendation_reason'] = self._generate_project_recommendation_reason(
                    project, score, user_context
                )
                scored_projects.append(scored_project)

            # Sort by score and return top results
            scored_projects.sort(key=lambda x: x['ml_recommendation_score'], reverse=True)
            return scored_projects[:limit]

        except Exception as e:
            logger.error(f"Failed to get project recommendations: {e}")
            return available_projects[:limit]

    def _calculate_project_score(self, project: Dict, user_context: Dict,
                               next_action_probs: Dict) -> float:
        """Calculate project recommendation score."""
        score = 0.0

        # Base score from recent access
        recent_projects = user_context.get('recent_projects', [])
        project_id = project.get('project_number') or project.get('project_name', '')

        if project_id in recent_projects:
            recency_index = recent_projects.index(project_id)
            score += (5 - recency_index) / 5 * 0.4  # Higher score for more recent

        # Score based on predicted next action
        if next_action_probs.get('project_access', 0) > 0.5:
            score += 0.3

        # Score based on project type similarity
        user_history = user_context.get('user_history', [])
        historical_projects = [h.get('project_id', '') for h in user_history
                             if h.get('event_type') == 'project_access']

        if historical_projects:
            # Simple similarity based on project number range
            project_num = self._extract_project_number(project_id)
            historical_nums = [self._extract_project_number(p) for p in historical_projects]

            if project_num and historical_nums:
                valid_nums = [n for n in historical_nums if n]
                if valid_nums:
                    avg_historical = np.mean(valid_nums)
                    # Closer project numbers might be related
                    similarity = 1.0 / (1.0 + abs(project_num - avg_historical) / 1000)
                    score += similarity * 0.2

        # Current context boost
        if user_context.get('has_active_project'):
            score += 0.1

        return min(score, 1.0)

    def _extract_project_number(self, project_id: str) -> Optional[int]:
        """Extract project number from project ID."""
        import re
        match = re.search(r'\b(\d{5})\b', project_id)
        return int(match.group(1)) if match else None

    def _generate_recommendation_reason(self, document: Dict, score: float,
                                      user_context: Dict) -> str:
        """Generate human-readable recommendation reason."""
        reasons = []

        if score > 0.7:
            reasons.append("High relevance based on your usage patterns")
        elif score > 0.5:
            reasons.append("Good match for your current context")

        doc_type = document.get('document_type', '')
        if doc_type in ['lld', 'hld'] and score > 0.6:
            reasons.append("Important design document")

        if document.get('project_number') in user_context.get('recent_projects', []):
            reasons.append("From a recently accessed project")

        query = user_context.get('query', '')
        if query and any(term in document.get('document_name', '').lower()
                        for term in query.lower().split()):
            reasons.append("Matches your search terms")

        return "; ".join(reasons) if reasons else "Recommended based on ML analysis"

    def _generate_project_recommendation_reason(self, project: Dict, score: float,
                                              user_context: Dict) -> str:
        """Generate human-readable project recommendation reason."""
        reasons = []

        project_id = project.get('project_number') or project.get('project_name', '')

        if project_id in user_context.get('recent_projects', []):
            reasons.append("Recently accessed project")

        if score > 0.7:
            reasons.append("High likelihood of being your next target")

        if user_context.get('user_history'):
            reasons.append("Based on your usage patterns")

        return "; ".join(reasons) if reasons else "ML-recommended project"

    async def get_smart_suggestions(self, user_context: Dict) -> Dict[str, List[str]]:
        """Get smart suggestions based on ML predictions."""
        try:
            # Predict next action
            next_action_probs = self.usage_prediction_model.predict_next_action(
                user_context.get('user_history', []),
                user_context
            )

            suggestions = {
                'actions': [],
                'projects': [],
                'searches': [],
                'general': []
            }

            # Action suggestions based on predictions
            if next_action_probs['project_access'] > 0.4:
                suggestions['actions'].append("You might want to access a project next")
                suggestions['actions'].append("Consider browsing your recent projects")

            if next_action_probs['search_query'] > 0.4:
                suggestions['actions'].append("A search might help you find what you need")
                suggestions['actions'].append("Try searching for specific document types")

            if next_action_probs['document_view'] > 0.4:
                suggestions['actions'].append("You might be looking for specific documents")

            # Project suggestions
            recent_projects = user_context.get('recent_projects', [])
            if recent_projects:
                suggestions['projects'].extend([
                    f"Continue working with project {project}"
                    for project in recent_projects[:3]
                ])

            # Search suggestions based on patterns
            user_history = user_context.get('user_history', [])
            search_history = [h.get('query_text', '') for h in user_history
                            if h.get('event_type') == 'search_query' and h.get('query_text')]

            if search_history:
                # Suggest related searches
                common_terms = self._extract_common_terms(search_history)
                suggestions['searches'].extend([
                    f"Search for {term} documents"
                    for term in common_terms[:3]
                ])

            # General AI-powered suggestions
            if len(user_context.get('user_history', [])) > 10:
                suggestions['general'].append("Your usage patterns suggest you're an experienced user")
                suggestions['general'].append("Consider using advanced search features")

            return suggestions

        except Exception as e:
            logger.error(f"Failed to get smart suggestions: {e}")
            return {'actions': [], 'projects': [], 'searches': [], 'general': []}

    def _extract_common_terms(self, search_history: List[str]) -> List[str]:
        """Extract common terms from search history."""
        all_terms = []
        for query in search_history:
            terms = query.lower().split()
            all_terms.extend(term for term in terms if len(term) > 2)

        from collections import Counter
        term_counts = Counter(all_terms)
        return [term for term, count in term_counts.most_common(10)]

    async def update_models(self, training_data: Dict):
        """Update ML models with new training data."""
        try:
            # Update document ranking model
            if 'document_interactions' in training_data:
                logger.info("Updating document ranking model...")
                self.document_ranking_model.train(
                    training_data.get('documents', []),
                    training_data['document_interactions']
                )

            # Update usage prediction model
            if 'user_sessions' in training_data:
                logger.info("Updating usage prediction model...")
                self.usage_prediction_model.train(training_data['user_sessions'])

            logger.info("ML models updated successfully")

        except Exception as e:
            logger.error(f"Failed to update ML models: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of ML models."""
        return {
            'document_ranking': {
                'trained': self.document_ranking_model.is_trained,
                'model_path': str(self.document_ranking_model.model_path),
                'features': len(self.document_ranking_model.feature_names)
            },
            'usage_prediction': {
                'trained': self.usage_prediction_model.is_trained,
                'model_path': str(self.usage_prediction_model.model_path),
                'features': len(self.usage_prediction_model.feature_names)
            },
            'pipeline_integration': {
                'feature_store_available': self.feature_store is not None,
                'cache_manager_available': self.cache_manager is not None
            }
        }


class MLPipelineIntegrator:
    """Integrates ML capabilities with the existing data pipeline."""

    def __init__(self):
        self.recommendation_engine = RecommendationEngine()
        self.feature_cache = {}
        self.prediction_cache = {}

    async def get_ml_enhanced_search_results(self, query: str, raw_results: List[Dict],
                                           user_context: Dict) -> List[Dict]:
        """Enhance search results using ML predictions."""
        try:
            # Get ML recommendations for the results
            enhanced_results = await self.recommendation_engine.get_document_recommendations(
                user_context, raw_results, limit=len(raw_results)
            )

            # Add ML insights to each result
            for result in enhanced_results:
                result['ml_insights'] = {
                    'relevance_score': result.get('ml_relevance_score', 0.5),
                    'recommendation_reason': result.get('recommendation_reason', ''),
                    'ai_enhanced': True
                }

            return enhanced_results

        except Exception as e:
            logger.error(f"Failed to enhance search results with ML: {e}")
            return raw_results

    async def get_predictive_suggestions(self, user_context: Dict) -> Dict[str, Any]:
        """Get predictive suggestions based on ML models and pipeline data."""
        try:
            # Get smart suggestions from recommendation engine
            suggestions = await self.recommendation_engine.get_smart_suggestions(user_context)

            # Add pipeline-based predictions if available
            if self.recommendation_engine.feature_store:
                # Get features from feature store
                pipeline_features = await self._get_pipeline_features(user_context)
                if pipeline_features:
                    suggestions['pipeline_insights'] = pipeline_features

            return suggestions

        except Exception as e:
            logger.error(f"Failed to get predictive suggestions: {e}")
            return {}

    async def _get_pipeline_features(self, user_context: Dict) -> Dict[str, Any]:
        """Get features from the data pipeline feature store."""
        try:
            # This would integrate with the actual feature store
            # For now, return mock features
            return {
                'trending_projects': ['17741', '17742'],
                'popular_document_types': ['lld', 'floor_plans'],
                'peak_usage_hours': [9, 14, 16]
            }

        except Exception as e:
            logger.error(f"Failed to get pipeline features: {e}")
            return {}

    async def train_models_from_pipeline(self):
        """Train ML models using data from the pipeline."""
        try:
            # This would extract training data from the pipeline
            # For now, create mock training data
            training_data = {
                'document_interactions': [
                    {
                        'document': {'document_type': 'lld', 'project_number': '17741'},
                        'user_context': {'recent_projects': ['17741']},
                        'relevance_score': 0.8
                    }
                ],
                'user_sessions': [
                    {
                        'history': [{'event_type': 'project_access', 'success': True}],
                        'context': {'has_active_project': True},
                        'next_action': 'search_query'
                    }
                ]
            }

            await self.recommendation_engine.update_models(training_data)
            logger.info("Models trained from pipeline data")

        except Exception as e:
            logger.error(f"Failed to train models from pipeline: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get ML pipeline integration status."""
        model_status = self.recommendation_engine.get_model_status()

        return {
            'ml_models': model_status,
            'pipeline_integration': {
                'data_pipeline_available': PIPELINE_AVAILABLE,
                'sklearn_available': SKLEARN_AVAILABLE,
                'feature_cache_size': len(self.feature_cache),
                'prediction_cache_size': len(self.prediction_cache)
            },
            'capabilities': {
                'document_ranking': model_status['document_ranking']['trained'],
                'usage_prediction': model_status['usage_prediction']['trained'],
                'smart_suggestions': True,
                'search_enhancement': True
            }
        }


# Export main classes
__all__ = [
    'MLPipelineIntegrator',
    'RecommendationEngine',
    'DocumentRankingModel',
    'UsagePredictionModel'
]