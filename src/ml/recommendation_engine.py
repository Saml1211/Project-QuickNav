"""
Advanced ML-powered Recommendation Engine for Project QuickNav

This module implements intelligent project and document recommendations using
multiple algorithms including collaborative filtering, content-based filtering,
and hybrid approaches.

Features:
- Real-time personalized recommendations
- Collaborative filtering using matrix factorization
- Content-based similarity using TF-IDF and embeddings
- Temporal pattern analysis for navigation prediction
- Context-aware recommendations based on usage patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib
import time

# Configure logging
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Advanced recommendation engine using multiple ML algorithms
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize recommendation engine

        Args:
            data_dir: Directory containing training data and models
        """
        self.data_dir = Path(data_dir) if data_dir else Path("training_data")
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Model components
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.collaborative_model = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.scaler = StandardScaler()

        # Data caches
        self.project_features = {}
        self.user_profiles = {}
        self.interaction_matrix = None
        self.temporal_patterns = {}

        # Configuration
        self.config = {
            'content_weight': 0.3,
            'collaborative_weight': 0.4,
            'temporal_weight': 0.2,
            'popularity_weight': 0.1,
            'min_interactions': 3,
            'embedding_dim': 50,
            'recency_decay': 0.95,
            'max_recommendations': 20
        }

        # Load models if they exist
        self._load_models()

    def train_models(self, training_data: List[Dict], user_history: List[Dict]) -> Dict[str, Any]:
        """
        Train all recommendation models using training data and user history

        Args:
            training_data: List of project/document metadata
            user_history: List of user interaction events

        Returns:
            Training metrics and model performance
        """
        logger.info("Starting recommendation model training...")
        start_time = time.time()

        # Prepare data
        project_df = pd.DataFrame(training_data)
        history_df = pd.DataFrame(user_history)

        # Train content-based model
        content_metrics = self._train_content_model(project_df)

        # Train collaborative filtering model
        collaborative_metrics = self._train_collaborative_model(history_df)

        # Build temporal patterns
        temporal_metrics = self._build_temporal_patterns(history_df)

        # Calculate popularity scores
        popularity_metrics = self._calculate_popularity_scores(project_df, history_df)

        # Save models
        self._save_models()

        training_time = time.time() - start_time

        metrics = {
            'training_time': training_time,
            'content_based': content_metrics,
            'collaborative': collaborative_metrics,
            'temporal': temporal_metrics,
            'popularity': popularity_metrics,
            'total_projects': len(project_df),
            'total_interactions': len(history_df),
            'model_version': self._get_model_version()
        }

        logger.info(f"Model training completed in {training_time:.2f}s")
        return metrics

    def get_recommendations(self, user_id: str, context: Dict = None,
                          num_recommendations: int = 10) -> List[Dict]:
        """
        Get personalized recommendations for a user

        Args:
            user_id: User identifier
            context: Current context (time, recent actions, etc.)
            num_recommendations: Number of recommendations to return

        Returns:
            List of recommended projects with scores and explanations
        """
        if not self._models_ready():
            logger.warning("Models not trained, returning fallback recommendations")
            return self._get_fallback_recommendations(num_recommendations)

        # Get recommendations from each algorithm
        content_recs = self._get_content_recommendations(user_id, context)
        collaborative_recs = self._get_collaborative_recommendations(user_id, context)
        temporal_recs = self._get_temporal_recommendations(user_id, context)
        popularity_recs = self._get_popularity_recommendations(context)

        # Hybrid scoring
        recommendations = self._combine_recommendations(
            content_recs, collaborative_recs, temporal_recs, popularity_recs
        )

        # Filter and rank
        final_recs = self._finalize_recommendations(
            recommendations, user_id, context, num_recommendations
        )

        return final_recs

    def update_user_interaction(self, user_id: str, project_id: str,
                              interaction_type: str, metadata: Dict = None):
        """
        Update user interaction for real-time learning

        Args:
            user_id: User identifier
            project_id: Project identifier
            interaction_type: Type of interaction (view, search, open, etc.)
            metadata: Additional interaction metadata
        """
        timestamp = datetime.utcnow()

        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interactions': [],
                'preferences': {},
                'temporal_patterns': {},
                'last_active': timestamp
            }

        interaction = {
            'project_id': project_id,
            'interaction_type': interaction_type,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }

        self.user_profiles[user_id]['interactions'].append(interaction)
        self.user_profiles[user_id]['last_active'] = timestamp

        # Update preferences based on interaction
        self._update_user_preferences(user_id, project_id, interaction_type, metadata)

        # Update temporal patterns
        self._update_temporal_patterns(user_id, timestamp, interaction_type)

    def get_similar_projects(self, project_id: str, num_similar: int = 5) -> List[Dict]:
        """
        Get projects similar to a given project

        Args:
            project_id: Reference project ID
            num_similar: Number of similar projects to return

        Returns:
            List of similar projects with similarity scores
        """
        if not self.content_similarity_matrix is None:
            return self._get_content_similar_projects(project_id, num_similar)
        else:
            return []

    def predict_next_action(self, user_id: str, current_context: Dict) -> Dict:
        """
        Predict user's next likely action based on patterns

        Args:
            user_id: User identifier
            current_context: Current session context

        Returns:
            Prediction with confidence score and suggested actions
        """
        if user_id not in self.user_profiles:
            return {'prediction': 'explore', 'confidence': 0.1, 'suggestions': []}

        user_profile = self.user_profiles[user_id]
        recent_interactions = user_profile['interactions'][-10:]  # Last 10 actions

        # Analyze patterns
        action_sequence = [i['interaction_type'] for i in recent_interactions]
        temporal_pattern = self._analyze_temporal_pattern(user_id, current_context)

        # Simple pattern matching (can be enhanced with LSTM)
        if len(action_sequence) >= 3:
            last_three = tuple(action_sequence[-3:])
            if last_three in self.temporal_patterns.get(user_id, {}):
                pattern_data = self.temporal_patterns[user_id][last_three]
                most_likely_next = max(pattern_data['next_actions'],
                                     key=pattern_data['next_actions'].get)
                confidence = pattern_data['next_actions'][most_likely_next] / sum(pattern_data['next_actions'].values())

                return {
                    'prediction': most_likely_next,
                    'confidence': confidence,
                    'suggestions': self._generate_action_suggestions(most_likely_next, current_context)
                }

        # Fallback to general patterns
        return {
            'prediction': 'search',
            'confidence': 0.3,
            'suggestions': ['Try searching for a project', 'Browse recent projects']
        }

    def get_analytics_insights(self) -> Dict[str, Any]:
        """
        Get analytics insights about recommendation performance

        Returns:
            Dictionary containing various analytics metrics
        """
        insights = {
            'model_status': {
                'content_model_ready': self.content_similarity_matrix is not None,
                'collaborative_model_ready': self.collaborative_model is not None,
                'total_users': len(self.user_profiles),
                'total_projects': len(self.project_features)
            },
            'user_engagement': self._calculate_user_engagement(),
            'recommendation_performance': self._calculate_recommendation_performance(),
            'popular_projects': self._get_popular_projects(),
            'temporal_insights': self._get_temporal_insights()
        }

        return insights

    # Private methods

    def _train_content_model(self, project_df: pd.DataFrame) -> Dict[str, Any]:
        """Train content-based recommendation model"""
        logger.info("Training content-based model...")

        # Extract text features from project metadata
        text_features = []
        project_ids = []

        for _, row in project_df.iterrows():
            # Combine project folder name and document names for content analysis
            content = f"{row.get('project_folder', '')} "
            if 'document_name' in row:
                content += f"{row['document_name']} "
            if 'extracted_info' in row and isinstance(row['extracted_info'], dict):
                content += " ".join(str(v) for v in row['extracted_info'].values())

            text_features.append(content.strip())
            project_ids.append(row.get('project_folder', f"project_{len(project_ids)}"))

        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)

        # Calculate similarity matrix
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)

        # Store project features
        for i, project_id in enumerate(project_ids):
            self.project_features[project_id] = {
                'tfidf_vector': tfidf_matrix[i],
                'similarity_index': i,
                'metadata': project_df.iloc[i].to_dict()
            }

        return {
            'num_projects': len(project_ids),
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
            'avg_similarity': np.mean(self.content_similarity_matrix),
            'sparsity': 1 - (np.count_nonzero(tfidf_matrix) / tfidf_matrix.shape[0] / tfidf_matrix.shape[1])
        }

    def _train_collaborative_model(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        """Train collaborative filtering model"""
        logger.info("Training collaborative filtering model...")

        if len(history_df) == 0:
            return {'status': 'no_data', 'num_interactions': 0}

        # Create user-item interaction matrix
        # Assume history has 'user_id', 'project_code', and 'timestamp'
        if 'user_id' not in history_df.columns or 'project_code' not in history_df.columns:
            return {'status': 'insufficient_columns', 'columns': list(history_df.columns)}

        # Count interactions per user-project pair
        interaction_counts = history_df.groupby(['user_id', 'project_code']).size().reset_index(name='count')

        # Create pivot table
        self.interaction_matrix = interaction_counts.pivot(
            index='user_id',
            columns='project_code',
            values='count'
        ).fillna(0)

        # Apply SVD for matrix factorization
        if self.interaction_matrix.shape[0] > 1 and self.interaction_matrix.shape[1] > 1:
            n_components = min(self.config['embedding_dim'],
                             min(self.interaction_matrix.shape) - 1)

            self.collaborative_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.user_embeddings = self.collaborative_model.fit_transform(self.interaction_matrix)
            self.item_embeddings = self.collaborative_model.components_.T

            explained_variance = np.sum(self.collaborative_model.explained_variance_ratio_)
        else:
            explained_variance = 0

        return {
            'num_users': len(self.interaction_matrix.index),
            'num_projects': len(self.interaction_matrix.columns),
            'total_interactions': interaction_counts['count'].sum(),
            'sparsity': 1 - (np.count_nonzero(self.interaction_matrix) / self.interaction_matrix.size),
            'explained_variance': explained_variance
        }

    def _build_temporal_patterns(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        """Build temporal usage patterns"""
        logger.info("Building temporal patterns...")

        if len(history_df) == 0:
            return {'status': 'no_data'}

        # Group by user and analyze sequences
        for user_id, user_data in history_df.groupby('user_id'):
            user_data = user_data.sort_values('timestamp')
            actions = user_data['action'].tolist() if 'action' in user_data.columns else ['unknown'] * len(user_data)

            # Build n-gram patterns
            if user_id not in self.temporal_patterns:
                self.temporal_patterns[user_id] = {}

            for i in range(len(actions) - 3):
                pattern = tuple(actions[i:i+3])
                next_action = actions[i+3]

                if pattern not in self.temporal_patterns[user_id]:
                    self.temporal_patterns[user_id][pattern] = {'next_actions': {}, 'count': 0}

                if next_action not in self.temporal_patterns[user_id][pattern]['next_actions']:
                    self.temporal_patterns[user_id][pattern]['next_actions'][next_action] = 0

                self.temporal_patterns[user_id][pattern]['next_actions'][next_action] += 1
                self.temporal_patterns[user_id][pattern]['count'] += 1

        total_patterns = sum(len(patterns) for patterns in self.temporal_patterns.values())

        return {
            'num_users_with_patterns': len(self.temporal_patterns),
            'total_patterns': total_patterns,
            'avg_patterns_per_user': total_patterns / max(len(self.temporal_patterns), 1)
        }

    def _calculate_popularity_scores(self, project_df: pd.DataFrame,
                                   history_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate project popularity scores"""
        logger.info("Calculating popularity scores...")

        popularity = {}

        # Count project mentions in training data
        if 'project_folder' in project_df.columns:
            project_counts = project_df['project_folder'].value_counts()
            for project, count in project_counts.items():
                popularity[project] = count

        # Add interaction-based popularity
        if 'project_code' in history_df.columns:
            interaction_counts = history_df['project_code'].value_counts()
            for project, count in interaction_counts.items():
                popularity[project] = popularity.get(project, 0) + count * 2  # Weight interactions higher

        # Normalize scores
        if popularity:
            max_score = max(popularity.values())
            popularity = {k: v / max_score for k, v in popularity.items()}

        return {
            'num_projects_with_popularity': len(popularity),
            'avg_popularity': np.mean(list(popularity.values())) if popularity else 0,
            'max_popularity': max(popularity.values()) if popularity else 0
        }

    def _get_content_recommendations(self, user_id: str, context: Dict) -> List[Tuple[str, float]]:
        """Get content-based recommendations"""
        if self.content_similarity_matrix is None:
            return []

        user_profile = self.user_profiles.get(user_id, {})
        recent_projects = [i['project_id'] for i in user_profile.get('interactions', [])[-5:]]

        if not recent_projects:
            return []

        # Average similarity scores for recent projects
        similarity_scores = np.zeros(self.content_similarity_matrix.shape[0])
        count = 0

        for project_id in recent_projects:
            if project_id in self.project_features:
                idx = self.project_features[project_id]['similarity_index']
                similarity_scores += self.content_similarity_matrix[idx]
                count += 1

        if count > 0:
            similarity_scores /= count

            # Get top recommendations
            top_indices = np.argsort(similarity_scores)[::-1][:20]
            recommendations = []

            for idx in top_indices:
                project_id = list(self.project_features.keys())[idx]
                if project_id not in recent_projects:  # Don't recommend recently viewed
                    recommendations.append((project_id, similarity_scores[idx]))

            return recommendations

        return []

    def _get_collaborative_recommendations(self, user_id: str, context: Dict) -> List[Tuple[str, float]]:
        """Get collaborative filtering recommendations"""
        if self.collaborative_model is None or self.interaction_matrix is None:
            return []

        if user_id not in self.interaction_matrix.index:
            return []

        user_idx = self.interaction_matrix.index.get_loc(user_id)
        user_embedding = self.user_embeddings[user_idx]

        # Calculate scores for all items
        scores = np.dot(self.item_embeddings, user_embedding)

        # Get user's existing interactions
        user_interactions = self.interaction_matrix.loc[user_id]
        interacted_items = user_interactions[user_interactions > 0].index.tolist()

        # Create recommendations
        recommendations = []
        for i, project_id in enumerate(self.interaction_matrix.columns):
            if project_id not in interacted_items:
                recommendations.append((project_id, scores[i]))

        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:20]

    def _get_temporal_recommendations(self, user_id: str, context: Dict) -> List[Tuple[str, float]]:
        """Get temporal pattern-based recommendations"""
        if user_id not in self.user_profiles:
            return []

        current_time = datetime.utcnow()
        hour = current_time.hour
        day_of_week = current_time.weekday()

        # Simple temporal scoring based on usage patterns
        recommendations = []

        for project_id, feature_data in self.project_features.items():
            # Calculate temporal relevance score
            score = 0.5  # Base score

            # Time-based adjustments (can be enhanced with actual temporal analysis)
            if 9 <= hour <= 17:  # Business hours
                score += 0.2
            if day_of_week < 5:  # Weekdays
                score += 0.1

            recommendations.append((project_id, score))

        return recommendations

    def _get_popularity_recommendations(self, context: Dict) -> List[Tuple[str, float]]:
        """Get popularity-based recommendations"""
        popularity_scores = getattr(self, 'popularity_scores', {})

        recommendations = [(project_id, score) for project_id, score in popularity_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:20]

    def _combine_recommendations(self, content_recs: List[Tuple[str, float]],
                               collaborative_recs: List[Tuple[str, float]],
                               temporal_recs: List[Tuple[str, float]],
                               popularity_recs: List[Tuple[str, float]]) -> Dict[str, float]:
        """Combine recommendations from different algorithms"""
        combined_scores = {}

        # Normalize and combine scores
        for project_id, score in content_recs:
            combined_scores[project_id] = combined_scores.get(project_id, 0) + score * self.config['content_weight']

        for project_id, score in collaborative_recs:
            combined_scores[project_id] = combined_scores.get(project_id, 0) + score * self.config['collaborative_weight']

        for project_id, score in temporal_recs:
            combined_scores[project_id] = combined_scores.get(project_id, 0) + score * self.config['temporal_weight']

        for project_id, score in popularity_recs:
            combined_scores[project_id] = combined_scores.get(project_id, 0) + score * self.config['popularity_weight']

        return combined_scores

    def _finalize_recommendations(self, recommendations: Dict[str, float],
                                user_id: str, context: Dict,
                                num_recommendations: int) -> List[Dict]:
        """Finalize and format recommendations"""
        # Sort by combined score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        final_recs = []
        for i, (project_id, score) in enumerate(sorted_recs[:num_recommendations]):
            rec = {
                'project_id': project_id,
                'score': float(score),
                'rank': i + 1,
                'explanation': self._generate_explanation(project_id, user_id, score),
                'metadata': self.project_features.get(project_id, {}).get('metadata', {}),
                'confidence': min(score, 1.0)
            }
            final_recs.append(rec)

        return final_recs

    def _generate_explanation(self, project_id: str, user_id: str, score: float) -> str:
        """Generate explanation for recommendation"""
        explanations = []

        if score > 0.7:
            explanations.append("Highly relevant based on your activity patterns")
        elif score > 0.5:
            explanations.append("Similar to projects you've accessed recently")
        elif score > 0.3:
            explanations.append("Popular among users with similar interests")
        else:
            explanations.append("Trending project worth exploring")

        return "; ".join(explanations)

    def _models_ready(self) -> bool:
        """Check if models are ready for recommendations"""
        return (self.content_similarity_matrix is not None or
                self.collaborative_model is not None)

    def _get_fallback_recommendations(self, num_recommendations: int) -> List[Dict]:
        """Get fallback recommendations when models aren't ready"""
        # Return most recently added projects or popular ones
        fallback_projects = list(self.project_features.keys())[:num_recommendations]

        recommendations = []
        for i, project_id in enumerate(fallback_projects):
            rec = {
                'project_id': project_id,
                'score': 0.5,
                'rank': i + 1,
                'explanation': "Recently available project",
                'metadata': self.project_features.get(project_id, {}).get('metadata', {}),
                'confidence': 0.3
            }
            recommendations.append(rec)

        return recommendations

    def _update_user_preferences(self, user_id: str, project_id: str,
                               interaction_type: str, metadata: Dict):
        """Update user preferences based on interaction"""
        preferences = self.user_profiles[user_id]['preferences']

        # Update interaction type preferences
        if 'interaction_types' not in preferences:
            preferences['interaction_types'] = {}
        preferences['interaction_types'][interaction_type] = preferences['interaction_types'].get(interaction_type, 0) + 1

        # Update project category preferences if available
        if metadata and 'category' in metadata:
            if 'categories' not in preferences:
                preferences['categories'] = {}
            category = metadata['category']
            preferences['categories'][category] = preferences['categories'].get(category, 0) + 1

    def _update_temporal_patterns(self, user_id: str, timestamp: datetime, interaction_type: str):
        """Update temporal usage patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        temporal = self.user_profiles[user_id].setdefault('temporal_patterns', {})

        # Update hourly patterns
        hourly = temporal.setdefault('hourly', {})
        hourly[hour] = hourly.get(hour, 0) + 1

        # Update daily patterns
        daily = temporal.setdefault('daily', {})
        daily[day_of_week] = daily.get(day_of_week, 0) + 1

        # Update interaction type by time
        time_interaction = temporal.setdefault('time_interaction', {})
        key = f"{hour}_{interaction_type}"
        time_interaction[key] = time_interaction.get(key, 0) + 1

    def _analyze_temporal_pattern(self, user_id: str, current_context: Dict) -> Dict:
        """Analyze temporal patterns for current context"""
        if user_id not in self.user_profiles:
            return {}

        temporal = self.user_profiles[user_id].get('temporal_patterns', {})
        current_hour = datetime.utcnow().hour

        analysis = {
            'current_hour_activity': temporal.get('hourly', {}).get(current_hour, 0),
            'preferred_hours': sorted(temporal.get('hourly', {}).items(),
                                   key=lambda x: x[1], reverse=True)[:3],
            'activity_score': self._calculate_activity_score(temporal, current_hour)
        }

        return analysis

    def _calculate_activity_score(self, temporal: Dict, current_hour: int) -> float:
        """Calculate activity score for current hour"""
        hourly = temporal.get('hourly', {})
        if not hourly:
            return 0.5

        current_activity = hourly.get(current_hour, 0)
        max_activity = max(hourly.values())

        return current_activity / max_activity if max_activity > 0 else 0.5

    def _calculate_user_engagement(self) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        if not self.user_profiles:
            return {'total_users': 0, 'avg_interactions': 0}

        total_interactions = sum(len(profile['interactions']) for profile in self.user_profiles.values())
        active_users = len([u for u in self.user_profiles.values()
                          if u['last_active'] > datetime.utcnow() - timedelta(days=7)])

        return {
            'total_users': len(self.user_profiles),
            'active_users_7d': active_users,
            'avg_interactions_per_user': total_interactions / len(self.user_profiles),
            'engagement_rate': active_users / len(self.user_profiles)
        }

    def _calculate_recommendation_performance(self) -> Dict[str, Any]:
        """Calculate recommendation performance metrics"""
        # This would typically require tracking recommendation clicks/conversions
        # For now, return basic model readiness metrics
        return {
            'content_model_ready': self.content_similarity_matrix is not None,
            'collaborative_model_ready': self.collaborative_model is not None,
            'avg_recommendation_score': 0.65,  # Placeholder
            'recommendation_diversity': 0.75   # Placeholder
        }

    def _get_popular_projects(self) -> List[Dict]:
        """Get most popular projects"""
        popularity = getattr(self, 'popularity_scores', {})
        popular = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:10]

        return [{'project_id': pid, 'popularity_score': score} for pid, score in popular]

    def _get_temporal_insights(self) -> Dict[str, Any]:
        """Get temporal usage insights"""
        all_hourly = {}
        all_daily = {}

        for profile in self.user_profiles.values():
            temporal = profile.get('temporal_patterns', {})

            # Aggregate hourly patterns
            for hour, count in temporal.get('hourly', {}).items():
                all_hourly[hour] = all_hourly.get(hour, 0) + count

            # Aggregate daily patterns
            for day, count in temporal.get('daily', {}).items():
                all_daily[day] = all_daily.get(day, 0) + count

        peak_hour = max(all_hourly.items(), key=lambda x: x[1])[0] if all_hourly else 9
        peak_day = max(all_daily.items(), key=lambda x: x[1])[0] if all_daily else 0

        return {
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'hourly_distribution': all_hourly,
            'daily_distribution': all_daily
        }

    def _save_models(self):
        """Save trained models to disk"""
        models_to_save = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'content_similarity_matrix': self.content_similarity_matrix,
            'collaborative_model': self.collaborative_model,
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings,
            'interaction_matrix': self.interaction_matrix,
            'project_features': self.project_features,
            'temporal_patterns': self.temporal_patterns,
            'config': self.config
        }

        for name, model in models_to_save.items():
            if model is not None:
                try:
                    model_path = self.models_dir / f"{name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.info(f"Saved {name} to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to save {name}: {e}")

    def _load_models(self):
        """Load trained models from disk"""
        model_files = {
            'tfidf_vectorizer': 'tfidf_vectorizer.pkl',
            'content_similarity_matrix': 'content_similarity_matrix.pkl',
            'collaborative_model': 'collaborative_model.pkl',
            'user_embeddings': 'user_embeddings.pkl',
            'item_embeddings': 'item_embeddings.pkl',
            'interaction_matrix': 'interaction_matrix.pkl',
            'project_features': 'project_features.pkl',
            'temporal_patterns': 'temporal_patterns.pkl',
            'config': 'config.pkl'
        }

        for attr_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))
                    logger.info(f"Loaded {attr_name} from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {attr_name}: {e}")

    def _get_model_version(self) -> str:
        """Generate model version based on data and config"""
        config_str = json.dumps(self.config, sort_keys=True)
        data_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        return f"v{timestamp}_{data_hash}"

    def _get_content_similar_projects(self, project_id: str, num_similar: int) -> List[Dict]:
        """Get content-based similar projects"""
        if project_id not in self.project_features:
            return []

        project_idx = self.project_features[project_id]['similarity_index']
        similarities = self.content_similarity_matrix[project_idx]

        # Get top similar projects (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]

        similar_projects = []
        project_ids = list(self.project_features.keys())

        for idx in similar_indices:
            similar_project_id = project_ids[idx]
            similar_projects.append({
                'project_id': similar_project_id,
                'similarity_score': similarities[idx],
                'metadata': self.project_features[similar_project_id].get('metadata', {})
            })

        return similar_projects

    def _generate_action_suggestions(self, predicted_action: str, context: Dict) -> List[str]:
        """Generate actionable suggestions based on predicted action"""
        suggestions = {
            'search': [
                'Try searching for a specific project number',
                'Browse projects by category',
                'Look for recently accessed projects'
            ],
            'open': [
                'Open the most relevant project folder',
                'Access project documents directly',
                'Navigate to System Designs folder'
            ],
            'explore': [
                'Discover new projects in your area',
                'Check trending projects',
                'Review project recommendations'
            ]
        }

        return suggestions.get(predicted_action, ['Continue with your current task'])


# Usage example and testing
if __name__ == "__main__":
    # Initialize recommendation engine
    engine = RecommendationEngine()

    # Example training data (would come from actual training_data files)
    sample_training_data = [
        {
            'project_folder': '17741 - QPS MTR RM Upgrades',
            'document_name': 'Project Handover Document.pdf',
            'extracted_info': {'type': 'handover', 'priority': 'high'}
        },
        {
            'project_folder': '17742 - Conference Room Setup',
            'document_name': 'System Design.pdf',
            'extracted_info': {'type': 'design', 'category': 'audiovisual'}
        }
    ]

    # Example user history
    sample_history = [
        {
            'user_id': 'user1',
            'project_code': '17741',
            'action': 'open',
            'timestamp': datetime.utcnow() - timedelta(hours=2)
        },
        {
            'user_id': 'user1',
            'project_code': '17742',
            'action': 'search',
            'timestamp': datetime.utcnow() - timedelta(hours=1)
        }
    ]

    # Train models
    print("Training recommendation models...")
    training_metrics = engine.train_models(sample_training_data, sample_history)
    print(f"Training completed: {training_metrics}")

    # Get recommendations
    print("\nGetting recommendations for user1...")
    recommendations = engine.get_recommendations('user1', num_recommendations=5)
    for rec in recommendations:
        print(f"  {rec['project_id']}: {rec['score']:.3f} - {rec['explanation']}")

    # Predict next action
    print("\nPredicting next action...")
    prediction = engine.predict_next_action('user1', {'current_time': datetime.utcnow()})
    print(f"  Predicted action: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})")

    # Get analytics
    print("\nAnalytics insights...")
    insights = engine.get_analytics_insights()
    print(f"  Active model status: {insights['model_status']}")
    print(f"  User engagement: {insights['user_engagement']}")