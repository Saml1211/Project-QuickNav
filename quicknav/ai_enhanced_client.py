"""
Enhanced AI Client for Project QuickNav

Integrates NLP, RAG, and ML pipeline capabilities with the existing LiteLLM-based
AI client to provide intelligent, context-aware project navigation assistance.

Features:
- Enhanced query understanding with NLP preprocessing
- RAG-powered contextual responses with document retrieval
- ML pipeline integration for predictive suggestions
- Conversational AI with project-aware context
- Analytics and insights generation
- Privacy-preserving local processing options
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
import os
import re
from pathlib import Path

from .ai_client import AIClient, ConversationMemory, AIToolFunction
from .ai_nlp_engine import NLPEngine, QueryRefinementEngine, QueryIntent
from .ai_rag_system import RAGSystem, SearchResult

logger = logging.getLogger(__name__)

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class ProjectContextManager:
    """Manages project-specific context for conversations."""

    def __init__(self):
        self.current_project = None
        self.recent_projects = []
        self.session_history = []
        self.user_preferences = {}
        self.project_insights = {}

    def set_current_project(self, project_info: Dict[str, Any]):
        """Set the current project context."""
        self.current_project = project_info

        # Add to recent projects
        project_id = project_info.get('project_number') or project_info.get('project_name')
        if project_id and project_id not in self.recent_projects:
            self.recent_projects.insert(0, project_id)
            self.recent_projects = self.recent_projects[:10]  # Keep last 10

    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get relevant context for a query."""
        context = {
            'current_project': self.current_project,
            'recent_projects': self.recent_projects,
            'user_preferences': self.user_preferences,
            'session_context': self._get_session_context()
        }

        # Add query-specific context
        if self.current_project:
            context['implied_project'] = self.current_project

        return context

    def _get_session_context(self) -> Dict[str, Any]:
        """Get context from current session."""
        return {
            'queries_count': len(self.session_history),
            'recent_topics': self._extract_recent_topics(),
            'interaction_patterns': self._analyze_interaction_patterns()
        }

    def _extract_recent_topics(self) -> List[str]:
        """Extract topics from recent queries."""
        topics = []
        for query in self.session_history[-5:]:  # Last 5 queries
            if 'document' in query.lower():
                topics.append('document_search')
            elif 'project' in query.lower():
                topics.append('project_navigation')
            elif 'analyze' in query.lower():
                topics.append('analysis')

        return list(set(topics))

    def _analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze user interaction patterns."""
        if len(self.session_history) < 3:
            return {}

        patterns = {
            'query_complexity': 'simple',
            'primary_focus': 'general',
            'interaction_style': 'direct'
        }

        # Analyze query complexity
        avg_length = sum(len(q.split()) for q in self.session_history) / len(self.session_history)
        if avg_length > 10:
            patterns['query_complexity'] = 'complex'
        elif avg_length > 5:
            patterns['query_complexity'] = 'moderate'

        # Determine primary focus
        focus_counts = {
            'documents': sum(1 for q in self.session_history if 'document' in q.lower()),
            'projects': sum(1 for q in self.session_history if 'project' in q.lower()),
            'analysis': sum(1 for q in self.session_history if 'analyze' in q.lower())
        }

        if focus_counts:
            patterns['primary_focus'] = max(focus_counts, key=focus_counts.get)

        return patterns


class ConversationalAnalytics:
    """Generates analytics and insights from conversations."""

    def __init__(self):
        self.conversation_stats = {}
        self.query_patterns = {}
        self.effectiveness_metrics = {}

    def analyze_conversation(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze a conversation for insights."""
        analysis = {
            'summary': self._generate_summary(messages),
            'key_topics': self._extract_key_topics(messages),
            'user_intent_patterns': self._analyze_intent_patterns(messages),
            'effectiveness_score': self._calculate_effectiveness(messages),
            'recommendations': self._generate_recommendations(messages)
        }

        return analysis

    def _generate_summary(self, messages: List[Dict]) -> str:
        """Generate a conversation summary."""
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']

        if not user_messages:
            return "No user queries in conversation."

        summary_parts = []

        # Count different types of queries
        query_types = {
            'project_search': 0,
            'document_search': 0,
            'analysis': 0,
            'help': 0
        }

        for msg in user_messages:
            content = msg.get('content', '').lower()
            if any(word in content for word in ['find project', 'locate project', 'project']):
                query_types['project_search'] += 1
            elif any(word in content for word in ['find document', 'search doc', 'lld', 'floor plan']):
                query_types['document_search'] += 1
            elif any(word in content for word in ['analyze', 'describe', 'what', 'details']):
                query_types['analysis'] += 1
            elif any(word in content for word in ['help', 'how', 'can you']):
                query_types['help'] += 1

        # Generate summary based on patterns
        total_queries = len(user_messages)
        summary_parts.append(f"Conversation with {total_queries} user queries")

        dominant_type = max(query_types, key=query_types.get)
        if query_types[dominant_type] > 0:
            summary_parts.append(f"Primarily focused on {dominant_type.replace('_', ' ')}")

        if len(assistant_messages) > 0:
            tool_usage = sum(1 for msg in assistant_messages if msg.get('tool_calls'))
            if tool_usage > 0:
                summary_parts.append(f"Used {tool_usage} tool interactions")

        return "; ".join(summary_parts)

    def _extract_key_topics(self, messages: List[Dict]) -> List[str]:
        """Extract key topics from conversation."""
        topics = set()

        for msg in messages:
            content = msg.get('content', '').lower()

            # Extract project numbers
            project_numbers = re.findall(r'\b\d{5}\b', content)
            topics.update(f"Project {num}" for num in project_numbers)

            # Extract document types
            doc_types = ['lld', 'hld', 'floor plan', 'change order', 'scope', 'photos']
            for doc_type in doc_types:
                if doc_type in content:
                    topics.add(doc_type.upper())

            # Extract project-related terms
            if 'conference room' in content:
                topics.add('Conference Room')
            if 'upgrade' in content:
                topics.add('Upgrade')
            if 'installation' in content:
                topics.add('Installation')

        return list(topics)[:10]  # Limit to top 10 topics

    def _analyze_intent_patterns(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze user intent patterns."""
        user_messages = [msg for msg in messages if msg.get('role') == 'user']

        patterns = {
            'query_progression': [],
            'specificity_trend': [],
            'topic_switches': 0
        }

        prev_topic = None
        for msg in user_messages:
            content = msg.get('content', '').lower()

            # Determine current topic
            current_topic = 'general'
            if 'project' in content:
                current_topic = 'project'
            elif 'document' in content:
                current_topic = 'document'
            elif 'analyze' in content:
                current_topic = 'analysis'

            patterns['query_progression'].append(current_topic)

            # Track topic switches
            if prev_topic and prev_topic != current_topic:
                patterns['topic_switches'] += 1

            prev_topic = current_topic

            # Measure specificity (more words + numbers = more specific)
            word_count = len(content.split())
            has_numbers = bool(re.search(r'\d', content))
            specificity = word_count + (5 if has_numbers else 0)
            patterns['specificity_trend'].append(specificity)

        return patterns

    def _calculate_effectiveness(self, messages: List[Dict]) -> float:
        """Calculate conversation effectiveness score."""
        if not messages:
            return 0.0

        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
        tool_messages = [msg for msg in messages if msg.get('role') == 'tool']

        # Base score factors
        response_ratio = len(assistant_messages) / max(len(user_messages), 1)
        tool_usage_ratio = len(tool_messages) / max(len(user_messages), 1)

        # Quality indicators
        error_count = sum(1 for msg in assistant_messages if 'error' in msg.get('content', '').lower())
        success_indicators = sum(1 for msg in assistant_messages if any(word in msg.get('content', '').lower()
                                                                       for word in ['found', 'located', 'here', 'success']))

        # Calculate score
        effectiveness = 0.0
        effectiveness += min(response_ratio, 1.0) * 0.3  # Response completeness
        effectiveness += min(tool_usage_ratio, 1.0) * 0.3  # Tool utilization
        effectiveness += min(success_indicators / max(len(user_messages), 1), 1.0) * 0.3  # Success rate
        effectiveness -= min(error_count / max(len(assistant_messages), 1), 0.5) * 0.1  # Error penalty

        return max(0.0, min(1.0, effectiveness))

    def _generate_recommendations(self, messages: List[Dict]) -> List[str]:
        """Generate recommendations for improving interactions."""
        recommendations = []

        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']

        # Analyze query quality
        short_queries = sum(1 for msg in user_messages if len(msg.get('content', '').split()) < 3)
        if short_queries > len(user_messages) * 0.5:
            recommendations.append("Consider providing more detailed queries for better results")

        # Check for repeated similar queries
        contents = [msg.get('content', '') for msg in user_messages]
        if len(set(contents)) < len(contents) * 0.8:
            recommendations.append("Try refining your queries instead of repeating similar requests")

        # Check for error patterns
        error_count = sum(1 for msg in assistant_messages if 'error' in msg.get('content', '').lower())
        if error_count > 2:
            recommendations.append("Check your input format or consider using the help function")

        # Usage patterns
        if not any('recent' in msg.get('content', '').lower() for msg in user_messages):
            recommendations.append("Try asking for recent projects to see your activity")

        return recommendations


class EnhancedAIClient(AIClient):
    """Enhanced AI client with NLP, RAG, and ML integration."""

    def __init__(self, controller=None, settings=None, training_data_path: str = "training_data"):
        # Initialize base AI client
        super().__init__(controller, settings)

        # Initialize enhanced components
        self.nlp_engine = NLPEngine(training_data_path)
        self.rag_system = RAGSystem(training_data_path)
        self.query_refinement = QueryRefinementEngine(self.nlp_engine)
        self.context_manager = ProjectContextManager()
        self.analytics = ConversationalAnalytics()

        # Enhanced configuration
        self.enhanced_features = {
            'query_preprocessing': True,
            'rag_context': True,
            'auto_refinement': True,
            'semantic_search': True,
            'analytics': True,
            'local_processing': settings.get('ai.local_processing', False) if settings else False
        }

        # Override system message for enhanced capabilities
        self._setup_enhanced_system_message()

        # Register enhanced tools
        self._register_enhanced_tools()

    def _setup_enhanced_system_message(self):
        """Set up enhanced system message with new capabilities."""
        enhanced_system_message = """You are QuickNav AI Pro, an advanced intelligent assistant for project navigation and document management with enhanced NLP and semantic search capabilities.

Enhanced Capabilities:
- Natural Language Processing for intelligent query understanding
- Semantic document search using vector embeddings
- Context-aware conversation with project memory
- Predictive suggestions based on usage patterns
- Real-time analytics and insights
- Multi-modal search across text, metadata, and file structure

Available Tools:
- search_projects: Find project folders with semantic similarity
- find_documents: Locate documents using advanced filters and similarity
- analyze_project: Deep project analysis with document insights
- list_project_structure: Hierarchical project organization
- get_recent_projects: Recent activity with usage patterns
- semantic_search: Advanced semantic similarity search
- get_suggestions: AI-powered suggestions based on context
- analyze_query: Query understanding and refinement suggestions

Key Features:
- Understand natural language queries and intent
- Provide contextual responses based on conversation history
- Suggest relevant projects and documents proactively
- Explain search results and provide insights
- Learn from user interactions to improve suggestions

You can understand complex queries like:
- "Find all LLD documents related to conference room upgrades"
- "Show me projects similar to the one I was working on yesterday"
- "What documents are missing from project 17741's design folder?"
- "Analyze the document patterns in my recent projects"

Always provide helpful, contextual responses and use your enhanced capabilities to deliver the most relevant information."""

        # Clear existing memory and add enhanced system message
        self.memory.clear()
        self.memory.add_message("system", enhanced_system_message)

    def _register_enhanced_tools(self):
        """Register enhanced AI tools."""
        # Semantic search tool
        self.register_tool(
            name="semantic_search",
            description="Perform semantic similarity search across all projects and documents",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for semantic similarity matching"
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search: 'projects', 'documents', or 'both'",
                        "enum": ["projects", "documents", "both"],
                        "default": "both"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            function=self._semantic_search
        )

        # Enhanced suggestions tool
        self.register_tool(
            name="get_suggestions",
            description="Get AI-powered suggestions based on context and user patterns",
            parameters={
                "type": "object",
                "properties": {
                    "context_type": {
                        "type": "string",
                        "description": "Type of suggestions needed",
                        "enum": ["next_actions", "related_projects", "relevant_documents", "query_refinement"]
                    },
                    "current_query": {
                        "type": "string",
                        "description": "Current user query for context"
                    }
                },
                "required": ["context_type"]
            },
            function=self._get_ai_suggestions
        )

        # Query analysis tool
        self.register_tool(
            name="analyze_query",
            description="Analyze user query for intent, entities, and provide refinement suggestions",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query to analyze"
                    },
                    "provide_refinements": {
                        "type": "boolean",
                        "description": "Whether to provide query refinement suggestions",
                        "default": True
                    }
                },
                "required": ["query"]
            },
            function=self._analyze_query
        )

        # Conversation analytics tool
        self.register_tool(
            name="get_conversation_insights",
            description="Get insights and analytics about the current conversation",
            parameters={
                "type": "object",
                "properties": {
                    "insight_type": {
                        "type": "string",
                        "description": "Type of insights to generate",
                        "enum": ["summary", "patterns", "effectiveness", "recommendations"],
                        "default": "summary"
                    }
                }
            },
            function=self._get_conversation_insights
        )

    async def _semantic_search(self, query: str, search_type: str = "both", top_k: int = 5) -> Dict[str, Any]:
        """Perform semantic similarity search."""
        try:
            results = {"query": query, "search_type": search_type, "results": []}

            if search_type in ["projects", "both"]:
                project_results = await self.rag_system.similarity_search_projects(query, top_k)
                results["results"].extend([
                    {
                        "type": "project",
                        "name": result["project_name"],
                        "similarity": result["similarity"],
                        "document_count": result["document_count"],
                        "document_types": result["document_types"]
                    }
                    for result in project_results
                ])

            if search_type in ["documents", "both"]:
                document_results = await self.rag_system.search_documents(query, top_k=top_k)
                results["results"].extend([
                    {
                        "type": "document",
                        "name": result.document_name,
                        "project": result.project_name,
                        "relevance": result.relevance_score,
                        "preview": result.content_preview,
                        "explanation": result.explanation
                    }
                    for result in document_results
                ])

            # Sort by relevance/similarity
            results["results"].sort(key=lambda x: x.get("similarity", x.get("relevance", 0)), reverse=True)
            results["results"] = results["results"][:top_k]

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"error": str(e)}

    async def _get_ai_suggestions(self, context_type: str, current_query: str = "") -> Dict[str, Any]:
        """Get AI-powered suggestions."""
        try:
            suggestions = {"type": context_type, "suggestions": []}

            context = self.context_manager.get_context_for_query(current_query)

            if context_type == "next_actions":
                suggestions["suggestions"] = await self._suggest_next_actions(context)
            elif context_type == "related_projects":
                suggestions["suggestions"] = await self._suggest_related_projects(current_query, context)
            elif context_type == "relevant_documents":
                suggestions["suggestions"] = await self._suggest_relevant_documents(current_query, context)
            elif context_type == "query_refinement":
                suggestions["suggestions"] = await self._suggest_query_refinements(current_query, context)

            return suggestions

        except Exception as e:
            logger.error(f"AI suggestions failed: {e}")
            return {"error": str(e)}

    async def _analyze_query(self, query: str, provide_refinements: bool = True) -> Dict[str, Any]:
        """Analyze user query for intent and entities."""
        try:
            # Parse query using NLP engine
            context = self.context_manager.get_context_for_query(query)
            query_intent = await self.nlp_engine.parse_query(query, context)

            # Get query insights
            insights = self.nlp_engine.get_query_insights(query)

            analysis = {
                "query": query,
                "intent": {
                    "type": query_intent.intent_type,
                    "confidence": query_intent.confidence
                },
                "entities": query_intent.entities,
                "structured_query": query_intent.structured_query,
                "insights": insights
            }

            if provide_refinements:
                analysis["refinements"] = query_intent.suggested_refinements

            # Get semantic suggestions
            semantic_suggestions = await self.nlp_engine.get_semantic_suggestions(query)
            if semantic_suggestions:
                analysis["semantic_suggestions"] = semantic_suggestions

            return analysis

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"error": str(e)}

    async def _get_conversation_insights(self, insight_type: str = "summary") -> Dict[str, Any]:
        """Get conversation insights and analytics."""
        try:
            messages = self.memory.get_messages()
            analysis = self.analytics.analyze_conversation(messages)

            if insight_type == "summary":
                return {"summary": analysis["summary"], "key_topics": analysis["key_topics"]}
            elif insight_type == "patterns":
                return {"patterns": analysis["user_intent_patterns"]}
            elif insight_type == "effectiveness":
                return {"effectiveness_score": analysis["effectiveness_score"]}
            elif insight_type == "recommendations":
                return {"recommendations": analysis["recommendations"]}
            else:
                return analysis

        except Exception as e:
            logger.error(f"Conversation insights failed: {e}")
            return {"error": str(e)}

    async def _suggest_next_actions(self, context: Dict) -> List[str]:
        """Suggest next actions based on context."""
        suggestions = []

        current_project = context.get("current_project")
        recent_projects = context.get("recent_projects", [])

        if current_project:
            project_name = current_project.get("project_name", "current project")
            suggestions.extend([
                f"Analyze {project_name} structure and documents",
                f"Find recent documents in {project_name}",
                f"Search for similar projects to {project_name}"
            ])
        elif recent_projects:
            suggestions.extend([
                f"Continue working with project {recent_projects[0]}",
                "Compare your recent projects",
                "Find documents across recent projects"
            ])
        else:
            suggestions.extend([
                "Search for a specific project by number or name",
                "Browse recent projects",
                "Find documents by type (LLD, floor plans, etc.)"
            ])

        return suggestions[:5]

    async def _suggest_related_projects(self, query: str, context: Dict) -> List[str]:
        """Suggest related projects."""
        suggestions = []

        if query:
            # Use semantic search to find related projects
            project_results = await self.rag_system.similarity_search_projects(query, top_k=3)
            suggestions.extend([
                f"Explore project: {result['project_name']}"
                for result in project_results
            ])

        recent_projects = context.get("recent_projects", [])
        if recent_projects:
            suggestions.extend([
                f"Return to recent project: {project}"
                for project in recent_projects[:2]
            ])

        return suggestions

    async def _suggest_relevant_documents(self, query: str, context: Dict) -> List[str]:
        """Suggest relevant documents."""
        suggestions = []

        if query:
            # Use RAG system to find relevant documents
            doc_results = await self.rag_system.search_documents(query, top_k=3)
            suggestions.extend([
                f"Review document: {result.document_name} in {result.project_name}"
                for result in doc_results
            ])

        current_project = context.get("current_project")
        if current_project:
            project_name = current_project.get("project_name")
            suggestions.extend([
                f"Find LLD documents in {project_name}",
                f"Check floor plans for {project_name}",
                f"Review change orders in {project_name}"
            ])

        return suggestions

    async def _suggest_query_refinements(self, query: str, context: Dict) -> List[str]:
        """Suggest query refinements."""
        try:
            refined_query = await self.query_refinement.refine_query(query, context)

            suggestions = [refined_query] if refined_query != query else []

            # Get auto-completions
            completions = self.nlp_engine.get_auto_completions(query)
            suggestions.extend(completions)

            return suggestions[:5]

        except Exception as e:
            logger.error(f"Query refinement suggestions failed: {e}")
            return []

    async def chat(self, message: str, stream: bool = False) -> Union[str, Any]:
        """Enhanced chat with preprocessing and contextual response."""
        if not self.enabled:
            return "AI features are not available. Please install LiteLLM to enable AI assistance."

        try:
            # Add to context manager
            self.context_manager.session_history.append(message)

            # Preprocess query if enabled
            if self.enhanced_features.get('query_preprocessing'):
                processed_message = await self._preprocess_query(message)
            else:
                processed_message = message

            # Get RAG context if enabled
            rag_context = ""
            if self.enhanced_features.get('rag_context'):
                rag_context = await self.rag_system.get_contextual_information(processed_message)

            # Add RAG context to conversation if relevant
            if rag_context and "No relevant context found" not in rag_context:
                context_message = f"Context from document database:\n\n{rag_context}\n\n"
                self.memory.add_message("system", context_message)

            # Process with enhanced AI client
            response = await super().chat(processed_message, stream)

            # Post-process response
            if isinstance(response, str):
                enhanced_response = self._post_process_response(response, message)
                return enhanced_response

            return response

        except Exception as e:
            logger.error(f"Enhanced chat failed: {e}")
            return await super().chat(message, stream)

    async def _preprocess_query(self, query: str) -> str:
        """Preprocess query using NLP engine."""
        try:
            # Get context for query
            context = self.context_manager.get_context_for_query(query)

            # Analyze query intent
            query_intent = await self.nlp_engine.parse_query(query, context)

            # Refine query if confidence is low
            if query_intent.confidence < 0.7 and self.enhanced_features.get('auto_refinement'):
                refined_query = await self.query_refinement.refine_query(query, context)
                logger.info(f"Refined query: '{query}' -> '{refined_query}'")
                return refined_query

            return query

        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            return query

    def _post_process_response(self, response: str, original_query: str) -> str:
        """Post-process AI response for enhancement."""
        try:
            # Add analytics if enabled
            if self.enhanced_features.get('analytics'):
                # Check if response was helpful
                if any(indicator in response.lower() for indicator in ['found', 'located', 'here are', 'success']):
                    analytics_note = "\n\n💡 *Response generated using enhanced AI with semantic search*"
                    response += analytics_note

            return response

        except Exception as e:
            logger.error(f"Response post-processing failed: {e}")
            return response

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced AI client statistics."""
        base_stats = self.get_usage_stats()

        enhanced_stats = {
            **base_stats,
            'enhanced_features': self.enhanced_features,
            'nlp_engine_loaded': self.nlp_engine is not None,
            'rag_system_stats': self.rag_system.get_statistics() if self.rag_system else {},
            'context_manager': {
                'current_project': self.context_manager.current_project is not None,
                'recent_projects_count': len(self.context_manager.recent_projects),
                'session_queries': len(self.context_manager.session_history)
            },
            'query_cache_size': len(self.nlp_engine.query_cache) if self.nlp_engine else 0
        }

        return enhanced_stats

    def enable_feature(self, feature_name: str, enabled: bool = True):
        """Enable or disable enhanced features."""
        if feature_name in self.enhanced_features:
            self.enhanced_features[feature_name] = enabled
            logger.info(f"Enhanced feature '{feature_name}' {'enabled' if enabled else 'disabled'}")

    def get_feature_status(self) -> Dict[str, bool]:
        """Get status of all enhanced features."""
        return self.enhanced_features.copy()


# Export main class
__all__ = ['EnhancedAIClient', 'ProjectContextManager', 'ConversationalAnalytics']