"""
Enhanced NLP Engine for Project QuickNav

Provides intelligent query understanding, intent classification, and natural
language processing capabilities for project navigation and document search.

Features:
- Query intent classification and entity extraction
- Natural language to structured search translation
- Context-aware query refinement and auto-completion
- Semantic similarity matching for projects and documents
- Multi-modal query understanding (text, metadata, structure)
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("spaCy not available. Advanced NLP features will be disabled.")
    SPACY_AVAILABLE = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. ML-based NLP features will be disabled.")
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Advanced embedding features will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class QueryIntent:
    """Represents a parsed query intent with entities and confidence."""
    intent_type: str  # 'find_project', 'search_documents', 'analyze_project', etc.
    confidence: float
    entities: Dict[str, Any]
    query_text: str
    suggested_refinements: List[str]
    structured_query: Dict[str, Any]


@dataclass
class EntityMatch:
    """Represents an extracted entity match."""
    entity_type: str  # 'project_number', 'project_name', 'document_type', etc.
    value: str
    confidence: float
    span: Tuple[int, int]  # Character positions in original text


class QueryPatterns:
    """Predefined query patterns for intent classification."""

    INTENT_PATTERNS = {
        'find_project': [
            r'(?:find|locate|get|show|open)\s+project\s+(\d{5})',
            r'project\s+(\d{5})',
            r'(?:find|locate|show)\s+(?:project\s+)?(?:named|called)?\s*["\']([^"\']+)["\']',
            r'(?:find|locate|show)\s+([^,\.]+)\s+project',
            r'navigate\s+to\s+(?:project\s+)?(\d{5}|[^,\.]+)',
        ],
        'search_documents': [
            r'(?:find|search|locate|get|show)\s+(?:all\s+)?(\w+)\s+(?:documents?|docs?|files?)',
            r'(?:find|search|locate)\s+(\w+)\s+in\s+(?:project\s+)?(\d{5}|[^,\.]+)',
            r'(?:show|list)\s+(\w+)\s+(?:for|in)\s+(?:project\s+)?(\d{5}|[^,\.]+)',
            r'what\s+(\w+)\s+(?:documents?|files?)\s+(?:are\s+)?(?:in|for)\s+(?:project\s+)?(\d{5}|[^,\.]+)',
        ],
        'analyze_project': [
            r'(?:analyze|examine|inspect|describe)\s+(?:project\s+)?(\d{5}|[^,\.]+)',
            r'what[\'s\s]+(?:in|inside)\s+(?:project\s+)?(\d{5}|[^,\.]+)',
            r'(?:show|tell)\s+me\s+about\s+(?:project\s+)?(\d{5}|[^,\.]+)',
            r'(?:project\s+)?(\d{5}|[^,\.]+)\s+(?:details|info|information|structure)',
        ],
        'list_recent': [
            r'(?:show|list|get)\s+(?:my\s+)?recent\s+projects?',
            r'what\s+projects?\s+(?:have\s+)?(?:i\s+)?(?:recently\s+)?(?:accessed|opened|worked\s+on)',
            r'recent\s+(?:activity|projects?|work)',
        ],
        'search_help': [
            r'(?:how\s+(?:do\s+i|can\s+i|to))\s+(?:find|search|locate)',
            r'help\s+(?:with\s+)?(?:finding|searching|locating)',
            r'what\s+can\s+(?:you\s+)?(?:help|do)',
            r'(?:search\s+)?(?:help|assistance)',
        ],
        'navigate_folder': [
            r'(?:open|navigate\s+to|go\s+to)\s+([^,\.]+)\s+(?:folder|directory)',
            r'(?:show|open)\s+(?:the\s+)?([^,\.]+)\s+(?:folder|subfolder)',
        ]
    }

    DOCUMENT_TYPES = {
        'lld': ['lld', 'low level design', 'detail design', 'detailed design'],
        'hld': ['hld', 'high level design', 'system design'],
        'floor_plans': ['floor plan', 'floorplan', 'layout', 'plan', 'drawing'],
        'sales_po': ['purchase order', 'po', 'order', 'sales order'],
        'change_order': ['change order', 'co', 'variation', 'addendum'],
        'scope': ['scope', 'statement of work', 'sow', 'requirements'],
        'qa_itp': ['qa', 'itp', 'inspection', 'test plan', 'quality'],
        'swms': ['swms', 'safe work', 'safety', 'method statement'],
        'supplier_quotes': ['quote', 'quotation', 'pricing', 'supplier'],
        'photos': ['photo', 'image', 'picture', 'jpg', 'png']
    }

    PROJECT_KEYWORDS = [
        'project', 'job', 'client', 'customer', 'site', 'location',
        'installation', 'upgrade', 'maintenance', 'conference room',
        'meeting room', 'auditorium', 'classroom', 'office'
    ]


class NLPEngine:
    """Enhanced NLP engine for intelligent query processing."""

    def __init__(self, training_data_path: Optional[str] = None):
        self.training_data_path = training_data_path
        self.nlp_model = None
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.document_embeddings = {}
        self.project_embeddings = {}
        self.query_cache = {}

        # Initialize models
        self._initialize_models()

        # Load training data for context
        if training_data_path:
            self._load_training_data()

    def _initialize_models(self):
        """Initialize NLP models."""
        # Load spaCy model
        if SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy English model")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                SPACY_AVAILABLE = False

        # Load sentence transformer for embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded SentenceTransformer model")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")

        # Initialize TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

    def _load_training_data(self):
        """Load training data for context-aware processing."""
        try:
            training_files = Path(self.training_data_path).glob("*.json")

            documents = []
            projects = []

            for file_path in training_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    for item in data:
                        project_name = item.get('project_folder', '')
                        doc_name = item.get('document_name', '')
                        doc_path = item.get('document_path', '')

                        if project_name and project_name not in projects:
                            projects.append(project_name)

                        if doc_name:
                            documents.append({
                                'name': doc_name,
                                'path': doc_path,
                                'project': project_name
                            })

            # Create embeddings for projects and documents
            if self.embedding_model:
                self._create_embeddings(projects, documents)

            logger.info(f"Loaded {len(projects)} projects and {len(documents)} documents for NLP context")

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")

    def _create_embeddings(self, projects: List[str], documents: List[Dict]):
        """Create embeddings for projects and documents."""
        try:
            # Create project embeddings
            if projects:
                project_embeddings = self.embedding_model.encode(projects)
                self.project_embeddings = {
                    project: embedding for project, embedding in zip(projects, project_embeddings)
                }

            # Create document embeddings
            if documents:
                doc_texts = [f"{doc['name']} {doc['project']}" for doc in documents]
                doc_embeddings = self.embedding_model.encode(doc_texts)
                self.document_embeddings = {
                    doc['path']: {
                        'embedding': embedding,
                        'metadata': doc
                    } for doc, embedding in zip(documents, doc_embeddings)
                }

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")

    async def parse_query(self, query: str, context: Optional[Dict] = None) -> QueryIntent:
        """Parse a natural language query into structured intent."""
        query = query.strip()
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # Check cache first
        if query_hash in self.query_cache:
            return self.query_cache[query_hash]

        # Extract entities
        entities = self._extract_entities(query)

        # Classify intent
        intent_type, confidence = self._classify_intent(query, entities)

        # Generate structured query
        structured_query = self._generate_structured_query(query, intent_type, entities)

        # Generate refinement suggestions
        refinements = self._generate_refinements(query, intent_type, entities, context)

        # Create query intent
        query_intent = QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            query_text=query,
            suggested_refinements=refinements,
            structured_query=structured_query
        )

        # Cache result
        self.query_cache[query_hash] = query_intent

        return query_intent

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query text."""
        entities = {}

        # Extract project numbers (5 digits)
        project_numbers = re.findall(r'\b\d{5}\b', query)
        if project_numbers:
            entities['project_numbers'] = project_numbers

        # Extract document types
        doc_types = []
        query_lower = query.lower()
        for doc_type, keywords in QueryPatterns.DOCUMENT_TYPES.items():
            for keyword in keywords:
                if keyword in query_lower:
                    doc_types.append(doc_type)
                    break

        if doc_types:
            entities['document_types'] = doc_types

        # Extract quoted project names
        quoted_names = re.findall(r'["\']([^"\']+)["\']', query)
        if quoted_names:
            entities['project_names'] = quoted_names

        # Extract room/location references
        room_patterns = [
            r'room\s+(\w+)',
            r'level\s+(\d+)',
            r'floor\s+(\d+)',
            r'(?:conference|meeting)\s+room\s+(\w+)',
        ]

        rooms = []
        for pattern in room_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            rooms.extend(matches)

        if rooms:
            entities['rooms'] = rooms

        # Use spaCy for advanced entity extraction
        if SPACY_AVAILABLE and self.nlp_model:
            doc = self.nlp_model(query)

            spacy_entities = {}
            for ent in doc.ents:
                if ent.label_ not in spacy_entities:
                    spacy_entities[ent.label_] = []
                spacy_entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0
                })

            if spacy_entities:
                entities['spacy_entities'] = spacy_entities

        return entities

    def _classify_intent(self, query: str, entities: Dict) -> Tuple[str, float]:
        """Classify the intent of the query."""
        query_lower = query.lower()
        best_intent = 'general_search'
        best_confidence = 0.0

        # Pattern-based classification
        for intent, patterns in QueryPatterns.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    confidence = 0.8  # Base confidence for pattern match

                    # Boost confidence based on entity presence
                    if intent == 'find_project' and 'project_numbers' in entities:
                        confidence += 0.15
                    elif intent == 'search_documents' and 'document_types' in entities:
                        confidence += 0.15
                    elif intent == 'analyze_project' and ('project_numbers' in entities or 'project_names' in entities):
                        confidence += 0.15

                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence

        # Keyword-based fallback
        if best_confidence < 0.5:
            keyword_scores = {
                'find_project': 0.0,
                'search_documents': 0.0,
                'analyze_project': 0.0,
                'list_recent': 0.0,
                'search_help': 0.0
            }

            # Score based on keywords
            if any(word in query_lower for word in ['find', 'locate', 'search', 'get', 'show']):
                if 'project_numbers' in entities or 'project_names' in entities:
                    keyword_scores['find_project'] += 0.6
                if 'document_types' in entities:
                    keyword_scores['search_documents'] += 0.6

            if any(word in query_lower for word in ['analyze', 'describe', 'what', 'about', 'details']):
                keyword_scores['analyze_project'] += 0.5

            if any(word in query_lower for word in ['recent', 'latest', 'last']):
                keyword_scores['list_recent'] += 0.7

            if any(word in query_lower for word in ['help', 'how', 'can you']):
                keyword_scores['search_help'] += 0.6

            best_intent = max(keyword_scores, key=keyword_scores.get)
            best_confidence = keyword_scores[best_intent]

        return best_intent, best_confidence

    def _generate_structured_query(self, query: str, intent_type: str, entities: Dict) -> Dict[str, Any]:
        """Generate a structured query from natural language."""
        structured = {
            'intent': intent_type,
            'parameters': {},
            'filters': {},
            'options': {}
        }

        if intent_type == 'find_project':
            if 'project_numbers' in entities:
                structured['parameters']['project_number'] = entities['project_numbers'][0]
            elif 'project_names' in entities:
                structured['parameters']['project_name'] = entities['project_names'][0]
            else:
                # Extract potential project name from query
                # Remove common action words
                clean_query = re.sub(r'\b(?:find|locate|show|get|project)\b', '', query, flags=re.IGNORECASE)
                clean_query = clean_query.strip()
                if clean_query:
                    structured['parameters']['search_term'] = clean_query

        elif intent_type == 'search_documents':
            if 'document_types' in entities:
                structured['parameters']['document_type'] = entities['document_types'][0]

            if 'project_numbers' in entities:
                structured['filters']['project_number'] = entities['project_numbers'][0]
            elif 'project_names' in entities:
                structured['filters']['project_name'] = entities['project_names'][0]

            if 'rooms' in entities:
                structured['filters']['room'] = entities['rooms'][0]

        elif intent_type == 'analyze_project':
            if 'project_numbers' in entities:
                structured['parameters']['project_number'] = entities['project_numbers'][0]
            elif 'project_names' in entities:
                structured['parameters']['project_name'] = entities['project_names'][0]

            structured['options']['include_documents'] = True
            structured['options']['include_structure'] = True

        elif intent_type == 'list_recent':
            structured['options']['limit'] = 10
            structured['options']['sort_by'] = 'access_time'

        return structured

    def _generate_refinements(self, query: str, intent_type: str, entities: Dict, context: Optional[Dict] = None) -> List[str]:
        """Generate query refinement suggestions."""
        refinements = []

        if intent_type == 'find_project':
            if 'project_numbers' not in entities and 'project_names' not in entities:
                refinements.append("Try including a specific project number (5 digits) or project name in quotes")
                refinements.append("Example: 'Find project 17741' or 'Find project \"Conference Room Upgrade\"'")

        elif intent_type == 'search_documents':
            if 'document_types' not in entities:
                refinements.append("Specify a document type like: LLD, floor plans, change orders, etc.")
                refinements.append("Example: 'Find LLD documents in project 17741'")

            if 'project_numbers' not in entities and 'project_names' not in entities:
                refinements.append("Include a project identifier to narrow the search")
                refinements.append("Example: 'Find floor plans in Test Project'")

        elif intent_type == 'general_search':
            refinements.append("Try being more specific about what you're looking for:")
            refinements.append("• 'Find project [number or name]' to locate a project")
            refinements.append("• 'Find [document type] in project [name]' to search documents")
            refinements.append("• 'Analyze project [name]' to get project details")
            refinements.append("• 'Show recent projects' to see recent activity")

        # Add context-aware suggestions
        if context and 'recent_projects' in context:
            recent_projects = context['recent_projects'][:3]
            if recent_projects:
                refinements.append("Recent projects you might be looking for:")
                for project in recent_projects:
                    refinements.append(f"• {project}")

        return refinements

    async def get_semantic_suggestions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get semantic suggestions based on query similarity."""
        if not self.embedding_model or not (self.project_embeddings or self.document_embeddings):
            return []

        try:
            query_embedding = self.embedding_model.encode([query])
            suggestions = []

            # Find similar projects
            if self.project_embeddings:
                project_names = list(self.project_embeddings.keys())
                project_embeddings = np.array(list(self.project_embeddings.values()))

                similarities = cosine_similarity(query_embedding, project_embeddings)[0]

                # Get top similar projects
                top_indices = np.argsort(similarities)[-limit:][::-1]

                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Minimum similarity threshold
                        suggestions.append({
                            'type': 'project',
                            'name': project_names[idx],
                            'similarity': float(similarities[idx]),
                            'suggestion': f"Find project: {project_names[idx]}"
                        })

            # Find similar documents
            if self.document_embeddings and len(suggestions) < limit:
                doc_paths = list(self.document_embeddings.keys())
                doc_embeddings = np.array([self.document_embeddings[path]['embedding'] for path in doc_paths])

                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                top_indices = np.argsort(similarities)[-(limit - len(suggestions)):][::-1]

                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        doc_meta = self.document_embeddings[doc_paths[idx]]['metadata']
                        suggestions.append({
                            'type': 'document',
                            'name': doc_meta['name'],
                            'project': doc_meta['project'],
                            'similarity': float(similarities[idx]),
                            'suggestion': f"Find document: {doc_meta['name']} in {doc_meta['project']}"
                        })

            return sorted(suggestions, key=lambda x: x['similarity'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get semantic suggestions: {e}")
            return []

    def get_auto_completions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get auto-completion suggestions for partial queries."""
        completions = []
        partial_lower = partial_query.lower()

        # Pattern-based completions
        completion_templates = [
            "Find project {number}",
            "Find project \"{name}\"",
            "Find LLD documents in project {project}",
            "Find floor plans in project {project}",
            "Analyze project {project}",
            "Show recent projects",
            "Find change orders in project {project}",
            "What documents are in project {project}?"
        ]

        # Filter templates based on partial query
        for template in completion_templates:
            template_lower = template.lower()
            if template_lower.startswith(partial_lower) or any(word in template_lower for word in partial_lower.split()):
                completions.append(template)

        # Add project-specific completions if we have project data
        if self.project_embeddings and len(partial_lower) > 2:
            matching_projects = [
                project for project in self.project_embeddings.keys()
                if partial_lower in project.lower()
            ][:3]

            for project in matching_projects:
                if len(completions) < limit:
                    completions.append(f"Find project \"{project}\"")
                if len(completions) < limit:
                    completions.append(f"Analyze project \"{project}\"")

        return completions[:limit]

    def extract_query_context(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Extract contextual information from query and conversation history."""
        context = {
            'current_query': query,
            'query_type': 'standalone',
            'references': {},
            'intent_continuity': None
        }

        # Analyze conversation history for context
        if conversation_history:
            recent_messages = conversation_history[-5:]  # Last 5 messages

            # Look for pronoun references (it, that, this, etc.)
            pronouns = re.findall(r'\b(?:it|that|this|they|them)\b', query, re.IGNORECASE)
            if pronouns:
                # Find the most recent project or document reference
                for msg in reversed(recent_messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        # Look for project mentions
                        project_refs = re.findall(r'project\s+(\d{5}|"[^"]+"|[A-Z][^.!?]*)', content, re.IGNORECASE)
                        if project_refs:
                            context['references']['implied_project'] = project_refs[0]
                            context['query_type'] = 'contextual'
                            break

            # Detect intent continuity
            if len(recent_messages) >= 2:
                last_user_msg = None
                for msg in reversed(recent_messages):
                    if msg.get('role') == 'user':
                        last_user_msg = msg.get('content', '')
                        break

                if last_user_msg:
                    last_intent = self._classify_intent(last_user_msg, {})[0]
                    current_intent = self._classify_intent(query, {})[0]

                    if last_intent == current_intent:
                        context['intent_continuity'] = last_intent

        return context

    def get_query_insights(self, query: str) -> Dict[str, Any]:
        """Get insights about query complexity and suggestions for improvement."""
        insights = {
            'complexity': 'simple',
            'completeness': 0.0,
            'suggestions': [],
            'entity_coverage': {},
            'confidence_score': 0.0
        }

        # Analyze query complexity
        word_count = len(query.split())
        entity_count = len(self._extract_entities(query))

        if word_count > 10 or entity_count > 3:
            insights['complexity'] = 'complex'
        elif word_count > 5 or entity_count > 1:
            insights['complexity'] = 'moderate'

        # Calculate completeness score
        has_action = any(word in query.lower() for word in ['find', 'show', 'get', 'search', 'locate', 'analyze'])
        has_target = any(word in query.lower() for word in ['project', 'document', 'file', 'folder'])
        has_identifier = bool(re.search(r'\d{5}|"[^"]+"|[A-Z][A-Za-z\s]+', query))

        completeness_factors = [has_action, has_target, has_identifier]
        insights['completeness'] = sum(completeness_factors) / len(completeness_factors)

        # Generate improvement suggestions
        if not has_action:
            insights['suggestions'].append("Add an action word like 'find', 'show', or 'search'")
        if not has_target:
            insights['suggestions'].append("Specify what you're looking for (project, document, etc.)")
        if not has_identifier:
            insights['suggestions'].append("Include a project number or name for better results")

        # Calculate overall confidence
        intent_confidence = self._classify_intent(query, self._extract_entities(query))[1]
        insights['confidence_score'] = (intent_confidence + insights['completeness']) / 2

        return insights


class QueryRefinementEngine:
    """Engine for refining and optimizing queries based on context and history."""

    def __init__(self, nlp_engine: NLPEngine):
        self.nlp_engine = nlp_engine
        self.refinement_history = {}

    async def refine_query(self, query: str, context: Dict[str, Any], user_feedback: Optional[Dict] = None) -> str:
        """Refine a query based on context and user feedback."""
        # Parse the original query
        query_intent = await self.nlp_engine.parse_query(query, context)

        refined_query = query

        # Apply refinements based on intent confidence
        if query_intent.confidence < 0.7:
            # Low confidence - suggest structural improvements
            if query_intent.intent_type == 'general_search':
                refined_query = self._add_structure_to_query(query, context)
            else:
                refined_query = self._enhance_low_confidence_query(query, query_intent, context)

        # Apply context-based refinements
        if context.get('references', {}).get('implied_project'):
            refined_query = self._resolve_project_references(refined_query, context)

        # Apply user feedback if provided
        if user_feedback:
            refined_query = self._apply_user_feedback(refined_query, user_feedback)

        return refined_query

    def _add_structure_to_query(self, query: str, context: Dict) -> str:
        """Add structure to unstructured queries."""
        # If query contains a number that looks like a project, make it explicit
        project_match = re.search(r'\b(\d{5})\b', query)
        if project_match:
            return f"Find project {project_match.group(1)}"

        # If query mentions document types, structure as document search
        doc_type_match = None
        for doc_type, keywords in QueryPatterns.DOCUMENT_TYPES.items():
            for keyword in keywords:
                if keyword in query.lower():
                    doc_type_match = keyword
                    break
            if doc_type_match:
                break

        if doc_type_match:
            return f"Find {doc_type_match} documents"

        # Default: treat as project search
        return f"Find project \"{query}\""

    def _enhance_low_confidence_query(self, query: str, intent: QueryIntent, context: Dict) -> str:
        """Enhance queries with low confidence scores."""
        enhanced = query

        if intent.intent_type == 'find_project':
            # Add missing project identifier
            if not intent.entities.get('project_numbers') and not intent.entities.get('project_names'):
                # Extract potential project name from query
                clean_query = re.sub(r'\b(?:find|locate|show|get|project)\b', '', query, flags=re.IGNORECASE).strip()
                if clean_query:
                    enhanced = f"Find project \"{clean_query}\""

        elif intent.intent_type == 'search_documents':
            # Add missing document type or project context
            if not intent.entities.get('document_types'):
                enhanced = f"Find documents {query}"

            if not intent.entities.get('project_numbers') and not intent.entities.get('project_names'):
                if context.get('references', {}).get('implied_project'):
                    enhanced += f" in project {context['references']['implied_project']}"

        return enhanced

    def _resolve_project_references(self, query: str, context: Dict) -> str:
        """Resolve pronoun references to specific projects."""
        implied_project = context.get('references', {}).get('implied_project')
        if not implied_project:
            return query

        # Replace pronouns with explicit project reference
        pronouns = ['it', 'that', 'this', 'them', 'they']
        for pronoun in pronouns:
            pattern = r'\b' + re.escape(pronoun) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                query = re.sub(pattern, f"project {implied_project}", query, flags=re.IGNORECASE)
                break

        return query

    def _apply_user_feedback(self, query: str, feedback: Dict) -> str:
        """Apply user feedback to refine the query."""
        if feedback.get('action') == 'add_filter':
            filter_type = feedback.get('filter_type')
            filter_value = feedback.get('filter_value')
            if filter_type and filter_value:
                if filter_type == 'room':
                    query += f" in room {filter_value}"
                elif filter_type == 'document_type':
                    query = f"Find {filter_value} {query}"

        elif feedback.get('action') == 'narrow_scope':
            scope = feedback.get('scope')
            if scope:
                query += f" {scope}"

        return query


# Export main classes
__all__ = [
    'NLPEngine',
    'QueryRefinementEngine',
    'QueryIntent',
    'EntityMatch',
    'QueryPatterns'
]