"""
AI Client Integration for Project QuickNav

This module provides AI-powered assistance for project navigation using LiteLLM
to support multiple AI providers (OpenAI, Anthropic, Azure, Local models, etc.).

Features:
- Multi-provider AI model support via LiteLLM
- Tool use for project navigation and document search
- Natural language query processing
- Document analysis and summarization
- Intelligent project suggestions
- Conversation memory management
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import completion, acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    logger.warning("LiteLLM not available. AI features will be disabled.")
    LITELLM_AVAILABLE = False


class AIToolFunction:
    """Represents an AI tool function that can be called by the AI."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any],
                 function: Callable, async_function: Optional[Callable] = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.async_function = async_function

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    async def call(self, **kwargs) -> Any:
        """Call the function (async if available, otherwise sync)."""
        try:
            if self.async_function:
                return await self.async_function(**kwargs)
            else:
                return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Error calling tool function {self.name}: {e}")
            return {"error": str(e)}


class ConversationMemory:
    """Manages conversation memory and context."""

    def __init__(self, max_messages: int = 50):
        self.messages = []
        self.max_messages = max_messages
        self.context = {}

    def add_message(self, role: str, content: str, tool_calls: Optional[List] = None,
                   tool_call_id: Optional[str] = None):
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        if tool_call_id:
            message["tool_call_id"] = tool_call_id

        self.messages.append(message)

        # Trim messages if too many
        if len(self.messages) > self.max_messages:
            # Keep system message and trim from the middle
            system_messages = [msg for msg in self.messages if msg["role"] == "system"]
            other_messages = [msg for msg in self.messages if msg["role"] != "system"]

            # Keep recent messages
            recent_messages = other_messages[-self.max_messages + len(system_messages):]
            self.messages = system_messages + recent_messages

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation messages in LiteLLM format."""
        return [
            {
                "role": msg["role"],
                "content": msg["content"],
                **({k: v for k, v in msg.items()
                   if k in ["tool_calls", "tool_call_id"]})
            }
            for msg in self.messages
        ]

    def set_context(self, key: str, value: Any):
        """Set context information."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information."""
        return self.context.get(key, default)

    def clear(self):
        """Clear conversation history."""
        self.messages = []
        self.context = {}


class AIClient:
    """AI client for project navigation assistance."""

    def __init__(self, controller=None, settings=None):
        self.controller = controller
        self.settings = settings
        self.memory = ConversationMemory()
        self.tools = {}
        self.enabled = LITELLM_AVAILABLE

        if not self.enabled:
            logger.warning("AI client disabled: LiteLLM not available")
            return

        # Configure LiteLLM
        self._configure_litellm()

        # Initialize tools
        self._register_tools()

        # Set up system message
        self._setup_system_message()

    def _configure_litellm(self):
        """Configure LiteLLM settings."""
        if not self.enabled:
            return

        # Configure logging
        litellm.set_verbose = False

        # Set API keys from environment or settings
        if self.settings:
            api_keys = self.settings.get("ai.api_keys", {})
            for provider, key in api_keys.items():
                os.environ[f"{provider.upper()}_API_KEY"] = key

        # Configure default model
        self.default_model = self._get_default_model()

    def _get_default_model(self) -> str:
        """Get the default AI model to use."""
        if self.settings:
            return self.settings.get("ai.default_model", "gpt-3.5-turbo")

        # Fallback order: OpenAI -> Anthropic -> Local models
        if os.getenv("OPENAI_API_KEY"):
            return "gpt-3.5-turbo"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "claude-3-haiku-20240307"
        else:
            return "ollama/llama2"  # Local model fallback

    def _setup_system_message(self):
        """Set up the system message for the AI."""
        system_message = """You are QuickNav AI, an intelligent assistant for project navigation and document management.

You help users with:
- Finding project folders by number or name
- Locating specific documents within projects
- Analyzing project structures and contents
- Providing intelligent suggestions for project navigation
- Answering questions about project organization

Available Tools:
- search_projects: Find project folders by number or search term
- find_documents: Locate specific documents within projects
- analyze_project: Get detailed information about a project
- list_project_structure: Show the folder structure of a project
- get_recent_projects: Show recently accessed projects

You have access to a comprehensive project database with thousands of projects.
Always provide helpful, accurate information and use the available tools when needed.
If you can't find something, suggest alternative search approaches.

Be concise but helpful in your responses. When using tools, explain what you're doing and why."""

        self.memory.add_message("system", system_message)

    def _register_tools(self):
        """Register available tools for the AI."""
        # Project search tool
        self.register_tool(
            name="search_projects",
            description="Search for project folders by project number or name",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Project number (5 digits) or search term"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
            function=self._search_projects
        )

        # Document search tool
        self.register_tool(
            name="find_documents",
            description="Find specific documents within a project",
            parameters={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project number or name"
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Document type",
                        "enum": ["lld", "hld", "change_order", "sales_po", "floor_plans",
                                "scope", "qa_itp", "swms", "supplier_quotes", "photos"]
                    },
                    "filters": {
                        "type": "object",
                        "description": "Additional filters",
                        "properties": {
                            "room": {"type": "string"},
                            "co": {"type": "string"},
                            "include_archive": {"type": "boolean"}
                        }
                    }
                },
                "required": ["project", "doc_type"]
            },
            function=self._find_documents
        )

        # Project analysis tool
        self.register_tool(
            name="analyze_project",
            description="Get detailed information about a project including structure and documents",
            parameters={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project number or name"
                    },
                    "include_documents": {
                        "type": "boolean",
                        "description": "Include document listing",
                        "default": True
                    }
                },
                "required": ["project"]
            },
            function=self._analyze_project
        )

        # Project structure tool
        self.register_tool(
            name="list_project_structure",
            description="Show the folder structure of a project",
            parameters={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project number or name"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse",
                        "default": 3
                    }
                },
                "required": ["project"]
            },
            function=self._list_project_structure
        )

        # Recent projects tool
        self.register_tool(
            name="get_recent_projects",
            description="Get recently accessed projects",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent projects to return",
                        "default": 10
                    }
                }
            },
            function=self._get_recent_projects
        )

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any],
                     function: Callable, async_function: Optional[Callable] = None):
        """Register a new tool function."""
        tool = AIToolFunction(name, description, parameters, function, async_function)
        self.tools[name] = tool

    def _search_projects(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for projects using the controller."""
        try:
            if not self.controller:
                return {"error": "Project controller not available"}

            # Use cached results if available
            cached = self.controller.get_cached_result("ai_project_search", query, max_age_minutes=10)
            if cached:
                return cached

            # Perform search
            if re.match(r"^\d{5}$", query):
                # Project number search
                result = self.controller.navigate_to_project(query, "", debug_mode=False)
            else:
                # Name search - use the backend directly
                from . import find_project_path
                onedrive_folder = find_project_path.get_onedrive_folder()
                pfolder = find_project_path.get_project_folders(onedrive_folder)
                matches = find_project_path.search_by_name(query, pfolder)

                if matches:
                    result = {
                        "status": "SEARCH" if len(matches) > 1 else "SUCCESS",
                        "paths": matches[:limit] if len(matches) > 1 else [matches[0]],
                        "count": len(matches)
                    }
                else:
                    result = {"status": "ERROR", "error": f"No projects found for '{query}'"}

            # Cache and return result
            if result.get("status") != "ERROR":
                self.controller._cache_result("ai_project_search", query, result)

            return result

        except Exception as e:
            logger.exception("Project search failed")
            return {"error": str(e)}

    def _find_documents(self, project: str, doc_type: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Find documents in a project."""
        try:
            if not self.controller:
                return {"error": "Project controller not available"}

            filters = filters or {}

            result = self.controller.navigate_to_document(
                project_input=project,
                doc_type=doc_type,
                room_filter=filters.get("room", ""),
                co_filter=filters.get("co", ""),
                include_archive=filters.get("include_archive", False),
                choose_mode=True  # Always return multiple results for AI
            )

            return result

        except Exception as e:
            logger.exception("Document search failed")
            return {"error": str(e)}

    def _analyze_project(self, project: str, include_documents: bool = True) -> Dict[str, Any]:
        """Analyze a project and return detailed information."""
        try:
            # First find the project
            search_result = self._search_projects(project, limit=1)

            if search_result.get("status") == "ERROR":
                return search_result

            project_paths = search_result.get("paths", [])
            if not project_paths:
                return {"error": f"Project '{project}' not found"}

            project_path = project_paths[0]
            project_name = os.path.basename(project_path)

            analysis = {
                "project_name": project_name,
                "project_path": project_path,
                "folders": [],
                "documents": [],
                "statistics": {}
            }

            # Analyze folder structure
            if os.path.exists(project_path):
                try:
                    folders = []
                    doc_count = 0
                    total_size = 0

                    for item in os.listdir(project_path):
                        item_path = os.path.join(project_path, item)
                        if os.path.isdir(item_path):
                            folder_info = {
                                "name": item,
                                "path": item_path,
                                "file_count": 0
                            }

                            # Count files in folder
                            try:
                                for root, dirs, files in os.walk(item_path):
                                    folder_info["file_count"] += len(files)
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        try:
                                            total_size += os.path.getsize(file_path)
                                            doc_count += 1
                                        except:
                                            pass
                            except:
                                pass

                            folders.append(folder_info)

                    analysis["folders"] = folders
                    analysis["statistics"] = {
                        "total_folders": len(folders),
                        "total_documents": doc_count,
                        "total_size_mb": round(total_size / (1024 * 1024), 2)
                    }

                except Exception as e:
                    analysis["error"] = f"Failed to analyze project structure: {e}"

            # Find documents if requested
            if include_documents:
                try:
                    from . import find_project_path
                    documents = find_project_path.discover_documents(project_path)

                    doc_info = []
                    for doc_path in documents[:20]:  # Limit to first 20 documents
                        doc_name = os.path.basename(doc_path)
                        doc_size = 0
                        try:
                            doc_size = os.path.getsize(doc_path)
                        except:
                            pass

                        doc_info.append({
                            "name": doc_name,
                            "path": doc_path,
                            "size": doc_size,
                            "type": os.path.splitext(doc_name)[1].lower()
                        })

                    analysis["documents"] = doc_info

                except Exception as e:
                    analysis["document_error"] = f"Failed to discover documents: {e}"

            return analysis

        except Exception as e:
            logger.exception("Project analysis failed")
            return {"error": str(e)}

    def _list_project_structure(self, project: str, max_depth: int = 3) -> Dict[str, Any]:
        """List the folder structure of a project."""
        try:
            # Find the project first
            search_result = self._search_projects(project, limit=1)

            if search_result.get("status") == "ERROR":
                return search_result

            project_paths = search_result.get("paths", [])
            if not project_paths:
                return {"error": f"Project '{project}' not found"}

            project_path = project_paths[0]

            def build_structure(path: str, current_depth: int = 0) -> Dict[str, Any]:
                """Recursively build folder structure."""
                if current_depth >= max_depth:
                    return {"name": os.path.basename(path), "type": "folder", "truncated": True}

                structure = {
                    "name": os.path.basename(path),
                    "type": "folder",
                    "children": []
                }

                try:
                    items = sorted(os.listdir(path))
                    for item in items[:50]:  # Limit items to prevent huge structures
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            structure["children"].append(
                                build_structure(item_path, current_depth + 1)
                            )
                        else:
                            structure["children"].append({
                                "name": item,
                                "type": "file",
                                "size": os.path.getsize(item_path) if os.path.exists(item_path) else 0
                            })

                    if len(os.listdir(path)) > 50:
                        structure["children"].append({
                            "name": f"... and {len(os.listdir(path)) - 50} more items",
                            "type": "truncated"
                        })

                except Exception as e:
                    structure["error"] = str(e)

                return structure

            structure = build_structure(project_path)
            return {
                "project": os.path.basename(project_path),
                "structure": structure
            }

        except Exception as e:
            logger.exception("Project structure listing failed")
            return {"error": str(e)}

    def _get_recent_projects(self, limit: int = 10) -> Dict[str, Any]:
        """Get recently accessed projects."""
        try:
            if not self.controller:
                return {"error": "Project controller not available"}

            recent = self.controller.get_recent_projects(limit)

            return {
                "recent_projects": recent,
                "count": len(recent)
            }

        except Exception as e:
            logger.exception("Failed to get recent projects")
            return {"error": str(e)}

    async def chat(self, message: str, stream: bool = False) -> Union[str, Any]:
        """Send a chat message to the AI and get a response."""
        if not self.enabled:
            return "AI features are not available. Please install LiteLLM to enable AI assistance."

        try:
            # Add user message to memory
            self.memory.add_message("user", message)

            # Prepare messages for API call
            messages = self.memory.get_messages()

            # Prepare tools for function calling
            tools = [tool.to_dict() for tool in self.tools.values()]

            # Make API call
            if stream:
                return await self._stream_chat(messages, tools)
            else:
                return await self._complete_chat(messages, tools)

        except Exception as e:
            logger.exception("Chat request failed")
            error_msg = f"I encountered an error: {str(e)}"
            self.memory.add_message("assistant", error_msg)
            return error_msg

    async def _complete_chat(self, messages: List[Dict], tools: List[Dict]) -> str:
        """Complete a chat request without streaming."""
        try:
            response = await acompletion(
                model=self.default_model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0.7,
                max_tokens=1000
            )

            response_message = response.choices[0].message

            # Handle tool calls
            if response_message.tool_calls:
                # Add the assistant's response with tool calls
                self.memory.add_message(
                    "assistant",
                    response_message.content or "",
                    tool_calls=[tc.model_dump() for tc in response_message.tool_calls]
                )

                # Execute tool calls
                tool_responses = []
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    if tool_name in self.tools:
                        result = await self.tools[tool_name].call(**tool_args)
                        tool_response = json.dumps(result, indent=2)
                    else:
                        tool_response = f"Unknown tool: {tool_name}"

                    # Add tool response to memory
                    self.memory.add_message(
                        "tool",
                        tool_response,
                        tool_call_id=tool_call.id
                    )

                    tool_responses.append(tool_response)

                # Get final response after tool execution
                final_response = await acompletion(
                    model=self.default_model,
                    messages=self.memory.get_messages(),
                    temperature=0.7,
                    max_tokens=1000
                )

                final_content = final_response.choices[0].message.content
                self.memory.add_message("assistant", final_content)
                return final_content

            else:
                # No tool calls, just return the response
                content = response_message.content or ""
                self.memory.add_message("assistant", content)
                return content

        except Exception as e:
            logger.exception("Completion request failed")
            raise

    async def _stream_chat(self, messages: List[Dict], tools: List[Dict]):
        """Stream a chat response."""
        try:
            response = await acompletion(
                model=self.default_model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )

            return response

        except Exception as e:
            logger.exception("Streaming request failed")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available AI models."""
        if not self.enabled:
            return []

        models = [
            # OpenAI models
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",

            # Anthropic models
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",

            # Azure OpenAI
            "azure/gpt-4",
            "azure/gpt-35-turbo",

            # Local models
            "ollama/llama2",
            "ollama/codellama",
            "ollama/mistral",
        ]

        return models

    def set_model(self, model: str):
        """Set the AI model to use."""
        self.default_model = model
        if self.settings:
            self.settings.set("ai.default_model", model)

    def clear_conversation(self):
        """Clear the conversation history."""
        self.memory.clear()
        self._setup_system_message()

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.memory.messages

    def is_enabled(self) -> bool:
        """Check if AI features are enabled."""
        return self.enabled

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "messages_sent": len([m for m in self.memory.messages if m["role"] == "user"]),
            "responses_received": len([m for m in self.memory.messages if m["role"] == "assistant"]),
            "tools_used": len([m for m in self.memory.messages if m["role"] == "tool"]),
            "current_model": self.default_model,
            "enabled": self.enabled
        }