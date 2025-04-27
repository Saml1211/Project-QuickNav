# Main MCP server implementation for Project QuickNav

from mcp.server.fastmcp import FastMCP
import logging
import traceback
import functools
import sys
import json

# Instantiate the MCP server
mcp = FastMCP("Project QuickNav MCP Server")

def error_handler(func):
    """Decorator to catch and log all exceptions, returning standardized JSON-RPC error responses."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error("Exception in %s: %s", func.__name__, traceback.format_exc())
            # Standardized error response: (can extend keys as needed)
            return {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            }
    return wrapper

# Import tools and resources definitions (they register their handlers via decorators)
import mcp_server.tools
import mcp_server.resources

def main():
    """CLI entrypoint: Start the MCP server"""
    # Centralized logging config: logs to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    mcp.run()