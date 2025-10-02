# MCP Server Configuration

## Security Setup

This directory contains Model Context Protocol (MCP) server configurations that require API keys.

### Initial Setup

1. **Copy the template file:**
   ```bash
   cp mcp.json.template mcp.json
   ```

2. **Set environment variables:**
   Create a `.env` file in the project root or set system environment variables:
   ```bash
   export SMITHERY_API_KEY="your-smithery-api-key-here"
   export SMITHERY_PROFILE="your-smithery-profile-here"
   ```

3. **Replace placeholders in mcp.json:**
   Replace `${SMITHERY_API_KEY}` and `${SMITHERY_PROFILE}` with your actual values, or use a script to substitute them from environment variables.

### Security Notes

- **NEVER commit `mcp.json` with actual API keys**
- The actual `mcp.json` file is in `.gitignore` for security
- Rotate API keys immediately if they are exposed
- Use environment variables for production deployments

### Exposed Key Recovery

If an API key was previously exposed:

1. **Immediately rotate/revoke the exposed key** on the Smithery platform
2. **Remove from git history** using git filter tools if needed
3. **Update all instances** with the new key via environment variables