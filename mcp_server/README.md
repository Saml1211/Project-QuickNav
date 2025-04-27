# Project QuickNav MCP Server

[![Lint](https://img.shields.io/github/actions/workflow/status/yourorg/QuickNav/lint.yml?label=lint)](../../actions/workflows/lint.yml)
[![Type Check](https://img.shields.io/github/actions/workflow/status/yourorg/QuickNav/typecheck.yml?label=mypy)](../../actions/workflows/typecheck.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/yourorg/QuickNav/test.yml?label=tests)](../../actions/workflows/test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/yourorg/QuickNav?label=coverage)](https://codecov.io/gh/yourorg/QuickNav)
[![Build](https://img.shields.io/github/actions/workflow/status/yourorg/QuickNav/build.yml?label=build)](../../actions/workflows/build.yml)

> **Note:** Replace `yourorg/QuickNav` in badge URLs with your actual GitHub org/repo.

---

This directory implements a custom Model Context Protocol (MCP) server for Project QuickNav using the [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk).

---

## Features

- Exposes Project QuickNav navigation features via MCP tools
- Provides resource access for project structure/context
- Standard MCP-compliant API and tool schemas
- CLI-based launch for easy integration
- Integration with `find_project_path.py`
- User preferences/history: persistent, cross-platform tracking via new MCP tools

---

## Requirements

- Python 3.8+
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
- Dependencies in `requirements.txt`

---

## Installation

```sh
cd mcp_server
pip install -r requirements.txt
```

---

## Starting the Server

```sh
python -m mcp_server
```

---

## Usage

- The server exposes MCP-compatible API endpoints and tools.
- Key tools:
    - **Project navigation**: wraps `find_project_path.py`
    - **User preferences/history tools**:
        - `get_user_preferences`: Get all stored user preferences (`dict`)
        - `set_user_preferences`: Set all preferences (replaces `dict`)
        - `clear_user_preferences`: Clear all preferences
        - `get_user_history`: Retrieve recent usage history (list)
        - `add_user_history_entry`: Add entry to usage history
        - `clear_user_history`: Clear all history
        - `recommend_projects`: Suggest recent/frequent project codes
        - `get_quicknav_usage_diagnostics`: Usage stats, error counts
- All preferences/history stored in `~/.quicknav_userdata.json` (cross-platform)
- History updated automatically on project navigation
- Resource: Project structure/context

---

## Development

- Main server logic: [`server.py`](server.py)
- Tool schemas/implementations: [`tools.py`](tools.py)
- Resource definitions: [`resources.py`](resources.py)

---

## Testing

Basic tests:

```sh
python -m mcp_server.test_server
```

---

## Documentation & API Reference

### Tool & Resource API

All tools/resources are compliant with the [MCP Tool Schema](https://github.com/modelcontextprotocol/spec).

#### Example Tool Schema

```json
{
  "name": "get_user_preferences",
  "parameters": {},
  "returns": { "preferences": "dict" },
  "description": "Get all stored user preferences."
}
```

**Sample Request (MCP tool call):**
```json
{
  "tool": "get_user_preferences",
  "params": {}
}
```

**Sample Response:**
```json
{
  "result": {
    "preferences": {
      "recentDirectories": ["~/Projects/FooBar", "~/Projects/Baz"],
      "theme": "dark"
    }
  }
}
```

See [`tools.py`](tools.py) and [`user_data.py`](user_data.py) for full schemas, parameters, and examples.

#### Resource API

- `project_structure` resource: Returns directory tree and context.
- Example:
    - **Request:** `GET /resource/project_structure`
    - **Response:** see [`resources.py`](resources.py) for format.

---

### Screenshots

> **Add working screenshots here!**
>
> - Place images in `docs/screenshots/` and link below.
> - To update: Replace placeholder links with real screenshot file paths.

**Examples:**

![Server CLI Output](../docs/screenshots/server_cli.png)
![Tool API Response](../docs/screenshots/api_response.png)

---

### Support & Community

- **Contribution Guide:** [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Code of Conduct:** [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
- **Community Guidelines:** [COMMUNITY.md](../COMMUNITY.md)
- **Bug Reports & Requests:** [GitHub Issues](../../issues)
- **General Support/Chat:** [GitHub Discussions](../../discussions)
- **Email:** [quicknav-maintainers@protonmail.com](mailto:quicknav-maintainers@protonmail.com)

---

## Project Status Badges

- **Lint:** Shows Ruff linting status ([workflow](../../actions/workflows/lint.yml))
- **Type Check:** Shows mypy type-check status ([workflow](../../actions/workflows/typecheck.yml))
- **Tests:** Shows test run status ([workflow](../../actions/workflows/test.yml))
- **Coverage:** Shows code coverage via Codecov
- **Build:** Shows build status ([workflow](../../actions/workflows/build.yml))

---

## CI/CD

This repository uses [GitHub Actions](https://docs.github.com/en/actions) for:

- Linting: [ruff](https://docs.astral.sh/ruff/)
- Type checking: [mypy](https://mypy-lang.org/)
- Automated tests: [pytest](https://pytest.org/)
- Coverage reporting: [Codecov](https://codecov.io/) (if enabled)

Workflows are defined in `.github/workflows/`. All PRs and pushes to `dev`/`main` are checked.

---

## Further Documentation

- Inline code documentation in all modules.
- See [DEVELOPER.md](../DEVELOPER.md) and [TESTING.md](../TESTING.md) for advanced and testing topics.
- [INSTALL.md](../INSTALL.md) for install details.
- [RELEASE.md](../RELEASE.md) for release notes.

---