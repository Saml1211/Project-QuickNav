# Contributing to QuickNav MCP Server

Welcome! We're excited to have you contribute to the MCP Server powering Project QuickNav. This guide will help you onboard, contribute code, and participate in the project community.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Branching Model](#branching-model)
- [Code Style & Conventions](#code-style--conventions)
    - [Python](#python-style)
    - [AutoHotkey (AHK)](#ahk-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)
- [Support](#support)

---

## Getting Started

1. **Read the [Code of Conduct](CODE_OF_CONDUCT.md)** to foster a welcoming environment.
2. **Read the [Community Guidelines](COMMUNITY.md)** for collaboration best practices.
3. **Fork** the repository and clone your fork.
4. Create a new branch from `dev` for your feature or bugfix.

## Development Environment Setup

### Python Requirements

- Python 3.8+
- [pip](https://pip.pypa.io/)
- [ruff](https://docs.astral.sh/ruff/) (linting)
- [mypy](https://mypy-lang.org/) (type checking)
- [pytest](https://pytest.org/) (testing)

Install requirements:

```bash
pip install -r mcp_server/requirements.txt
pip install ruff mypy pytest
```

### AHK (AutoHotkey) Development

- [AutoHotkey v2](https://www.autohotkey.com/) (for scripts in `lld_navigator.ahk`)
- Recommend: [AHK Studio](https://www.ahkscript.org/boards/viewtopic.php?t=30077) for development

---

## Branching Model

- **main**: Stable releases only.
- **dev**: Ongoing development.
- **feature/\***: New features (branch from `dev`).
- **bugfix/\***: Bug fixes (branch from `dev`).
- **hotfix/\***: Critical fixes (branch from `main`).

---

## Code Style & Conventions

### Python Style

- [PEP8](https://pep8.org/) enforced via `ruff`.
- Type annotations encouraged; enforced by `mypy`.
- Run `ruff check .` and `mypy mcp_server/ quicknav/` before committing.

### AHK Style

- Use clear, descriptive variable names.
- Indent with 4 spaces.
- Comment non-obvious logic.
- Group related functions.
- Line endings: LF (`\n`).

---

## Testing

- All new features and bugfixes require tests.
- Place tests in the same directory as the module or in `tests/`.
- Run:

```bash
pytest
```

- For AHK scripts: see `AHK_TEST_PLAN.md` for manual/automated test steps.

---

## Pull Request Process

1. Ensure your branch is up-to-date with `dev`.
2. Write descriptive PR titles and summaries.
3. Complete the PR checklist (see PR template).
4. Expect at least one approving review before merge.
5. All CI checks (lint, type, tests) must pass.
6. After approval, maintainers will handle merges.

---

## Community Guidelines

- Please be respectful and collaborative.
- See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [COMMUNITY.md](COMMUNITY.md).
- For questions, open a GitHub Discussion or use listed support contacts.

---

## Support

- File issues on GitHub for bugs and feature requests.
- For general support, see [COMMUNITY.md](COMMUNITY.md).

---

Thank you for helping make the MCP Server project awesome!