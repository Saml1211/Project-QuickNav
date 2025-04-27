# Pull Request Template

Thank you for contributing to the QuickNav MCP Server! Please review and complete the checklist below before submitting your PR.

---

## Description

- [ ] Clearly describe what this PR changes or adds.
- [ ] Reference relevant issues (use `Fixes #123` or `Closes #456` syntax).

---

## Checklist

- [ ] The branch is based on `dev` and up to date.
- [ ] All new and existing tests pass locally.
- [ ] Code is linted with [ruff](https://docs.astral.sh/ruff/) (`ruff check .`).
- [ ] Code passes type-check with [mypy](https://mypy-lang.org/) (`mypy mcp_server/ quicknav/`).
- [ ] Documentation is updated as needed.
- [ ] Added or updated relevant tests.
- [ ] I have reviewed the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines.
- [ ] I have added screenshots if UI/API changes are included.
- [ ] All CI checks pass.

---

## Reviewer Notes

- [ ] I have reviewed the changes and approve merging into `dev`.
- [ ] I have run all local and CI tests and verified results.

---

Thank you for your contribution!