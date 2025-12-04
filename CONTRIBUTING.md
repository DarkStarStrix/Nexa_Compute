# Contributing to Nexa Compute

First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to Nexa Compute. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report.

- **Use the search tool** on GitHub to check if the issue has already been reported.
- **Check if the issue has been fixed** by trying to reproduce it using the latest `main` branch in the repository.
- **Collect information** about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Python version
  - Steps to reproduce

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

- **Use a clear and descriptive title** for the issue to identify the suggestion.
- **Provide a step-by-step description of the suggested enhancement** in as much detail as possible.
- **Explain why this enhancement would be useful** to most users.

### Pull Requests

1.  Fork the repo and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  If you've changed APIs, update the documentation.
4.  Ensure the test suite passes (`pytest tests/`).
5.  Make sure your code lints (`ruff check .`).

## Styleguides

We follow specific coding conventions to maintain a high-quality codebase.

### Python Styleguide

- **Linter**: We use `ruff` for linting.
- **Formatter**: We use `black` compatibility via `ruff` or `black` itself.
- **Type Checking**: We use `mypy` for static type checking.

See [docs/conventions/](docs/conventions/) for detailed info on:
- Naming conventions
- Data organization
- Project structure

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Monthly Pruning Process

Per the [Scaling Policy](docs/conventions/Scaling_Policy.md), NexaCompute requires monthly maintenance to prevent sprawl:

### Checklist (Run Monthly)

1. **Review `experiments/` folder**
   - Move production-ready code to appropriate modules
   - Remove abandoned or stale experiments
   - Document active experiments with purpose statements

2. **Audit abandoned scripts**
   - Check `scripts/` and `nexa_*/scripts/` for unused scripts
   - Remove scripts that are no longer referenced
   - Consolidate duplicate functionality

3. **Clean up outdated configs**
   - Review `nexa_infra/configs/` and `nexa_train/configs/`
   - Remove deprecated configuration templates
   - Archive old configs to `docs/archive/` if historically significant

4. **Consolidate utilities**
   - Review any new utility modules that may have appeared
   - Ensure utilities are in domain-specific modules (not generic `utils/`)
   - Merge duplicate utility functions

5. **Remove unused pathways**
   - Check for dead code paths
   - Remove unused imports and dependencies
   - Clean up deprecated API endpoints

6. **Update documentation**
   - Ensure module responsibility statements are current
   - Update architecture diagrams if structure changed
   - Refresh dependency graphs

### Process

1. Create a monthly pruning issue/PR
2. Run through the checklist above
3. Document what was removed and why
4. Update this document if process changes

