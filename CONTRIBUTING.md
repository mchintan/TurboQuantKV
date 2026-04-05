# Contributing to TurboQuantKV

Thanks for your interest in contributing! This document outlines how to get started.

## Getting Started

1. Fork the repository and clone your fork
2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Run the tests to verify your setup:
   ```bash
   pytest tests/ -v
   ```

## Making Changes

1. Create a branch from `main`:
   ```bash
   git checkout -b your-feature-name
   ```
2. Make your changes, keeping commits focused and atomic
3. Add or update tests for any new functionality
4. Run the full test suite before submitting:
   ```bash
   pytest tests/ -v
   ```

## Submitting a Pull Request

1. Push your branch to your fork
2. Open a PR against `main` on this repository
3. Fill in the PR template with a clear description of your changes
4. Ensure CI passes (pytest on Python 3.10, 3.11, 3.12)

## Code Style

- Follow the existing code conventions in the file you're editing
- Use type annotations for function signatures
- Keep docstrings consistent with the existing style (Google-style)
- Prefer minimal, focused changes over large refactors

## Reporting Issues

- Use GitHub Issues to report bugs or request features
- For bugs, include: Python version, PyTorch version, steps to reproduce, and the full error traceback
- For feature requests, describe the use case and expected behavior

## Running Benchmarks

To run the benchmark suite and regenerate graphs:

```bash
pip install -e ".[bench]"
python benchmarks/run_benchmarks.py
python benchmarks/generate_graphs.py
```

## Questions?

Open a GitHub Issue with the "question" label.
