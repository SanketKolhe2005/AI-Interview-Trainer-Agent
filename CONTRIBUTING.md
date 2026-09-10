# Contributing to AI Interview Trainer Agent

Thank you for your interest in contributing to the AI Interview Trainer Agent!

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/AI-Interview-Trainer-Agent.git
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Create a new branch for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code.
- Add module-level docstrings to all new Python files.
- Add function/method docstrings for all public functions.
- Keep functions focused — single responsibility.

### Architecture Rules
- **Do not** duplicate RAG or Granite logic inside `app.py`.
- **Do not** replace IBM Granite with another LLM.
- **Do not** bypass the RAG pipeline.
- Agent modules in `agent/` use the existing `rag.retriever` and `llm.granite_client`.
- All configuration must come from `config/settings.py` (loaded from `.env`).

### Secrets & Credentials
- **Never** commit `.env` or any file containing real API keys.
- `.env.example` must only contain placeholder values.
- Run a secrets scan before submitting a PR:
  ```bash
  grep -r "api_key" --include="*.py" .
  ```

### Adding a New Job Role
1. Create `knowledge_base/technical_questions/<role_slug>.json` with 15 questions.
2. Create `knowledge_base/role_descriptions/<role_slug>.txt`.
3. Add the role name to `settings.SUPPORTED_ROLES` in `config/settings.py`.
4. Re-run `python -m rag.ingest --rebuild`.

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New functionality has corresponding tests
- [ ] No `.env` or real credentials committed
- [ ] Code is PEP 8 compliant
- [ ] Docstrings added for new public functions
- [ ] `README.md` updated if new features or setup steps are added

## Reporting Issues

Open a GitHub Issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS

## Code of Conduct

Be respectful, constructive, and collaborative. This project follows standard
open-source community norms.
