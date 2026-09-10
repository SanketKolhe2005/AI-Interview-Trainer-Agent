"""
prompts/__init__.py
===================
Prompt template loader for the AI Interview Trainer Agent.

Provides a single ``load_prompt(name)`` function that reads a prompt
template file from the prompts/ directory and returns it as a string.
Template variables (``{variable_name}``) are filled in by the caller
using Python's ``str.format_map()`` or ``str.format()``.

Usage
-----
    from prompts import load_prompt

    template = load_prompt("question_generation")
    filled   = template.format(
        role="Software Engineer",
        experience="Mid-level (3-6 years)",
        skills="Python, Django, PostgreSQL",
        interview_type="Technical",
        num_questions=5,
        difficulty="intermediate",
        context="<retrieved RAG context here>",
    )
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name.

    Parameters
    ----------
    name : str
        The filename stem (without ``.txt``) of the prompt template.
        Example: ``"question_generation"`` loads ``prompts/question_generation.txt``.

    Returns
    -------
    str
        The full text of the prompt template, with ``{variable}`` placeholders
        ready for ``.format()`` substitution.

    Raises
    ------
    FileNotFoundError
        If the requested prompt file does not exist in the prompts/ directory.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {path}\n"
            f"Available templates: {[p.stem for p in _PROMPTS_DIR.glob('*.txt')]}"
        )
    content = path.read_text(encoding="utf-8")
    logger.debug("Loaded prompt template: %s (%d chars)", name, len(content))
    return content
