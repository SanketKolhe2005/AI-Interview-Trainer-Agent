"""
utils/formatter.py
==================
Converts agent output objects into display-ready markdown strings for the
Gradio UI.  All functions accept plain Python objects (strings, lists, dicts)
and return markdown-formatted strings — no UI logic lives here.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------

def format_questions(questions: List[str], title: str = "Interview Questions") -> str:
    """Render a list of questions as a numbered markdown list.

    Parameters
    ----------
    questions : list of str
    title : str

    Returns
    -------
    str
    """
    if not questions:
        return f"### {title}\n\n_No questions generated._"

    lines = [f"### {title}\n"]
    for i, q in enumerate(questions, 1):
        # Strip any existing numbering from the question text
        q_clean = q.strip().lstrip("0123456789.) ")
        lines.append(f"{i}. {q_clean}")
    return "\n".join(lines)


def format_all_questions(
    technical: List[str],
    role_specific: List[str],
    hr: List[str],
    behavioral: List[str],
) -> str:
    """Combine all four question sets into one tabbed markdown string."""
    sections = [
        format_questions(technical,    "🔧 Technical Questions"),
        format_questions(role_specific, "🎯 Role-Specific Questions"),
        format_questions(hr,            "🤝 HR Questions"),
        format_questions(behavioral,    "⭐ Behavioral Questions"),
    ]
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Answer / evaluation formatting
# ---------------------------------------------------------------------------

def format_model_answers(answers: Dict[str, str]) -> str:
    """Render a dict of {question: model_answer} as markdown.

    Parameters
    ----------
    answers : dict
        Keys are question strings; values are model answer strings.

    Returns
    -------
    str
    """
    if not answers:
        return "### 📝 Model Answers\n\n_No answers generated._"

    lines = ["### 📝 Model Answers\n"]
    for i, (question, answer) in enumerate(answers.items(), 1):
        lines.append(f"**Q{i}. {question.strip()}**\n")
        lines.append(f"{answer.strip()}\n")
        lines.append("---")
    return "\n".join(lines)


def format_evaluation(evaluation_text: str, question: str) -> str:
    """Wrap a raw evaluation string with the question context."""
    header = f"### Evaluation for:\n> {question.strip()}\n\n"
    return header + evaluation_text.strip()


# ---------------------------------------------------------------------------
# Tips and strategy formatting
# ---------------------------------------------------------------------------

def format_tips(tips_text: str) -> str:
    """Render improvement tips as a clean markdown block."""
    if not tips_text or not tips_text.strip():
        return "### 💡 Improvement Tips\n\n_No tips generated._"
    return f"### 💡 Improvement Tips\n\n{tips_text.strip()}"


def format_strategy(strategy_text: str) -> str:
    """Render preparation strategy as a clean markdown block."""
    if not strategy_text or not strategy_text.strip():
        return "### 📅 Preparation Strategy\n\n_No strategy generated._"
    return f"### 📅 7-Day Interview Preparation Strategy\n\n{strategy_text.strip()}"


def format_feedback(feedback_text: str) -> str:
    """Render full feedback report as a markdown block."""
    if not feedback_text or not feedback_text.strip():
        return "### 📊 Interview Feedback Report\n\n_No feedback generated._"
    return f"### 📊 Interview Feedback Report\n\n{feedback_text.strip()}"


# ---------------------------------------------------------------------------
# Mock interview formatting
# ---------------------------------------------------------------------------

def format_mock_turn(turn_text: str) -> str:
    """Render a single mock interview turn for the chat interface."""
    return turn_text.strip()


# ---------------------------------------------------------------------------
# InterviewKit summary formatting
# ---------------------------------------------------------------------------

def format_interview_kit_summary(kit: dict) -> str:
    """Produce a one-paragraph summary of a completed InterviewKit.

    Parameters
    ----------
    kit : dict
        The InterviewKit serialised to a dict via ``to_dict()``.

    Returns
    -------
    str
    """
    role     = kit.get("profile", {}).get("role", "the target role")
    exp      = kit.get("profile", {}).get("experience_level", "")
    n_tech   = len(kit.get("technical_questions", []))
    n_role   = len(kit.get("role_questions", []))
    n_hr     = len(kit.get("hr_questions", []))
    n_beh    = len(kit.get("behavioral_questions", []))
    total_q  = n_tech + n_role + n_hr + n_beh

    return (
        f"Your **{role}** ({exp}) interview kit is ready.  "
        f"Generated **{total_q} questions** total: "
        f"{n_tech} technical, {n_role} role-specific, "
        f"{n_hr} HR, and {n_beh} behavioral.  "
        "Navigate the tabs above to review questions, model answers, "
        "tips, and your preparation strategy."
    )
