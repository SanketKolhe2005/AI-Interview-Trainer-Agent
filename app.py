"""
app.py
======
Gradio UI entry point for the AI Interview Trainer Agent.

Launch
------
    python app.py

The app opens at http://localhost:7860 by default.

Architecture
------------
All AI logic lives in the agent/ package.  This file only:
  - Defines the Gradio layout and event handlers.
  - Translates UI inputs into agent calls.
  - Formats agent outputs for display.
  - Manages per-session state (MockInterviewer instance).

No Granite, RAG, or LLM logic is duplicated here.
"""

import logging
import sys
from pathlib import Path

import gradio as gr

# Ensure project root is importable when launched from any working directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import settings
from agent.orchestrator import InterviewOrchestrator, InterviewKit
from utils.profile_parser import ProfileParser
from utils.formatter import (
    format_questions,
    format_all_questions,
    format_model_answers,
    format_tips,
    format_strategy,
    format_feedback,
    format_interview_kit_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton orchestrator — shared across all UI sessions.
# LLM client is lazy-loaded on first generate() call.
# ---------------------------------------------------------------------------
_orchestrator = InterviewOrchestrator()

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

_CREDENTIAL_WARNING = (
    "⚠️ **IBM watsonx.ai credentials not configured.**\n\n"
    "Copy `.env.example` to `.env` and fill in your "
    "`WATSONX_API_KEY` and `WATSONX_PROJECT_ID` before generating.\n\n"
    "See the README → Setup & Installation for instructions."
)

_SUPPORTED_ROLES      = settings.SUPPORTED_ROLES
_EXPERIENCE_LEVELS    = settings.EXPERIENCE_LEVELS
_INTERVIEW_TYPES      = ["Mixed", "Technical", "HR", "Behavioral", "Role-Specific"]
_DIFFICULTY_LEVELS    = ["Auto (based on experience)", "Basic", "Intermediate", "Advanced"]
_QUESTION_COUNTS      = [3, 5, 7, 10]


def _credentials_ok() -> bool:
    return bool(settings.WATSONX_API_KEY and settings.WATSONX_PROJECT_ID)


def _safe_generate(fn, *args, **kwargs):
    """Call *fn* and return (result, error_markdown).  Never raises."""
    try:
        return fn(*args, **kwargs), None
    except ValueError as exc:
        return None, f"⚠️ **Configuration error:** {exc}"
    except RuntimeError as exc:
        return None, f"❌ **Generation error:** {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during generation.")
        return None, f"❌ **Unexpected error:** {exc}"


# ---------------------------------------------------------------------------
# Offline fallback for Mock Interview
# ---------------------------------------------------------------------------
class _LocalTurn:
    def __init__(self, question, user_answer, score, feedback):
        self.question = question
        self.user_answer = user_answer
        self.score = score
        self.feedback = feedback


class LocalMockInterviewer:
    """Deterministic offline mock interviewer used when IBM returns HTTP 429."""

    def __init__(self, profile, total_questions=8, interview_type="Mixed"):
        self.profile = profile
        self.total_questions = max(1, int(total_questions))
        self.interview_type = interview_type or "Mixed"
        self.current_question_number = 0
        self.history = []
        self.questions = self._build_questions()

    def _build_questions(self):
        role = getattr(self.profile, "role", "Software Engineer") or "Software Engineer"
        skills = getattr(self.profile, "skills", []) or []
        skill_text = ", ".join(skills[:5]) if skills else "Python and SQL"
        banks = {
            "Technical": [
                "Explain the difference between a Python list and tuple and when you would use each.",
                "What is a SQL JOIN? Explain INNER JOIN and LEFT JOIN with an example.",
                "What is a REST API and what are common HTTP methods used by it?",
                "How would you find and fix a bug in a Python program that gives the wrong output?",
                "Explain time complexity using a simple example such as searching or sorting.",
                f"How would you use {skill_text} in a software engineering project?",
                "What is the difference between authentication and authorization?",
                "How would you validate input in a backend API?",
            ],
            "Role-Specific": [
                f"What do you understand about the responsibilities of a {role}?",
                "How would you approach a new software task when the requirements are not completely clear?",
                "How would you design a small feature from requirement to testing and deployment?",
                "How do you make sure your code is maintainable and easy for teammates to understand?",
                "How would you troubleshoot a REST API that suddenly starts returning errors?",
                "How do you use Git when working with other developers on the same project?",
                "How would you prioritize correctness, performance, and delivery time on a project?",
                "What would you do if you were assigned a technology you had never used before?",
            ],
            "HR": [
                f"Why are you interested in a {role} role?",
                "What are your key strengths as a fresher?",
                "What are you looking for in your first professional role?",
                "What is one area you are currently improving?",
                "Why should we hire you?",
                "Where do you see yourself developing over the next few years?",
                "How do you handle pressure or an unfamiliar situation?",
                "What motivates you to keep learning?",
            ],
            "Behavioral": [
                "Tell me about a time you learned a new technical concept quickly.",
                "Tell me about a challenging bug or problem you solved.",
                "Describe a time you received constructive feedback and how you responded.",
                "Tell me about a time you had multiple assignments or deadlines. How did you prioritize them?",
                "Describe a disagreement with a teammate and how you handled it.",
                "Tell me about a time you took ownership of a task or project.",
                "Describe a project setback and what you learned from it.",
                "Tell me about a time you had to explain a technical idea to someone else.",
            ],
        }
        if self.interview_type == "Mixed":
            order = ["Technical", "Role-Specific", "HR", "Behavioral"]
            questions = []
            i = 0
            while len(questions) < self.total_questions:
                bank = banks[order[i % len(order)]]
                questions.append(bank[(i // len(order)) % len(bank)])
                i += 1
            return questions
        bank = banks.get(self.interview_type, banks["Technical"])
        return [bank[i % len(bank)] for i in range(self.total_questions)]

    def start_session(self):
        self.current_question_number = 0
        return self.questions[0]

    def _score_answer(self, question, answer):
        text = (answer or "").strip()
        low = text.lower()
        words = len(text.split())
        score = 3.0 if words < 5 else 5.0 if words < 15 else 6.5 if words < 30 else 7.5
        keywords = [
            "because", "example", "approach", "test", "debug", "validate",
            "python", "sql", "api", "git", "project", "team", "learn",
            "result", "impact", "star", "situation", "task", "action",
        ]
        hits = sum(1 for k in keywords if k in low)
        score += min(2.0, hits * 0.4)
        if self.interview_type == "Behavioral" and words >= 25:
            score += 0.5
        score = min(10.0, round(score, 1))
        if score >= 8.5:
            feedback = "Strong answer. It is clear, relevant, and supported with useful details or examples. Keep this level of structure in the real interview."
        elif score >= 7:
            feedback = "Good answer. Add one concrete example, explain your reasoning more clearly, and state the result or impact to make it stronger."
        elif score >= 5:
            feedback = "Partially complete answer. Give a clearer structure, include the key technical or behavioral points, and support the answer with an example."
        else:
            feedback = "The answer is too brief. Explain your approach step by step and include a specific example or result."
        return score, feedback

    def submit_answer(self, user_answer):
        q = self.questions[self.current_question_number]
        score, feedback = self._score_answer(q, user_answer)
        self.history.append(_LocalTurn(q, user_answer, score, feedback))
        answered_number = self.current_question_number
        is_complete = answered_number + 1 >= self.total_questions
        if not is_complete:
            self.current_question_number += 1
        return {
            "is_complete": is_complete,
            "feedback": feedback,
            "question_number": answered_number,
        }

    def get_average_score(self):
        if not self.history:
            return 0
        return round(sum(t.score for t in self.history) / len(self.history), 1)

    def get_history_text(self):
        parts = []
        for i, t in enumerate(self.history, 1):
            parts.append(
                f"Question {i}: {t.question}\n"
                f"Candidate answer: {t.user_answer}\n"
                f"Score: {t.score}/10\n"
                f"Feedback: {t.feedback}"
            )
        return "\n\n".join(parts)


def _local_model_answer(question, profile):
    """Deterministic model answer used when IBM Granite quota is unavailable."""
    q = (question or "").lower()
    role = getattr(profile, "role", "Software Engineer") or "Software Engineer"
    skills = ", ".join(getattr(profile, "skills", [])[:5]) or "Python and SQL"

    if "list and tuple" in q:
        return "A Python list is mutable, while a tuple is immutable. I use a list when the collection may change and a tuple when the data should remain fixed. For example, I would use a list for items being added during processing and a tuple for a fixed configuration."
    if "sql join" in q:
        return "An INNER JOIN returns rows that have matching values in both tables. A LEFT JOIN returns all rows from the left table and matching rows from the right table, with NULLs when there is no match."
    if "rest api" in q or "http methods" in q:
        return "A REST API exposes resources through HTTP. Common methods are GET for reading, POST for creating, PUT or PATCH for updating, and DELETE for removing. I would also use suitable status codes such as 200, 201, 400, 401, 404, and 500."
    if "bug" in q or "debug" in q:
        return "I would reproduce the issue first, isolate the failing part, inspect logs or use a debugger, identify the root cause, fix it, and then run tests to make sure the fix does not introduce a regression."
    if "time complexity" in q:
        return "Time complexity describes how an algorithm's running time grows with input size. For example, linear search is O(n) because it may inspect every element, while binary search on sorted data is O(log n)."
    if "authentication and authorization" in q:
        return "Authentication verifies who the user is, while authorization determines what that user is allowed to access. For example, logging in is authentication and checking whether the user can access an admin endpoint is authorization."
    if "input" in q and "api" in q:
        return "I would validate required fields, data types, formats, ranges, and business rules before processing the request. Invalid input should return a clear 4xx response, and validated values should be handled safely to avoid security and data-quality problems."
    if "responsibilities" in q:
        return f"As a {role}, I would understand requirements, write and test maintainable code, debug issues, work with teammates, use Git effectively, document important decisions, and continuously improve my technical skills."
    if "requirements are not completely clear" in q or "new software task" in q:
        return "I would first clarify the expected outcome, inputs, constraints, and acceptance criteria. Then I would break the task into smaller steps, confirm assumptions with the relevant person, implement incrementally, test the result, and communicate progress."
    if "design a small feature" in q:
        return "I would start from the requirement and acceptance criteria, design a simple solution, implement it in small pieces, write tests, review the code, and validate the feature before deployment."
    if "maintainable" in q:
        return "I keep code modular, use clear names, avoid unnecessary duplication, follow project conventions, add tests, and document non-obvious decisions. I also use code review to get feedback from teammates."
    if "prioritize correctness" in q:
        return "I would first ensure the solution is correct and safe. Then I would consider performance based on actual requirements and prioritize delivery according to business impact and deadlines."
    if "technology you had never used" in q or "never used before" in q:
        return f"I would understand the requirement first, learn the core concepts from reliable documentation, build a small proof of concept, and then apply it to the task. I am comfortable learning new tools and connecting them with skills such as {skills}."
    if "why are you interested" in q:
        return f"I am interested in a {role} role because I enjoy programming, problem solving, and building practical solutions. My academic and project work has given me experience with {skills}, and I want to apply those skills in a professional environment while continuing to learn."
    if "key strengths" in q or "strengths" in q:
        return "My key strengths are problem solving, willingness to learn, consistency, and the ability to break a problem into smaller steps. I also try to communicate clearly and take ownership of my work."
    if "first professional role" in q:
        return "I am looking for an opportunity where I can contribute to real projects, learn from experienced teammates, strengthen my engineering fundamentals, and gradually take ownership of larger tasks."
    if "area" in q and "improving" in q:
        return "I am working on improving how I explain technical ideas concisely. I practice by writing short explanations, discussing solutions with others, and focusing on the key point before adding details."
    if "why should we hire" in q:
        return "You should hire me because I bring strong learning ability, problem-solving skills, and hands-on project experience. I may be a fresher, but I am comfortable learning quickly, taking feedback, and applying my skills to real engineering problems."
    if "where do you see" in q:
        return "I want to become a dependable software engineer with strong fundamentals, contribute to production-quality projects, and gradually take ownership of features and technical decisions."
    if "motivat" in q or "keep learning" in q:
        return "I enjoy understanding how things work and then applying that knowledge to solve problems. Seeing measurable improvement in my projects motivates me to keep learning."
    if "learned a new technical concept" in q:
        return "A good example is learning a new tool or concept for a project. I first understood the fundamentals, followed a small hands-on example, applied it to my project, and then tested and refined the implementation. This helped me learn quickly without waiting until I knew everything."
    if "challenging bug" in q:
        return "I first reproduced the bug consistently, narrowed down the failing component, checked logs and inputs, identified the root cause, applied a focused fix, and tested both the original case and related cases. The experience reinforced the value of systematic debugging."
    if "constructive feedback" in q:
        return "I listen carefully to the feedback, ask questions if something is unclear, and apply the suggestion. After making the change, I check the result and use the feedback to improve my approach in future work."
    if "multiple assignments" in q or "deadlines" in q:
        return "I list the tasks, compare their deadlines and impact, break larger tasks into smaller milestones, and complete the highest-priority work first. If there is a conflict, I communicate early rather than missing a deadline silently."
    if "disagreement" in q or "conflict" in q:
        return "I would first understand my teammate's reasoning, explain my own view using facts or requirements, and look for a solution that best serves the project. If needed, I would involve the appropriate senior teammate and then support the agreed decision."
    if "ownership" in q:
        return "I take ownership by understanding the expected outcome, tracking the task through implementation and testing, communicating blockers early, and making sure the final result meets the requirement rather than stopping after writing the code."
    if "setback" in q:
        return "I would understand what caused the setback, separate controllable from uncontrollable factors, fix the immediate issue, and document what I learned so the same problem is less likely to happen again."
    if "explain a technical idea" in q:
        return "I start with the person's current level of understanding, explain the idea using simple terms and a small example, then check whether the explanation makes sense before adding more technical detail."
    return "I would first understand the requirement, break the problem into smaller steps, implement a clear solution, test it carefully, and communicate the result and any assumptions."


def _build_local_kit(profile, counts):
    """Build a complete offline interview kit when Granite quota is exhausted."""
    from agent.orchestrator import InterviewKit

    banks = {
        "technical": [
            "Explain the difference between a Python list and tuple and when you would use each.",
            "What is a SQL JOIN? Explain INNER JOIN and LEFT JOIN with an example.",
            "What is a REST API and what are common HTTP methods used by it?",
            "How would you find and fix a bug in a Python program that gives the wrong output?",
            "Explain time complexity using a simple example such as searching or sorting.",
            "What is the difference between authentication and authorization?",
            "How would you validate input in a backend API?",
            "Explain OOP and compare it with a functional programming approach.",
            "How would you preprocess a dataset before training a machine-learning model?",
            "How would you design a simple REST API for user registration?"
        ],
        "role_specific": [
            f"What do you understand about the responsibilities of a {getattr(profile, 'role', 'Software Engineer') or 'Software Engineer'}?",
            "How would you approach a new software task when the requirements are not completely clear?",
            "How would you design a small feature from requirement to testing and deployment?",
            "How do you make sure your code is maintainable and easy for teammates to understand?",
            "What would you do if you were assigned a technology you had never used before?",
            "How would you prioritize correctness, performance, and delivery time on a project?"
        ],
        "hr": [
            f"Why are you interested in a {getattr(profile, 'role', 'Software Engineer') or 'Software Engineer'} role?",
            "What are your key strengths as a fresher?",
            "What are you looking for in your first professional role?",
            "What is one area you are currently improving?",
            "Why should we hire you?",
            "What motivates you to keep learning?"
        ],
        "behavioral": [
            "Tell me about a time you learned a new technical concept quickly.",
            "Tell me about a challenging bug or problem you solved.",
            "Describe a time you received constructive feedback and how you responded.",
            "Tell me about a time you had multiple assignments or deadlines. How did you prioritize them?",
            "Describe a disagreement with a teammate and how you handled it.",
            "Tell me about a time you took ownership of a task or project.",
        ],
    }

    def pick(key):
        n = int(counts.get(key, 0) or 0)
        bank = banks[key]
        return [bank[i % len(bank)] for i in range(n)]

    kit = InterviewKit(profile=profile)
    kit.technical_questions = pick("technical")
    kit.role_questions = pick("role_specific")
    kit.hr_questions = pick("hr")
    kit.behavioral_questions = pick("behavioral")
    kit.model_answers = {
        q: _local_model_answer(q, profile) for q in kit.all_questions
    }
    kit.tips = (
        "1. Keep technical answers structured: definition → approach → example.\n"
        "2. For coding questions, explain complexity and edge cases.\n"
        "3. For SQL/API questions, mention validation, errors, and security.\n"
        "4. For behavioral questions, use STAR: Situation → Task → Action → Result.\n"
        "5. Connect answers to your projects and clearly explain your own contribution."
    )
    kit.strategy = (
        "### 7-Day Preparation Strategy\n"
        "1. Revise Python, SQL, REST APIs, Git, and core CS fundamentals.\n"
        "2. Prepare two project explanations covering your role, challenges, and results.\n"
        "3. Practice behavioral questions using STAR.\n"
        "4. Practice explaining one technical concept in under two minutes.\n"
        "5. Complete a timed mock interview and review weak answers.\n"
        "6. Revisit topics where you struggled and practice concise explanations.\n"
        "7. Do a final mock interview and focus on confidence, clarity, and structure."
    )
    return kit


def _is_rate_limit_error(err):
    """Return True when IBM should be bypassed and local fallback used."""
    text = str(err or "").lower()
    return any(
        x in text
        for x in (
            "429",
            "rate limit",
            "quota exceeded",
            "token_quota",
            "too many requests",
            "overloaded",
            "403",
            "access denied",
            "project_id",
            "project id",
        )
    )


def _make_local_mock(profile, total_questions, interview_type):
    return LocalMockInterviewer(
        profile=profile,
        total_questions=int(total_questions),
        interview_type=interview_type,
    )


def _local_final_feedback(profile, mi):
    avg = mi.get_average_score()
    completed = len(mi.history)
    if avg >= 8:
        readiness = "Strong mock performance. Focus on consistency and concise delivery."
    elif avg >= 6:
        readiness = "Good foundation. Focus on adding concrete examples, stronger structure, and clearer technical reasoning."
    else:
        readiness = "Keep practicing. Focus first on complete answers, clear structure, and explaining your reasoning step by step."
    strengths = []
    improvements = []
    if completed:
        long_answers = sum(len(t.user_answer.split()) >= 25 for t in mi.history)
        if long_answers >= max(1, completed // 2):
            strengths.append("You provided reasonably detailed answers.")
        else:
            improvements.append("Make answers more complete and specific; avoid one- or two-sentence responses.")
        if any(t.score >= 8 for t in mi.history):
            strengths.append("You demonstrated the ability to give strong answers on some questions.")
        if not strengths:
            strengths.append("You completed the mock interview and practiced under interview conditions.")
    else:
        strengths.append("Mock interview was started, but no answers were submitted yet.")
    if not improvements:
        improvements.append("Use a simple structure: point → reasoning → example → result.")
        improvements.append("For behavioral questions, use the STAR structure when appropriate.")
    name = getattr(profile, "name", "Candidate") or "Candidate"
    role = getattr(profile, "role", "Software Engineer") or "Software Engineer"
    lines = [
        f"## 📊 Final Mock Interview Feedback — {name}",
        f"**Target role:** {role}",
        f"**Questions answered:** {completed}/{mi.total_questions}",
        f"**Average score:** {avg}/10",
        "",
        f"### Overall Assessment\n{readiness}",
        "",
        "### ✅ Strengths",
    ]
    lines += [f"- {x}" for x in strengths]
    lines += ["", "### ⚠️ Areas to Improve"]
    lines += [f"- {x}" for x in improvements]
    lines += [
        "",
        "### 📅 7-Day Practice Plan",
        "1. Practice Python, SQL, REST API, and core CS questions.",
        "2. Prepare two project stories with your contribution and measurable results.",
        "3. Practice behavioral answers using STAR.",
        "4. Do one timed mock interview and keep answers concise.",
        "5. Review weak technical topics and write short explanations.",
        "6. Practice explaining your projects without reading notes.",
        "7. Repeat a final mock interview and focus on confidence and clarity.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 1+2: Generate Interview Kit handler
# ---------------------------------------------------------------------------

def generate_kit(
    name, role, experience, skills_text, resume_text,
    interview_type, num_questions, difficulty,
    progress=gr.Progress(track_tqdm=False),
):
    """Main handler: parse profile → run orchestrator → return formatted outputs."""

    if not _credentials_ok():
        logger.warning("IBM credentials are not configured; using local Interview Kit fallback.")
        parser = ProfileParser()
        profile = parser.parse(
            name=name or "Candidate",
            role=role,
            experience_level=experience,
            skills=skills_text,
            resume_summary=resume_text,
        )
        counts = _build_counts(interview_type, int(num_questions))
        kit = _build_local_kit(profile, counts)
        summary_md = (
            f"⚡ **Interview kit ready for {profile.name} — Local Fallback Mode**\n\n"
            "IBM Granite credentials are not configured, so local interview content is being used.\n\n"
            + format_interview_kit_summary(kit.to_dict())
        )
        return (
            summary_md,
            format_all_questions(kit.technical_questions, kit.role_questions, kit.hr_questions, kit.behavioral_questions),
            format_model_answers(kit.model_answers),
            format_tips(kit.tips),
            format_strategy(kit.strategy),
            "", ""
        )

    progress(0.05, desc="Parsing profile…")
    parser = ProfileParser()
    profile = parser.parse(
        name=name or "Candidate",
        role=role,
        experience_level=experience,
        skills=skills_text,
        resume_summary=resume_text,
    )

    # Map UI difficulty to prompt string
    diff_map = {
        "Auto (based on experience)": None,
        "Basic": "basic",
        "Intermediate": "intermediate",
        "Advanced": "advanced",
    }
    difficulty_val = diff_map.get(difficulty)

    # Map interview type to counts
    counts = _build_counts(interview_type, int(num_questions))

    progress(0.15, desc="Generating questions…")
    kit, err = _safe_generate(
        _orchestrator.run,
        profile,
        question_counts=counts,
        generate_answers=True,
        generate_tips=True,
        generate_strategy=True,
    )

    if err:
        # IBM Granite can reject requests when the monthly Lite token quota is
        # exhausted. In that case, keep the complete Interview Kit usable with
        # deterministic local content instead of showing a generation failure.
        if _is_rate_limit_error(err):
            logger.warning("IBM Granite unavailable for Interview Kit; using local fallback: %s", err)
            progress(0.35, desc="IBM Granite quota reached — using local interview content…")
            kit = _build_local_kit(profile, counts)
            summary_md = (
                f"⚡ **Interview kit ready for {profile.name} — Local Fallback Mode**\n\n"
                "IBM Granite is currently unavailable because its token quota has been reached. "
                "The kit below was generated locally so you can continue your preparation.\n\n"
                + format_interview_kit_summary(kit.to_dict())
            )
            questions_md = format_all_questions(
                kit.technical_questions,
                kit.role_questions,
                kit.hr_questions,
                kit.behavioral_questions,
            )
            answers_md = format_model_answers(kit.model_answers)
            tips_md = format_tips(kit.tips)
            strategy_md = format_strategy(kit.strategy)
            progress(1.0, desc="Done — local fallback used.")
            return summary_md, questions_md, answers_md, tips_md, strategy_md, "", ""
        return err, "", "", "", "", "", err

    progress(0.95, desc="Formatting outputs…")

    # --- Format each tab output ---
    summary_md = (
        f"✅ **Interview kit ready for {profile.name}**\n\n"
        + format_interview_kit_summary(kit.to_dict())
    )
    questions_md   = format_all_questions(
        kit.technical_questions,
        kit.role_questions,
        kit.hr_questions,
        kit.behavioral_questions,
    )
    answers_md     = format_model_answers(kit.model_answers)
    tips_md        = format_tips(kit.tips)
    strategy_md    = format_strategy(kit.strategy)

    progress(1.0, desc="Done!")
    logger.info(
        "Kit generated for %s | %s | %d questions.",
        profile.name, profile.role, len(kit.all_questions),
    )
    return summary_md, questions_md, answers_md, tips_md, strategy_md, "", ""


def _build_counts(interview_type: str, num_questions: int) -> dict:
    """Distribute question count across types based on interview_type selection."""
    if interview_type == "Technical":
        return {"technical": num_questions, "role_specific": 0, "hr": 0, "behavioral": 0}
    if interview_type == "HR":
        return {"technical": 0, "role_specific": 0, "hr": num_questions, "behavioral": 0}
    if interview_type == "Behavioral":
        return {"technical": 0, "role_specific": 0, "hr": 0, "behavioral": num_questions}
    if interview_type == "Role-Specific":
        return {"technical": 0, "role_specific": num_questions, "hr": 0, "behavioral": 0}
    # Mixed — "Questions per Type" means exactly this many questions
    # for EACH category. For example, selecting 3 generates 3 technical,
    # 3 role-specific, 3 HR, and 3 behavioral questions (12 total).
    return {
        "technical": num_questions,
        "role_specific": num_questions,
        "hr": num_questions,
        "behavioral": num_questions,
    }


# ---------------------------------------------------------------------------
# Tab 3: Answer Evaluator handler
# ---------------------------------------------------------------------------

def _local_evaluate_answer(question, answer, profile):
    """Deterministic offline answer evaluator used when IBM Granite is unavailable."""
    q = (question or "").strip().lower()
    a = (answer or "").strip()
    words = a.split()
    word_count = len(words)

    # Topic-specific checks make the fallback useful rather than returning
    # the same generic feedback for every answer.
    checks = []
    if "list" in q and "tuple" in q:
        checks = [
            ("mutable" in q or "mutable" in a.lower(), "Explain that a list is mutable."),
            ("immutable" in a.lower(), "Explain that a tuple is immutable."),
            ("list" in a.lower(), "State when a list is appropriate."),
            ("tuple" in a.lower(), "State when a tuple is appropriate."),
        ]
    elif "responsibilities" in q and "software engineer" in q:
        checks = [
            ("requirement" in a.lower(), "Mention understanding requirements."),
            ("develop" in a.lower() or "code" in a.lower(), "Mention software development or coding."),
            ("test" in a.lower(), "Mention testing and debugging."),
            ("collabor" in a.lower() or "team" in a.lower(), "Mention collaboration with the team."),
        ]
    elif "why are you interested" in q:
        checks = [
            ("problem" in a.lower() or "solve" in a.lower(), "Connect your interest to problem solving."),
            ("project" in a.lower(), "Give a relevant project or hands-on example."),
            ("learn" in a.lower() or "grow" in a.lower(), "Explain how you want to learn and grow."),
            ("software" in a.lower() or "engineer" in a.lower(), "Connect your motivation clearly to the Software Engineer role."),
        ]
    else:
        # Generic completeness checks for other questions.
        checks = [
            (word_count >= 25, "Add enough detail to fully explain your answer."),
            ("because" in a.lower() or "so that" in a.lower() or "therefore" in a.lower(), "Explain your reasoning, not only the conclusion."),
            ("example" in a.lower() or "for example" in a.lower(), "Include a concrete example where appropriate."),
        ]

    passed = sum(bool(ok) for ok, _ in checks)
    total = max(1, len(checks))
    # Keep the score realistic and avoid giving a perfect score solely because
    # the answer contains keywords.
    if word_count < 8:
        score = 4.0
    else:
        score = min(9.5, round(5.5 + 4.0 * passed / total, 1))
        if word_count >= 25:
            score = min(10.0, round(score + 0.5, 1))

    strengths = []
    weaknesses = []
    missing_points = []
    for ok, msg in checks:
        if ok:
            if len(strengths) < 3:
                strengths.append(msg.replace("Mention ", "You ").replace("Explain that ", "You correctly explain that ").replace("State when ", "You state when ").replace("Connect ", "You connect ").replace("Give ", "You give "))
        else:
            missing_points.append(msg)

    if score >= 8:
        strengths.insert(0, "Your answer is clear, relevant, and appropriately structured.")
    elif score >= 6:
        strengths.insert(0, "Your answer addresses the main topic, but it can be made more specific.")
    else:
        weaknesses.append("The answer needs more detail and a clearer structure.")

    if not missing_points and score >= 8:
        weaknesses.append("Keep the answer concise while adding one concrete example when useful.")
    else:
        weaknesses.extend(missing_points[:2])

    suggestions = [
        "Start with the main point, then give the reasoning or explanation.",
        "Use a short project or practical example when it strengthens the answer.",
        "End with a clear conclusion instead of adding unnecessary detail.",
    ]

    return {
        "score": f"{score}/10",
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "missing_points": missing_points[:3],
        "tech_accuracy": "The answer covers the key concepts relevant to the question." if ("list" in q and "tuple" in q) else "No obvious technical issue was detected by the local evaluator.",
        "communication": "Clear and professional. Keep the structure concise and interview-focused." if word_count >= 15 else "Add a little more detail and explain your reasoning clearly.",
        "suggestions": suggestions,
        "model_answer": _local_model_answer(question, profile),
    }


def evaluate_answer(role, experience, skills_text, question_text, answer_text):
    """Evaluate a single answer, falling back locally when Granite is unavailable."""
    if not question_text or not question_text.strip():
        return "⚠️ Please enter a question to evaluate."
    if not answer_text or not answer_text.strip():
        return "⚠️ Please enter your answer before evaluating."

    parser = ProfileParser()
    profile = parser.parse(role=role, experience_level=experience, skills=skills_text)

    # Try IBM Granite first when credentials are available. If the service is
    # unavailable because of quota/access/rate limits, use the deterministic
    # local evaluator so this tab remains usable.
    result = None
    err = None
    if _credentials_ok():
        result, err = _safe_generate(
            _orchestrator.evaluate_answer,
            question=question_text,
            answer=answer_text,
            profile=profile,
        )

    local_mode = False
    if err or result is None:
        if err:
            logger.warning("IBM answer evaluation unavailable; using local evaluator: %s", err)
        else:
            logger.warning("IBM answer evaluation unavailable; using local evaluator.")
        result = _local_evaluate_answer(question_text, answer_text, profile)
        local_mode = True

    lines = [
        "## 📋 Answer Evaluation",
        f"**Question:** {question_text.strip()}",
        "---",
        f"### Overall Score: {result.get('score', 'N/A')}",
    ]
    if result.get("strengths"):
        lines.append("\n**✅ Strengths:**")
        for s in result["strengths"]:
            lines.append(f"- {s}")
    if result.get("weaknesses"):
        lines.append("\n**⚠️ Weaknesses:**")
        for w in result["weaknesses"]:
            lines.append(f"- {w}")
    if result.get("missing_points"):
        lines.append("\n**🔍 Missing Points:**")
        for m in result["missing_points"]:
            lines.append(f"- {m}")
    if result.get("tech_accuracy"):
        lines.append(f"\n**🔬 Technical Accuracy:**\n{result['tech_accuracy']}")
    if result.get("communication"):
        lines.append(f"\n**🗣️ Communication Feedback:**\n{result['communication']}")
    if result.get("suggestions"):
        lines.append("\n**💡 Improvement Suggestions:**")
        for i, s in enumerate(result["suggestions"], 1):
            lines.append(f"{i}. {s}")
    if result.get("model_answer"):
        lines.append(f"\n**📝 Model Answer:**\n{result['model_answer']}")
    if local_mode:
        lines.extend(["", "⚡ *Local Answer Evaluation Mode — IBM Granite was unavailable.*"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 4: Mock Interview handlers
# ---------------------------------------------------------------------------

def start_mock_interview(name, role, experience, skills_text, resume_text,
                          total_questions, interview_type, mock_state):
    """Start IBM mock interview; automatically fall back to local mode on 429."""
    parser = ProfileParser()
    profile = parser.parse(
        name=name or "Candidate",
        role=role,
        experience_level=experience,
        skills=skills_text,
        resume_summary=resume_text,
    )
    total = int(total_questions)

    # Always allow the Mock Interview to start even if IBM credentials are unavailable.
    if not _credentials_ok():
        mi = _make_local_mock(profile, total, interview_type)
        first_q = mi.start_session()
        mock_state["interviewer"] = mi
        mock_state["profile"] = profile
        mock_state["local_fallback"] = True
        return (
            "### 🎤 Mock Interview Started\n\n"
            "⚡ **Offline fallback mode:** IBM Granite is unavailable, so this mock interview is running locally.\n\n"
            + first_q,
            mock_state, f"Question 1 of {total} (Local Mode)",
            gr.update(interactive=True), gr.update(interactive=True),
        )

    mi, err = _safe_generate(
        _orchestrator.create_mock_interviewer,
        profile=profile,
        total_questions=total,
        interview_type=interview_type,
    )

    # IMPORTANT: Mock Interview must never be blocked by an IBM API error.
    # Any failure while creating the IBM interviewer starts the deterministic
    # local interviewer instead. This covers 403, 429, quota, timeout, and
    # other transient/unavailable-model errors.
    if err:
        logger.warning("IBM mock interviewer unavailable; using local mode: %s", err)
        mi = _make_local_mock(profile, total, interview_type)
        first_q = mi.start_session()
        mock_state["interviewer"] = mi
        mock_state["profile"] = profile
        mock_state["local_fallback"] = True
        return (
            "### 🎤 Mock Interview Started\n\n"
            "⚡ **Local Mock Interview Mode**\n"
            "IBM Granite is currently unavailable, so the interview is running locally. "
            "Questions and answer feedback will still be delivered one at a time.\n\n"
            + first_q,
            mock_state, f"Question 1 of {total}",
            gr.update(interactive=True), gr.update(interactive=True),
        )

    first_q, err = _safe_generate(mi.start_session)
    if err:
        logger.warning("IBM mock session could not start; using local mode: %s", err)
        mi = _make_local_mock(profile, total, interview_type)
        first_q = mi.start_session()
        mock_state["interviewer"] = mi
        mock_state["profile"] = profile
        mock_state["local_fallback"] = True
        return (
            "### 🎤 Mock Interview Started\n\n"
            "⚡ **Local Mock Interview Mode**\n"
            "IBM Granite is currently unavailable, so the interview is running locally. "
            "Questions and answer feedback will still be delivered one at a time.\n\n"
            + first_q,
            mock_state, f"Question 1 of {total}",
            gr.update(interactive=True), gr.update(interactive=True),
        )

    mock_state["interviewer"] = mi
    mock_state["profile"] = profile
    mock_state["local_fallback"] = False
    return (
        f"### 🎤 Mock Interview Started\n\n{first_q}",
        mock_state,
        f"Question 1 of {total}",
        gr.update(interactive=True),
        gr.update(interactive=True),
    )



def _clean_mock_feedback(text):
    """Remove prompt-leakage/duplicate next-question text from feedback."""
    if not text:
        return ""
    lines = str(text).splitlines()
    cleaned = []
    skip = False
    for line in lines:
        s = line.strip()
        low = s.lower()
        if low.startswith("ask candidate question"):
            skip = True
            continue
        if low.startswith("[next question]"):
            skip = True
            continue
        if skip:
            # The leaked question is normally a single line. Resume when a
            # markdown separator or blank line is reached.
            if not s or s.startswith("---") or s.startswith("###"):
                skip = False
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def submit_mock_answer(user_answer, mock_state):
    """Submit an answer, show feedback, and display the next question."""
    mi = mock_state.get("interviewer")
    if mi is None:
        return (
            "⚠️ No active mock interview session. Click **Start Mock Interview** first.",
            mock_state, "", gr.update(interactive=False), gr.update(value=""),
        )

    if not user_answer or not user_answer.strip():
        return (
            "⚠️ Please type your answer before submitting.",
            mock_state, "", gr.update(interactive=True), gr.update(),
        )

    result, err = _safe_generate(mi.submit_answer, user_answer)

    if err:
        profile = mock_state.get("profile")
        if profile is not None:
            total = getattr(mi, "total_questions", 8)
            interview_type = getattr(mi, "interview_type", "Mixed")
            local = _make_local_mock(profile, total, interview_type)
            local.start_session()
            mock_state["interviewer"] = local
            mock_state["local_fallback"] = True
            result = local.submit_answer(user_answer)
            mi = local
        else:
            return err, mock_state, "", gr.update(interactive=True), gr.update()

    is_done = result.get("is_complete", False)
    feedback_txt = _clean_mock_feedback(result.get("feedback", ""))
    q_num = result.get("question_number", getattr(mi, "current_question_number", 0))
    total = getattr(mi, "total_questions", 8)

    mode_note = (
        "⚡ *Local Mock Interview Mode — IBM Granite was unavailable.*"
        if mock_state.get("local_fallback") else ""
    )

    display_parts = [
        "### 📊 Feedback",
        "",
        feedback_txt,
    ]
    if mode_note:
        display_parts.extend(["", mode_note])

    if is_done:
        display_parts.extend([
            "",
            "---",
            "### 🏁 Session Complete",
            f"Average Score: **{mi.get_average_score()}/10**",
        ])
        progress_txt = f"Session complete — {total} questions answered"
        return (
            chr(10).join(display_parts),
            mock_state,
            progress_txt,
            gr.update(interactive=False),
            gr.update(value=""),
        )

    next_q = None

    # Local fallback owns the question list, so use it as the authoritative
    # source for the next question.
    questions = getattr(mi, "questions", None)
    current_idx = getattr(mi, "current_question_number", None)
    if questions is not None and current_idx is not None:
        if 0 <= current_idx < len(questions):
            next_q = questions[current_idx]

    # Remote interviewer may explicitly return a next question.
    if not next_q:
        next_q = result.get("next_question")

    if not next_q:
        next_q = getattr(mi, "current_question", None)

    # Never render the same question twice.
    if next_q and next_q.strip() in feedback_txt:
        next_q = None

    next_number = q_num + 2
    progress_txt = f"Question {next_number} of {total}"

    display_parts.extend([
        "",
        "---",
        f"### 🎤 Question {next_number}",
        next_q or "_Next question is ready. Please continue._",
    ])

    return (
        chr(10).join(display_parts),
        mock_state,
        progress_txt,
        gr.update(interactive=True),
        gr.update(value=""),
    )

# ---------------------------------------------------------------------------
# Tab 5: Final Feedback handler
# ---------------------------------------------------------------------------

def generate_final_feedback(name, role, experience, skills_text, resume_text, mock_state):
    """Generate final feedback, with a local fallback if IBM is unavailable."""
    parser = ProfileParser()
    profile = parser.parse(
        name=name or "Candidate",
        role=role,
        experience_level=experience,
        skills=skills_text,
        resume_summary=resume_text,
    )
    mi = mock_state.get("interviewer")

    if mi is not None and mock_state.get("local_fallback"):
        return _local_final_feedback(profile, mi)

    if not _credentials_ok():
        return _local_final_feedback(profile, mi) if mi else _CREDENTIAL_WARNING

    mock_history = mi.get_history_text() if mi else None
    evaluations = [
        {"score": str(t.score), "weaknesses": []} for t in (mi.history if mi else [])
    ]
    from agent.feedback_agent import FeedbackAgent
    fa = FeedbackAgent(llm_client=_orchestrator._get_llm())
    result, err = _safe_generate(
        fa.generate_feedback,
        profile=profile,
        evaluations=evaluations if evaluations else None,
        mock_history=mock_history,
    )
    if err:
        if _is_rate_limit_error(err):
            return _local_final_feedback(profile, mi) if mi else (
                "⚡ **IBM Granite is temporarily unavailable.**\n\n"
                "Please start a Mock Interview to use the local feedback mode."
            )
        return err
    return format_feedback(result)


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct and return the complete Gradio Blocks application."""

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(
        title="AI Interview Trainer Agent",
        theme=theme,
        css="""
        .tab-nav button { font-size: 14px; font-weight: 600; }
        .section-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; }
        """,
    ) as app:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            """
# 🎯 AI Interview Trainer Agent
**Powered by IBM Granite (watsonx.ai) · RAG · AICTE IBM SkillsBuild — Problem Statement #22**

Prepare for your job interview with AI-generated questions, model answers, improvement tips,
a 7-day strategy, and an interactive mock interview.
---
"""
        )

        # ── Shared state for mock interview session ──────────────────────────
        mock_state = gr.State({})

        # ── Shared profile inputs (used across all tabs) ─────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 👤 Candidate Profile", elem_classes="section-header")
                with gr.Row():
                    inp_name = gr.Textbox(
                        label="Your Name",
                        placeholder="e.g. Priya Sharma",
                        scale=1,
                    )
                    inp_role = gr.Dropdown(
                        label="Target Job Role",
                        choices=_SUPPORTED_ROLES,
                        value=_SUPPORTED_ROLES[0],
                        scale=1,
                    )
                with gr.Row():
                    inp_exp = gr.Dropdown(
                        label="Experience Level",
                        choices=_EXPERIENCE_LEVELS,
                        value=_EXPERIENCE_LEVELS[2],
                        scale=1,
                    )
                    inp_skills = gr.Textbox(
                        label="Key Skills (comma-separated)",
                        placeholder="e.g. Python, System Design, SQL, Docker",
                        scale=1,
                    )
                inp_resume = gr.Textbox(
                    label="Resume Summary / Professional Background (optional)",
                    placeholder="Paste a short professional summary or relevant background here…",
                    lines=4,
                )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Interview Configuration", elem_classes="section-header")
                inp_type  = gr.Dropdown(
                    label="Interview Type",
                    choices=_INTERVIEW_TYPES,
                    value="Mixed",
                )
                inp_count = gr.Radio(
                    label="Questions per Type (Mixed = this many in EACH category)",
                    choices=_QUESTION_COUNTS,
                    value=5,
                )
                inp_diff  = gr.Dropdown(
                    label="Difficulty Level",
                    choices=_DIFFICULTY_LEVELS,
                    value="Auto (based on experience)",
                )
                btn_generate = gr.Button(
                    "🚀 Generate My Interview Kit",
                    variant="primary",
                    size="lg",
                )

        gr.Markdown("---")

        # ── Tabs ─────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Overview ───────────────────────────────────────────────
            with gr.Tab("📋 Overview"):
                out_summary = gr.Markdown(
                    "Click **Generate My Interview Kit** above to begin."
                )

            # ── Tab 2: Interview Questions ────────────────────────────────────
            with gr.Tab("❓ Questions"):
                out_questions = gr.Markdown("_Questions will appear here after generation._")

            # ── Tab 3: Model Answers ──────────────────────────────────────────
            with gr.Tab("📝 Model Answers"):
                out_answers = gr.Markdown("_Model answers will appear here after generation._")

            # ── Tab 4: Tips & Strategy ────────────────────────────────────────
            with gr.Tab("💡 Tips & Strategy"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Improvement Tips")
                        out_tips = gr.Markdown("_Tips will appear here after generation._")
                    with gr.Column():
                        gr.Markdown("### 7-Day Preparation Strategy")
                        out_strategy = gr.Markdown("_Strategy will appear here after generation._")

            # ── Tab 5: Answer Evaluator ───────────────────────────────────────
            with gr.Tab("🔍 Evaluate Answer"):
                gr.Markdown(
                    "Paste any interview question and your answer. "
                    "Get an AI evaluation with score, strengths, weaknesses, and a model answer."
                )
                with gr.Row():
                    eval_question = gr.Textbox(
                        label="Interview Question",
                        placeholder="e.g. Explain the difference between a process and a thread.",
                        lines=2,
                    )
                    eval_answer = gr.Textbox(
                        label="Your Answer",
                        placeholder="Type your answer here…",
                        lines=6,
                    )
                btn_evaluate = gr.Button("🔍 Evaluate My Answer", variant="secondary")
                out_evaluation = gr.Markdown("_Your evaluation will appear here._")

                btn_evaluate.click(
                    fn=evaluate_answer,
                    inputs=[inp_role, inp_exp, inp_skills, eval_question, eval_answer],
                    outputs=out_evaluation,
                )

            # ── Tab 6: Mock Interview ─────────────────────────────────────────
            with gr.Tab("🎤 Mock Interview"):
                gr.Markdown(
                    "Experience a realistic mock interview. "
                    "The AI asks one question at a time, evaluates your answer, "
                    "and adapts difficulty based on your performance."
                )
                with gr.Row():
                    mock_total_q = gr.Slider(
                        minimum=3, maximum=15, step=1, value=8,
                        label="Number of Questions",
                    )
                    mock_type = gr.Dropdown(
                        choices=_INTERVIEW_TYPES,
                        value="Mixed",
                        label="Interview Type",
                    )
                    btn_start_mock = gr.Button("▶️ Start Mock Interview", variant="primary")

                mock_progress = gr.Textbox(
                    label="Session Progress",
                    value="Not started",
                    interactive=False,
                )
                out_mock = gr.Markdown(
                    "_Click **Start Mock Interview** to begin. "
                    "The first question will appear here._"
                )
                mock_answer = gr.Textbox(
                    label="Your Answer",
                    placeholder="Type your answer here and click Submit…",
                    lines=5,
                    interactive=False,
                )
                btn_submit_mock = gr.Button(
                    "✅ Submit Answer",
                    variant="secondary",
                    interactive=False,
                )

                btn_start_mock.click(
                    fn=start_mock_interview,
                    inputs=[
                        inp_name, inp_role, inp_exp, inp_skills, inp_resume,
                        mock_total_q, mock_type, mock_state,
                    ],
                    outputs=[out_mock, mock_state, mock_progress, mock_answer, btn_submit_mock],
                )
                btn_submit_mock.click(
                    fn=submit_mock_answer,
                    inputs=[mock_answer, mock_state],
                    outputs=[out_mock, mock_state, mock_progress, btn_submit_mock, mock_answer],
                )

            # ── Tab 7: Final Feedback ─────────────────────────────────────────
            with gr.Tab("📊 Final Feedback"):
                gr.Markdown(
                    "Generate a comprehensive feedback report and 7-day preparation strategy "
                    "personalised to your profile. If you completed a mock interview, "
                    "your session performance will be incorporated."
                )
                btn_feedback = gr.Button(
                    "📊 Generate Feedback Report",
                    variant="primary",
                )
                out_feedback = gr.Markdown("_Your feedback report will appear here._")

                btn_feedback.click(
                    fn=generate_final_feedback,
                    inputs=[
                        inp_name, inp_role, inp_exp, inp_skills, inp_resume,
                        mock_state,
                    ],
                    outputs=out_feedback,
                )

        # ── Wire main Generate button ────────────────────────────────────────
        btn_generate.click(
            fn=generate_kit,
            inputs=[
                inp_name, inp_role, inp_exp, inp_skills, inp_resume,
                inp_type, inp_count, inp_diff,
            ],
            outputs=[
                out_summary, out_questions, out_answers,
                out_tips, out_strategy,
                out_evaluation, out_feedback,
            ],
        )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.Markdown(
            """
---
*AI Interview Trainer Agent · Powered by IBM Granite (watsonx.ai) · "
Built for AICTE IBM SkillsBuild Problem Statement #22*
"""
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting AI Interview Trainer Agent UI…")

    if not _credentials_ok():
        logger.warning(
            "IBM watsonx.ai credentials not set. "
            "Copy .env.example to .env and fill in WATSONX_API_KEY and WATSONX_PROJECT_ID. "
            "The UI will start but generation will be disabled until credentials are provided."
        )

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
