"""
agent/mock_interviewer.py
==========================
Manages a stateful mock interview session.

The ``MockInterviewer`` maintains the full conversation history, tracks the
current question number, stores per-turn scores, and adapts subsequent
question selection based on candidate performance (difficulty increases on
strong answers, decreases on weak ones).

Session lifecycle
-----------------
  1. Instantiate: ``interviewer = MockInterviewer(llm_client, profile)``
  2. Start:       ``first_question = interviewer.start_session()``
  3. Loop:        ``response = interviewer.submit_answer(user_answer)``
  4. Finish:      ``debrief = interviewer.get_debrief()``  (auto when last Q answered)

Each ``submit_answer()`` call returns a dict containing the AI feedback on the
previous answer and the next question (or the debrief if the session is over).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import settings
from prompts import load_prompt
from rag.retriever import retrieve
from utils.profile_parser import UserProfile

logger = logging.getLogger(__name__)

_DEFAULT_TOTAL_QUESTIONS = 8


@dataclass
class MockTurn:
    """Represents one turn in the mock interview."""
    question_number: int
    question: str
    answer: str = ""
    feedback: str = ""
    score: float = 0.0           # Parsed score (0–10)
    question_type: str = "mixed"  # "technical" | "behavioral" | "hr"


class MockInterviewer:
    """Conduct a stateful mock interview session.

    Parameters
    ----------
    llm_client : GraniteClient
        Initialised IBM Granite client.
    profile : UserProfile
        The candidate's profile — used to personalise questions and evaluation.
    total_questions : int, optional
        Number of questions in the session.  Defaults to 8.
    interview_type : str, optional
        Initial interview type ("Technical", "Mixed", "Behavioral").
    """

    def __init__(
        self,
        llm_client,
        profile: UserProfile,
        total_questions: int = _DEFAULT_TOTAL_QUESTIONS,
        interview_type: str = "Mixed",
    ) -> None:
        self._llm = llm_client
        self.profile = profile
        self.total_questions = total_questions
        self.interview_type = interview_type

        self.history: List[MockTurn] = []
        self.current_question_number: int = 0
        self._session_started: bool = False
        self._session_complete: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when all questions have been asked and answered."""
        return self._session_complete

    @property
    def questions_remaining(self) -> int:
        return max(0, self.total_questions - self.current_question_number)

    def start_session(self) -> str:
        """Begin the mock interview and return the first question.

        Returns
        -------
        str
            The formatted first question turn from the AI.
        """
        if self._session_started:
            raise RuntimeError("Session already started. Call submit_answer() to continue.")

        self._session_started = True
        self.current_question_number = 1
        response = self._call_granite(answer="", phase="QUESTION")
        self._record_turn(question_text=response, answer="", feedback="", score=0.0)
        logger.info(
            "Mock interview started for %s (%s). Total: %d questions.",
            self.profile.name,
            self.profile.role,
            self.total_questions,
        )
        return response

    def submit_answer(self, user_answer: str) -> Dict:
        """Submit the candidate's answer and receive feedback + next question.

        Parameters
        ----------
        user_answer : str
            The candidate's answer to the current question.

        Returns
        -------
        dict with keys:
            feedback        : str  — AI evaluation of the submitted answer
            next_question   : str  — the next question text (empty if session complete)
            is_complete     : bool — True when the last question has been answered
            debrief         : str  — session debrief (present only when is_complete=True)
            question_number : int  — the question number that was answered
        """
        if not self._session_started:
            raise RuntimeError("Call start_session() before submit_answer().")
        if self._session_complete:
            return {"feedback": "", "next_question": "", "is_complete": True,
                    "debrief": self._build_debrief_summary(), "question_number": self.current_question_number}

        # Record the answer on the current (last) turn
        if self.history:
            self.history[-1].answer = user_answer

        # Determine phase
        self.current_question_number += 1
        if self.current_question_number > self.total_questions:
            phase = "DEBRIEF"
            self._session_complete = True
        else:
            phase = "EVALUATION"

        response = self._call_granite(answer=user_answer, phase=phase)
        score = self._parse_score(response)

        # Store feedback on the previous turn
        if self.history:
            self.history[-1].feedback = response
            self.history[-1].score = score

        # Record new turn (question embedded in AI response) — unless debrief
        if not self._session_complete:
            self._record_turn(
                question_text=response,
                answer="",
                feedback="",
                score=0.0,
            )

        result = {
            "feedback":        response,
            "next_question":   "" if self._session_complete else response,
            "is_complete":     self._session_complete,
            "question_number": self.current_question_number - 1,
        }
        if self._session_complete:
            result["debrief"] = response

        return result

    def get_history_text(self) -> str:
        """Return the full conversation history as a formatted string."""
        lines = []
        for turn in self.history:
            lines.append(f"Q{turn.question_number}: {turn.question}")
            if turn.answer:
                lines.append(f"Candidate: {turn.answer}")
            if turn.feedback:
                lines.append(f"Feedback: {turn.feedback}")
            lines.append("")
        return "\n".join(lines)

    def get_average_score(self) -> float:
        """Return the average score across all evaluated turns (0-10)."""
        scored = [t.score for t in self.history if t.score > 0]
        return round(sum(scored) / len(scored), 1) if scored else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_granite(self, answer: str, phase: str) -> str:
        """Build the mock interview prompt and call Granite."""
        context = self._retrieve_context()
        history_text = self.get_history_text()

        # Determine adaptive session phase
        if phase == "QUESTION" or phase == "EVALUATION":
            session_phase = (
                "DEBRIEF" if self.current_question_number > self.total_questions
                else ("EVALUATION" if answer.strip() else "QUESTION")
            )
        else:
            session_phase = "DEBRIEF"

        prompt = load_prompt("mock_interview").format(
            name=self.profile.name,
            role=self.profile.role,
            experience=self.profile.experience_level,
            skills=self.profile.skills_str,
            interview_type=self.interview_type,
            total_questions=self.total_questions,
            current_question_number=self.current_question_number,
            session_phase=session_phase,
            context=context,
            history=history_text or "No history yet.",
            answer=answer or "N/A",
        )

        try:
            return self._llm.generate(prompt, max_new_tokens=500, temperature=0.6)
        except Exception as exc:
            logger.error("Granite mock interview call failed: %s", exc)
            raise RuntimeError(f"Mock interview error: {exc}") from exc

    def _retrieve_context(self) -> str:
        """Retrieve relevant questions and role context for the current session."""
        query = (
            f"interview questions {self.profile.role} "
            f"{self.profile.experience_level} technical behavioral"
        )
        try:
            docs = retrieve(query, role_filter=self.profile.role, k=5)
            return "\n\n".join(d.page_content for d in docs) if docs else "No context."
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG retrieval failed in mock interview: %s", exc)
            return "Context unavailable."

    def _record_turn(
        self, question_text: str, answer: str, feedback: str, score: float
    ) -> None:
        """Add a new turn to history."""
        turn = MockTurn(
            question_number=len(self.history) + 1,
            question=question_text,
            answer=answer,
            feedback=feedback,
            score=score,
        )
        self.history.append(turn)

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a numeric score (x/10) from evaluation text."""
        import re
        match = re.search(r"Quick\s+Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.0

    def _build_debrief_summary(self) -> str:
        avg = self.get_average_score()
        return (
            f"[SESSION COMPLETE]\n"
            f"Overall Score: {avg}/10\n"
            f"Total Questions: {self.total_questions}\n"
            f"Session already finished."
        )
