"""
agent/answer_evaluator.py

Evaluates a candidate's interview answer using RAG context and IBM Granite.

The evaluator retrieves relevant knowledge-base context for the question,
injects it into the answer_evaluation prompt template, and calls Granite
to produce structured evaluation covering score, strengths, weaknesses,
technical accuracy, communication, improvement suggestions, and a model answer.

The raw Granite response is also parsed into a structured dictionary so
downstream consumers such as the UI and orchestrator can access individual fields.
"""

import logging
import re
from typing import Dict, List

from config.settings import settings
from prompts import load_prompt
from rag.retriever import retrieve
from utils.profile_parser import UserProfile

logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """Evaluate a candidate's interview answer and produce structured feedback."""

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        answer: str,
        profile: UserProfile,
        question_type: str = "general",
    ) -> Dict:
        """Evaluate an interview answer."""

        if not answer or not answer.strip():
            return self._empty_evaluation("No answer provided.")

        context = self._retrieve_context(question, profile.role)

        prompt = load_prompt("answer_evaluation").format(
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            question=question,
            answer=answer,
            question_type=question_type,
            context=context,
        )

        try:
            raw = self._llm.generate(
                prompt,
                max_new_tokens=600,
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("Granite evaluation failed: %s", exc)
            raise RuntimeError(f"Answer evaluation failed: {exc}") from exc

        parsed = self._parse_evaluation(raw)
        parsed["raw_text"] = raw

        return parsed

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        profile: UserProfile,
        question_types: List[str] | None = None,
    ) -> List[Dict]:
        """Evaluate multiple question/answer pairs."""

        if len(questions) != len(answers):
            raise ValueError(
                "questions and answers lists must have equal length."
            )

        if question_types is not None and len(question_types) != len(questions):
            raise ValueError(
                "questions and question_types lists must have equal length."
            )

        results = []

        for index, (question, answer) in enumerate(zip(questions, answers)):
            q_type = (
                question_types[index]
                if question_types is not None
                else "general"
            )

            try:
                result = self.evaluate(
                    question,
                    answer,
                    profile,
                    question_type=q_type,
                )
            except Exception as exc:
                logger.warning(
                    "Batch evaluation error for question '%s…': %s",
                    question[:40],
                    exc,
                )
                result = self._empty_evaluation(
                    f"Evaluation error: {exc}"
                )

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, question: str, role: str) -> str:
        """Retrieve relevant context for the question and role."""

        try:
            docs = retrieve(
                question,
                role_filter=role,
                k=settings.TOP_K_RETRIEVAL,
            )

            if not docs:
                return "No additional context retrieved."

            return "\n\n".join(
                doc.page_content for doc in docs
            )

        except Exception as exc:
            logger.warning(
                "RAG retrieval failed during evaluation: %s",
                exc,
            )
            return "Context unavailable."

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_evaluation(text: str) -> Dict:
        """Parse Granite's structured evaluation response."""

        if not text:
            return AnswerEvaluator._empty_evaluation(
                "No evaluation was generated."
            )

        # Remove common generation artifacts.
        cleaned = text.strip()

        cleaned = re.sub(
            r"```(?:markdown|text|python|sql)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.replace("svgsvg", "")

        def extract_section(label: str) -> str:
            """
            Extract text belonging to a section.

            Example:
                STRENGTHS:
                - Point one
                - Point two

            Stops at the next known section heading.
            """

            labels = [
                "OVERALL SCORE",
                "STRENGTHS",
                "WEAKNESSES",
                "MISSING POINTS",
                "TECHNICAL ACCURACY",
                "COMMUNICATION FEEDBACK",
                "IMPROVEMENT SUGGESTIONS",
                "MODEL ANSWER",
            ]

            other_labels = [
                item for item in labels
                if item.upper() != label.upper()
            ]

            next_section = "|".join(
                re.escape(item) for item in other_labels
            )

            pattern = re.compile(
                rf"{re.escape(label)}\s*:?\s*"
                rf"(.*?)"
                rf"(?=\n\s*(?:{next_section})\s*:|\Z)",
                re.IGNORECASE | re.DOTALL,
            )

            match = pattern.search(cleaned)

            if not match:
                return ""

            return match.group(1).strip()

        def to_bullets(section_text: str) -> List[str]:
            """Convert a section into clean list items."""

            if not section_text:
                return []

            items = []

            for line in section_text.splitlines():
                line = line.strip()

                if not line:
                    continue

                # Remove markdown bullets.
                line = re.sub(
                    r"^[-*•]\s*",
                    "",
                    line,
                )

                # Remove numbered bullets.
                line = re.sub(
                    r"^\d+\s*[\.\)]\s*",
                    "",
                    line,
                )

                # Remove markdown emphasis.
                line = re.sub(
                    r"^\*+|\*+$",
                    "",
                    line,
                ).strip()

                if line:
                    items.append(line)

            return items

        # --------------------------------------------------------------
        # Score
        # --------------------------------------------------------------

        score_match = re.search(
            r"(?:OVERALL\s+SCORE|SCORE)"
            r"\s*:?\s*"
            r"(?:\*\*)?\s*"
            r"(\d+(?:\.\d+)?)"
            r"\s*/\s*10",
            cleaned,
            re.IGNORECASE,
        )

        score = (
            f"{score_match.group(1)}/10"
            if score_match
            else "N/A"
        )

        # --------------------------------------------------------------
        # Structured fields
        # --------------------------------------------------------------

        strengths = to_bullets(
            extract_section("STRENGTHS")
        )

        weaknesses = to_bullets(
            extract_section("WEAKNESSES")
        )

        missing_points = to_bullets(
            extract_section("MISSING POINTS")
        )

        suggestions = to_bullets(
            extract_section("IMPROVEMENT SUGGESTIONS")
        )

        technical_accuracy = extract_section(
            "TECHNICAL ACCURACY"
        )

        communication = extract_section(
            "COMMUNICATION FEEDBACK"
        )

        model_answer = extract_section(
            "MODEL ANSWER"
        )

        # Remove accidental trailing artifacts.
        model_answer = re.sub(
            r"```$",
            "",
            model_answer,
        ).strip()

        return {
            "raw_text": cleaned,
            "score": score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "missing_points": missing_points,
            "tech_accuracy": technical_accuracy,
            "communication": communication,
            "suggestions": suggestions,
            "model_answer": model_answer,
        }

    # ------------------------------------------------------------------
    # Empty evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_evaluation(reason: str) -> Dict:
        """Return an empty evaluation structure."""

        return {
            "raw_text": reason,
            "score": "N/A",
            "strengths": [],
            "weaknesses": [],
            "missing_points": [],
            "tech_accuracy": reason,
            "communication": reason,
            "suggestions": [],
            "model_answer": "",
        }