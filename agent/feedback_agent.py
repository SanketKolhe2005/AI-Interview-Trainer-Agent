"""
agent/feedback_agent.py
========================
Generates a comprehensive interview preparation feedback report using IBM
Granite and relevant RAG context.

The ``FeedbackAgent`` is designed to be called after a complete mock interview
session or after the full question set has been reviewed.  It synthesises
information about the candidate's profile and (optionally) mock interview
performance into an actionable, structured feedback report and 7-day plan.
"""

import logging
from typing import List, Optional

from config.settings import settings
from prompts import load_prompt
from rag.retriever import retrieve
from utils.profile_parser import UserProfile

logger = logging.getLogger(__name__)


class FeedbackAgent:
    """Generate comprehensive interview preparation feedback.

    Parameters
    ----------
    llm_client : GraniteClient
        Initialised IBM Granite client.
    """

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_feedback(
        self,
        profile: UserProfile,
        questions: Optional[List[str]] = None,
        evaluations: Optional[List[dict]] = None,
        mock_history: Optional[str] = None,
    ) -> str:
        """Generate a full feedback report for the candidate.

        Parameters
        ----------
        profile : UserProfile
            Candidate profile.
        questions : list of str, optional
            The questions that were generated/reviewed in the session.
        evaluations : list of dict, optional
            Evaluation results from :class:`~agent.answer_evaluator.AnswerEvaluator`.
        mock_history : str, optional
            Transcript of the mock interview session (if any).

        Returns
        -------
        str
            The full feedback report text (markdown-formatted).
        """
        performance_context = self._build_performance_context(
            questions, evaluations, mock_history
        )
        context = self._retrieve_context(profile)
        prompt = load_prompt("feedback").format(
            name=profile.name,
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            resume_summary=profile.resume_summary or "Not provided",
            performance_context=performance_context,
            context=context,
        )

        try:
            feedback_text = self._llm.generate(
                prompt, max_new_tokens=800, temperature=0.5
            )
        except Exception as exc:
            logger.error("Granite feedback generation failed: %s", exc)
            raise RuntimeError(f"Feedback generation failed: {exc}") from exc

        logger.info("Feedback report generated (%d chars).", len(feedback_text))
        return feedback_text

    def generate_tips(
        self,
        profile: UserProfile,
        questions: Optional[List[str]] = None,
    ) -> str:
        """Generate targeted improvement tips for the candidate.

        Parameters
        ----------
        profile : UserProfile
        questions : list of str, optional
            Questions the candidate has reviewed; used to identify topic gaps.

        Returns
        -------
        str
            Improvement tips text (markdown-formatted).
        """
        questions_text = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(questions or [])
        ) or "No questions provided."

        context = self._retrieve_context(profile)
        prompt = load_prompt("tips_prompt").format(
            name=profile.name,
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            resume_summary=profile.resume_summary or "Not provided",
            questions=questions_text,
            context=context,
        )

        try:
            tips_text = self._llm.generate(
                prompt, max_new_tokens=600, temperature=0.5
            )
        except Exception as exc:
            logger.error("Granite tips generation failed: %s", exc)
            raise RuntimeError(f"Tips generation failed: {exc}") from exc

        logger.info("Improvement tips generated (%d chars).", len(tips_text))
        return tips_text

    def generate_strategy(self, profile: UserProfile) -> str:
        """Generate a 7-day interview preparation strategy for the candidate.

        Parameters
        ----------
        profile : UserProfile

        Returns
        -------
        str
            7-day strategy text (markdown-formatted).
        """
        context = self._retrieve_context(profile, query_suffix="interview preparation strategy")
        prompt = load_prompt("strategy_prompt").format(
            name=profile.name,
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            resume_summary=profile.resume_summary or "Not provided",
            context=context,
        )

        try:
            strategy_text = self._llm.generate(
                prompt, max_new_tokens=800, temperature=0.5
            )
        except Exception as exc:
            logger.error("Granite strategy generation failed: %s", exc)
            raise RuntimeError(f"Strategy generation failed: {exc}") from exc

        logger.info("Preparation strategy generated (%d chars).", len(strategy_text))
        return strategy_text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_context(
        self,
        profile: UserProfile,
        query_suffix: str = "interview preparation guidelines",
    ) -> str:
        """Retrieve preparation and role-specific context from RAG."""
        query = f"{profile.role} {profile.experience_level} {query_suffix}"
        try:
            docs = retrieve(query, role_filter=profile.role, k=settings.TOP_K_RETRIEVAL)
            if not docs:
                return "No additional context retrieved."
            return "\n\n".join(d.page_content for d in docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG retrieval failed in FeedbackAgent: %s", exc)
            return "Context unavailable."

    @staticmethod
    def _build_performance_context(
        questions: Optional[List[str]],
        evaluations: Optional[List[dict]],
        mock_history: Optional[str],
    ) -> str:
        """Assemble a performance summary string from available session data."""
        parts = []

        if questions:
            parts.append(
                f"Questions reviewed ({len(questions)} total):\n"
                + "\n".join(f"  - {q}" for q in questions[:10])
            )

        if evaluations:
            scores = [e.get("score", "N/A") for e in evaluations]
            parts.append(f"Answer scores: {', '.join(scores)}")

            # Collect all weaknesses for summary
            all_weak = []
            for e in evaluations:
                all_weak.extend(e.get("weaknesses", []))
            if all_weak:
                parts.append(
                    "Key weaknesses identified:\n"
                    + "\n".join(f"  - {w}" for w in all_weak[:6])
                )

        if mock_history:
            parts.append(f"Mock interview transcript:\n{mock_history[:800]}")

        if not parts:
            parts.append(
                "No detailed performance data available. "
                "Generate feedback based on the candidate profile only."
            )

        return "\n\n".join(parts)
