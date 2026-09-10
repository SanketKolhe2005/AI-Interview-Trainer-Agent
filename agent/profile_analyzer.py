"""
agent/profile_analyzer.py
==========================
Accepts raw candidate input (from the UI or programmatic callers), parses it
into a :class:`~utils.profile_parser.UserProfile`, and optionally uses IBM
Granite to extract a concise professional summary from a raw resume paste.

The ``ProfileAnalyzer`` is the entry point of every agent workflow — all other
agents receive the ``UserProfile`` it produces.
"""

import logging
from typing import Optional

from utils.profile_parser import ProfileParser, UserProfile
from config.settings import settings

logger = logging.getLogger(__name__)


class ProfileAnalyzer:
    """Validate and enrich a candidate's profile.

    Parameters
    ----------
    llm_client : optional
        An instance of :class:`~llm.granite_client.GraniteClient`.  When
        provided and a resume is supplied, Granite is used to extract a clean
        professional summary.  When ``None``, raw input is used as-is.
    """

    def __init__(self, llm_client=None) -> None:
        self._parser = ProfileParser()
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        name: Optional[str] = None,
        role: Optional[str] = None,
        experience_level: Optional[str] = None,
        skills=None,
        resume_text: Optional[str] = None,
    ) -> UserProfile:
        """Parse raw inputs and return a validated :class:`UserProfile`.

        If ``resume_text`` is provided and a Granite client is available, an
        AI-generated professional summary is extracted and stored in
        ``profile.resume_summary``.  Otherwise the raw text is used.

        Parameters
        ----------
        name : str, optional
        role : str, optional
        experience_level : str, optional
        skills : str or list, optional
        resume_text : str, optional
            Raw resume paste or free-text professional summary.

        Returns
        -------
        UserProfile
        """
        profile = self._parser.parse(
            name=name,
            role=role,
            experience_level=experience_level,
            skills=skills,
            resume_summary=resume_text,
        )

        # If we have a Granite client and non-trivial resume text, extract summary
        if self._llm and resume_text and len(resume_text.strip()) > 100:
            profile.resume_summary = self._extract_summary(resume_text, profile)

        logger.info(
            "Profile analysed: %s | %s | %s | %d skills",
            profile.name,
            profile.role,
            profile.experience_level,
            len(profile.skills),
        )
        return profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_summary(self, resume_text: str, profile: UserProfile) -> str:
        """Use Granite to extract a concise professional summary from resume text.

        Falls back to the raw resume text if generation fails.
        """
        prompt = (
            f"You are a professional resume analyser.\n\n"
            f"Extract a concise 3-5 sentence professional summary from the following "
            f"resume/profile text for a candidate applying for a {profile.role} role "
            f"at the {profile.experience_level} level.\n\n"
            f"Focus on: key experience, relevant technical skills, career highlights.\n"
            f"Write in third-person professional style.\n\n"
            f"Resume text:\n{resume_text[:2000]}\n\n"
            f"Professional Summary:"
        )
        try:
            summary = self._llm.generate(prompt, max_new_tokens=150, temperature=0.3)
            if summary and len(summary.strip()) > 20:
                logger.debug("Granite extracted resume summary (%d chars).", len(summary))
                return summary.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Granite resume summary extraction failed: %s — using raw text.", exc)

        return resume_text.strip()[:500]  # Fallback: first 500 chars of raw text

    @staticmethod
    def get_supported_roles() -> list:
        """Return the list of supported job roles from settings."""
        return settings.SUPPORTED_ROLES

    @staticmethod
    def get_experience_levels() -> list:
        """Return the list of supported experience level strings from settings."""
        return settings.EXPERIENCE_LEVELS
