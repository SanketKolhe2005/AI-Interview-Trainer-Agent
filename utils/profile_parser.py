"""
utils/profile_parser.py
========================
Defines the ``UserProfile`` dataclass and a ``ProfileParser`` that validates
and normalises raw user input from the Gradio UI into a clean, typed object
consumed by every agent module.

The parser deliberately never raises on missing optional fields — it fills
defaults so downstream agents always receive a complete object.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Structured representation of a candidate's interview profile.

    Attributes
    ----------
    name : str
        Candidate's display name (default "Candidate").
    role : str
        Target job role.  Must match one of ``settings.SUPPORTED_ROLES``.
    experience_level : str
        Experience band (e.g. "Mid-level (3-6 years)").
    skills : list of str
        Key skills extracted or entered by the candidate.
    resume_summary : str
        Free-text professional summary or pasted resume excerpt.
    """
    name: str = "Candidate"
    role: str = "Software Engineer"
    experience_level: str = "Mid-level (3-6 years)"
    skills: List[str] = field(default_factory=list)
    resume_summary: str = ""

    @property
    def skills_str(self) -> str:
        """Return skills as a comma-separated string."""
        return ", ".join(self.skills) if self.skills else "Not specified"

    def to_dict(self) -> dict:
        """Serialise the profile to a plain dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "experience_level": self.experience_level,
            "skills": self.skills,
            "skills_str": self.skills_str,
            "resume_summary": self.resume_summary,
        }


class ProfileParser:
    """Parses and validates raw user input into a :class:`UserProfile`.

    All parsing methods are defensive — they apply defaults rather than
    raising exceptions on missing or malformed input.
    """

    @staticmethod
    def _normalise_role(raw_role: str) -> str:
        """Fuzzy-match *raw_role* against supported roles.

        Returns the closest supported role, or the first supported role as
        a safe default if no match is found.
        """
        if not raw_role:
            return settings.SUPPORTED_ROLES[0]

        raw_lower = raw_role.strip().lower()

        # Exact match (case-insensitive)
        for role in settings.SUPPORTED_ROLES:
            if role.lower() == raw_lower:
                return role

        # Partial match — e.g. "software" -> "Software Engineer"
        for role in settings.SUPPORTED_ROLES:
            if raw_lower in role.lower() or role.lower().split()[0] in raw_lower:
                return role

        logger.warning(
            "Role '%s' not in supported list %s — using default '%s'.",
            raw_role,
            settings.SUPPORTED_ROLES,
            settings.SUPPORTED_ROLES[0],
        )
        return settings.SUPPORTED_ROLES[0]

    @staticmethod
    def _normalise_experience(raw_exp: str) -> str:
        """Map *raw_exp* to a canonical experience level string."""
        if not raw_exp:
            return settings.EXPERIENCE_LEVELS[2]  # Mid-level default

        raw_lower = raw_exp.strip().lower()

        for level in settings.EXPERIENCE_LEVELS:
            if level.lower() == raw_lower:
                return level
            # Partial keyword match
            keyword = level.split("(")[0].strip().lower()
            if keyword in raw_lower or raw_lower in keyword:
                return level

        return settings.EXPERIENCE_LEVELS[2]

    @staticmethod
    def _parse_skills(raw_skills) -> List[str]:
        """Parse skills from a comma/newline/semicolon separated string or list.

        Returns a deduplicated, stripped list of skill strings.
        """
        if isinstance(raw_skills, list):
            items = raw_skills
        elif isinstance(raw_skills, str):
            # Split on commas, semicolons, or newlines
            items = re.split(r"[,;\n]+", raw_skills)
        else:
            return []

        cleaned = [s.strip() for s in items if s.strip()]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in cleaned:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    def parse(
        self,
        name: Optional[str] = None,
        role: Optional[str] = None,
        experience_level: Optional[str] = None,
        skills=None,
        resume_summary: Optional[str] = None,
    ) -> UserProfile:
        """Build a validated :class:`UserProfile` from raw UI inputs.

        All parameters are optional; defaults are applied for any that are
        missing or empty.

        Parameters
        ----------
        name : str, optional
        role : str, optional
        experience_level : str, optional
        skills : str or list, optional
        resume_summary : str, optional

        Returns
        -------
        UserProfile
        """
        profile = UserProfile(
            name=(name or "Candidate").strip(),
            role=self._normalise_role(role or ""),
            experience_level=self._normalise_experience(experience_level or ""),
            skills=self._parse_skills(skills or []),
            resume_summary=(resume_summary or "").strip(),
        )

        logger.debug(
            "Parsed profile: name=%s, role=%s, exp=%s, skills=%d",
            profile.name,
            profile.role,
            profile.experience_level,
            len(profile.skills),
        )
        return profile
