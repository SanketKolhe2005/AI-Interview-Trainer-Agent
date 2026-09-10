"""
agent/question_generator.py
===========================

Generates all four interview question types by combining RAG retrieval
with IBM Granite text generation.
"""

import logging
import re
from typing import List, Optional

from config.settings import settings
from prompts import load_prompt
from rag.retriever import retrieve
from utils.profile_parser import UserProfile

logger = logging.getLogger(__name__)


# Default number of questions per type
_DEFAULT_COUNTS = {
    "technical": 5,
    "role_specific": 5,
    "hr": 5,
    "behavioral": 5,
}


class QuestionGenerator:
    """Generate interview questions using RAG + IBM Granite."""

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_technical(
        self,
        profile: UserProfile,
        num_questions: int = _DEFAULT_COUNTS["technical"],
        difficulty: Optional[str] = None,
    ) -> List[str]:
        """Generate technical questions for the candidate."""

        difficulty = difficulty or self._infer_difficulty(
            profile.experience_level
        )

        query = (
            f"technical interview questions {profile.role} "
            f"{profile.skills_str} {profile.experience_level}"
        )

        context = self._retrieve_context(
            query,
            profile.role,
        )

        prompt = load_prompt("technical_prompt").format(
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            num_questions=num_questions,
            difficulty=difficulty,
            context=context,
        )

        raw = self._generate(
            prompt,
            label="technical",
        )

        return self._generate_questions_with_retry(
            prompt,
            raw,
            num_questions,
            "technical",
            profile,
        )

    # ------------------------------------------------------------------

    def generate_role_specific(
        self,
        profile: UserProfile,
        num_questions: int = _DEFAULT_COUNTS["role_specific"],
    ) -> List[str]:
        """Generate role-specific scenario questions."""

        query = (
            f"role responsibilities interview questions "
            f"scenario {profile.role}"
        )

        context = self._retrieve_context(
            query,
            profile.role,
        )

        prompt = load_prompt("role_specific_prompt").format(
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            resume_summary=profile.resume_summary or "Not provided",
            num_questions=num_questions,
            context=context,
        )

        raw = self._generate(
            prompt,
            label="role_specific",
        )

        return self._generate_questions_with_retry(
            prompt,
            raw,
            num_questions,
            "role_specific",
            profile,
        )

    # ------------------------------------------------------------------

    def generate_hr(
        self,
        profile: UserProfile,
        num_questions: int = _DEFAULT_COUNTS["hr"],
    ) -> List[str]:
        """Generate HR and culture-fit questions."""

        query = (
            f"HR interview questions motivation strengths weaknesses "
            f"career goals {profile.role}"
        )

        context = self._retrieve_context(
            query,
            role_filter=None,
        )

        prompt = load_prompt("hr_prompt").format(
            name=profile.name,
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            resume_summary=profile.resume_summary or "Not provided",
            num_questions=num_questions,
            context=context,
        )

        raw = self._generate(
            prompt,
            label="hr",
        )

        return self._generate_questions_with_retry(
            prompt,
            raw,
            num_questions,
            "hr",
            profile,
        )

    # ------------------------------------------------------------------

    def generate_behavioral(
        self,
        profile: UserProfile,
        num_questions: int = _DEFAULT_COUNTS["behavioral"],
    ) -> List[str]:
        """Generate STAR behavioral questions."""

        query = (
            f"behavioral interview questions STAR leadership teamwork "
            f"conflict problem solving {profile.role}"
        )

        context = self._retrieve_context(
            query,
            role_filter=None,
        )

        prompt = load_prompt("behavioral_prompt").format(
            role=profile.role,
            experience=profile.experience_level,
            skills=profile.skills_str,
            resume_summary=profile.resume_summary or "Not provided",
            num_questions=num_questions,
            context=context,
        )

        raw = self._generate(
            prompt,
            label="behavioral",
        )

        return self._generate_questions_with_retry(
            prompt,
            raw,
            num_questions,
            "behavioral",
            profile,
        )

    # ------------------------------------------------------------------

    def generate_all(
        self,
        profile: UserProfile,
        counts: Optional[dict] = None,
    ) -> dict:
        """Generate all four interview question types."""

        c = dict(_DEFAULT_COUNTS)

        if counts:
            c.update(counts)

        return {
            "technical": self.generate_technical(
                profile,
                c["technical"],
            ),
            "role_specific": self.generate_role_specific(
                profile,
                c["role_specific"],
            ),
            "hr": self.generate_hr(
                profile,
                c["hr"],
            ),
            "behavioral": self.generate_behavioral(
                profile,
                c["behavioral"],
            ),
        }

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    def _retrieve_context(
        self,
        query: str,
        role_filter: Optional[str] = None,
    ) -> str:
        """Retrieve RAG context and concatenate chunk texts."""

        try:
            docs = retrieve(
                query,
                role_filter=role_filter,
                k=settings.TOP_K_RETRIEVAL,
            )

            if not docs:
                return "No additional context retrieved."

            return "\n\n".join(
                d.page_content
                for d in docs
            )

        except Exception as exc:
            logger.warning(
                "RAG retrieval failed: %s — continuing without context.",
                exc,
            )

            return "Context unavailable."

    # ------------------------------------------------------------------
    # Granite generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        label: str = "",
    ) -> str:
        """Call Granite and return the raw response text."""

        try:
            raw = self._llm.generate(
                prompt,
                max_new_tokens=500,
                temperature=0.25,
            )
            raw_text = raw if isinstance(raw, str) else str(raw)
            logger.info(
                "Granite raw output for '%s': %s",
                label,
                repr(raw_text[:2000]),
            )
            return raw_text

        except Exception as exc:
            logger.error(
                "Granite generation failed for '%s': %s",
                label,
                exc,
            )

            raise RuntimeError(
                f"Question generation failed ({label}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Robust generation + retry
    # ------------------------------------------------------------------

    def _generate_questions_with_retry(
        self,
        prompt: str,
        raw: str,
        expected: int,
        label: str,
        profile: UserProfile,
    ) -> List[str]:
        """Parse Granite output and retry once if formatting is unusable."""
        questions = self._parse_numbered_list(raw, expected)

        if len(questions) >= expected:
            return questions[:expected]

        logger.warning(
            "%s parser found %d/%d questions. Retrying with strict format.",
            label,
            len(questions),
            expected,
        )

        retry_prompt = f"""
{prompt}

IMPORTANT OUTPUT FORMAT:
Return exactly {expected} interview questions.

Use ONLY:
1. First question
2. Second question
3. Third question

Continue until exactly {expected} questions are present.

Do NOT use:
- bullet points
- JSON
- a Python list
- `questions = [...]`
- headings
- explanations
- competency labels

Return ONLY the numbered questions.
""".strip()

        try:
            retry_raw = self._generate(
                retry_prompt,
                label=f"{label}_retry",
            )
            retry_questions = self._parse_numbered_list(
                retry_raw,
                expected,
            )
            if retry_questions:
                return retry_questions[:expected]
        except Exception as exc:
            logger.warning(
                "Retry generation failed for '%s': %s",
                label,
                exc,
            )

        # Last safe attempt: parse bullets or quoted list items.
        alternative = self._parse_alternative_list(raw, expected)
        if alternative:
            return alternative[:expected]

        # Final deterministic fallback so the UI never loses a category
        # merely because Granite returned an unexpected text format.
        return self._fallback_questions(label, profile, expected)

    @staticmethod
    def _fallback_questions(label: str, profile: UserProfile, expected: int) -> List[str]:
        role = profile.role or "Software Engineer"
        skills = profile.skills_str or "your technical skills"
        fallback = {
            "technical": [
                f"Explain the key concepts you would need to know for a {role} role, and how you would apply them using {skills}.",
                f"How would you approach debugging a problem in a {role} application, and what steps would you follow to identify the root cause?",
                f"How would you design a simple solution for a {role} problem while considering correctness, performance, and maintainability?",
            ],
            "role_specific": [
                f"What would you consider your main responsibilities when starting in a {role} position?",
                f"If you were given an unfamiliar task in a {role} role, how would you understand the requirement and decide how to implement it?",
                f"How would you collaborate with developers or other stakeholders to deliver a feature successfully in a {role} role?",
            ],
            "hr": [
                f"Why are you interested in starting your career as a {role}?",
                "What are your key strengths, and how would they help you contribute as a fresher?",
                "What are you looking for in your first professional role, and how do you plan to continue learning?",
            ],
            "behavioral": [
                "Tell me about a time during your studies when you faced a difficult problem. How did you approach solving it?",
                "Tell me about a time when you had to learn something new to complete an academic or personal project. How did you handle it?",
                "Describe a time when you received feedback on your work. How did you respond and what did you learn from it?",
            ],
        }
        return fallback.get(label, fallback["behavioral"])[:expected]

    # ------------------------------------------------------------------
    # Question parser
    # ------------------------------------------------------------------


    @staticmethod
    def _parse_numbered_list(
        text: str,
        expected: int,
    ) -> List[str]:
        """
        Extract clean numbered questions from Granite output.

        Supported formats:

            1. Question
            1) Question
            Q1. Question
            Q1) Question

        The parser also supports questions spanning multiple lines.

        It deliberately does NOT use arbitrary-line fallback parsing,
        because that can accidentally turn model artifacts such as
        "questions = [" into a question.
        """

        if not text:
            return []

        # --------------------------------------------------------------
        # Clean common generation artifacts
        # --------------------------------------------------------------

        text = text.replace("```python", "")
        text = text.replace("```sql", "")
        text = text.replace("```markdown", "")
        text = text.replace("```text", "")
        text = text.replace("```", "")
        text = text.replace("svgsvg", "")

        text = text.strip()

        # Remove accidental Python-list wrapper.
        text = re.sub(
            r"^\s*questions\s*=\s*\[\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove an accidental closing Python-list bracket.
        text = re.sub(
            r"\s*\]\s*$",
            "",
            text,
        ).strip()

        # Remove obvious heading lines.
        cleaned_lines = []

        ignored_lines = {
            "begin:",
            "questions:",
            "technical questions:",
            "hr questions:",
            "behavioral questions:",
            "role-specific questions:",
            "role specific questions:",
        }

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append("")
                continue

            if stripped.lower() in ignored_lines:
                continue

            if stripped.startswith("```"):
                continue

            if stripped.lower() == "svgsvg":
                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines).strip()

        if not text:
            return []

        # --------------------------------------------------------------
        # Main parser
        #
        # Capture everything after a question number until the next
        # question number or end of text.
        # --------------------------------------------------------------

        pattern = re.compile(
            r"^\s*(?:Q\s*)?\d+\s*[\.\):\-]\s*"
            r"(.*?)"
            r"(?=^\s*(?:Q\s*)?\d+\s*[\.\):\-]\s*|\Z)",
            re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )

        questions: List[str] = []

        for match in pattern.finditer(text):

            question = match.group(1).strip()

            # Convert multi-line questions into clean readable text.
            question = re.sub(
                r"\s+",
                " ",
                question,
            ).strip()

            # Remove accidental Markdown/code markers.
            question = re.sub(
                r"^`+|`+$",
                "",
                question,
            ).strip()

            question = re.sub(
                r"^\*+|\*+$",
                "",
                question,
            ).strip()

            # Reject obvious artifacts.
            if not question:
                continue

            if question.lower() in {
                "questions",
                "questions =",
                "questions = [",
            }:
                continue

            if question.lower().startswith("questions ="):
                continue

            questions.append(question)

            if len(questions) >= expected:
                break

        # --------------------------------------------------------------
        # Final safety filtering
        # --------------------------------------------------------------

        valid_questions = []

        for question in questions:

            lower = question.lower()

            # Never allow Python-list artifacts.
            if lower.startswith("questions ="):
                continue

            # Never allow obvious output labels.
            if lower in {
                "begin",
                "begin:",
                "questions",
                "questions:",
            }:
                continue

            # Avoid very short garbage outputs.
            if len(question) < 15:
                continue

            valid_questions.append(question)

            if len(valid_questions) >= expected:
                break

        return valid_questions

    @staticmethod
    def _parse_alternative_list(
        text: str,
        expected: int,
    ) -> List[str]:
        """Safely parse bullet or quoted-list output without arbitrary fallback."""
        if not text:
            return []

        text = re.sub(
            r"```(?:python|json|markdown|text|sql)?|```",
            "",
            str(text),
            flags=re.IGNORECASE,
        )
        text = text.replace("svgsvg", "").strip()

        candidates: List[str] = []

        # Markdown bullets.
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^[-*•]\s+", line):
                candidates.append(re.sub(r"^[-*•]\s+", "", line))

        # Python/JSON-style quoted strings.
        candidates.extend(
            re.findall(r"""["']([^"']{15,300})["']""", text)
        )

        result: List[str] = []
        seen = set()

        for question in candidates:
            question = re.sub(r"\s+", " ", question).strip()
            question = question.strip("`\"' ,")

            if len(question) < 15:
                continue

            lower = question.lower()

            if lower.startswith(
                ("questions =", "questions:", "answer:", "model answer:")
            ):
                continue

            if "?" not in question and not lower.startswith(
                (
                    "what ", "why ", "how ", "when ", "where ", "which ",
                    "explain ", "describe ", "tell me ", "walk me ",
                    "can you ", "could you ", "would you ", "have you ",
                    "do you ", "if ",
                )
            ):
                continue

            key = lower
            if key in seen:
                continue

            seen.add(key)
            result.append(question)

            if len(result) >= expected:
                break

        return result

    # ------------------------------------------------------------------
    # Difficulty inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_difficulty(
        experience_level: str,
    ) -> str:
        """Map experience level to a difficulty label."""

        level_lower = experience_level.lower()

        if (
            "fresher" in level_lower
            or "0-1" in level_lower
        ):
            return "basic"

        if (
            "junior" in level_lower
            or "1-3" in level_lower
        ):
            return "basic to intermediate"

        if (
            "senior" in level_lower
            or "6-10" in level_lower
        ):
            return "advanced"

        if (
            "lead" in level_lower
            or "principal" in level_lower
            or "10+" in level_lower
        ):
            return "advanced"

        return "intermediate"