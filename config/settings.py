"""
config/settings.py
==================
Centralised application configuration for the AI Interview Trainer Agent.

All values are loaded from environment variables (via a .env file).
Import this module anywhere in the project to access typed constants:

    from config.settings import settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """Application-wide configuration constants loaded from environment variables."""

    # ── IBM watsonx.ai ──────────────────────────────────────────────────────
    WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
    WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
    WATSONX_URL: str = os.getenv(
        "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
    )

    # ── IBM Granite model ────────────────────────────────────────────────────
    GRANITE_MODEL_ID: str = os.getenv(
        "GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct"
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── ChromaDB vector store ────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db")
    )
    CHROMA_COLLECTION_NAME: str = "interview_knowledge_base"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    # ── Text chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── LLM generation defaults ───────────────────────────────────────────────
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # ── Knowledge base paths ──────────────────────────────────────────────────
    KNOWLEDGE_BASE_DIR: Path = _PROJECT_ROOT / "knowledge_base"
    TECHNICAL_QUESTIONS_DIR: Path = KNOWLEDGE_BASE_DIR / "technical_questions"
    HR_QUESTIONS_DIR: Path = KNOWLEDGE_BASE_DIR / "hr_questions"
    BEHAVIORAL_QUESTIONS_DIR: Path = KNOWLEDGE_BASE_DIR / "behavioral_questions"
    ROLE_DESCRIPTIONS_DIR: Path = KNOWLEDGE_BASE_DIR / "role_descriptions"
    INDUSTRY_GUIDELINES_DIR: Path = KNOWLEDGE_BASE_DIR / "industry_guidelines"

    # ── Prompts directory ─────────────────────────────────────────────────────
    PROMPTS_DIR: Path = _PROJECT_ROOT / "prompts"

    # ── Supported job roles ───────────────────────────────────────────────────
    SUPPORTED_ROLES: list = [
        "Software Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "Product Manager",
        "Business Analyst",
    ]

    # ── Experience levels ─────────────────────────────────────────────────────
    EXPERIENCE_LEVELS: list = [
        "Fresher (0-1 years)",
        "Junior (1-3 years)",
        "Mid-level (3-6 years)",
        "Senior (6-10 years)",
        "Lead / Principal (10+ years)",
    ]

    def validate(self) -> None:
        """
        Raises ValueError if mandatory IBM watsonx.ai credentials are missing.
        Call this at application startup to fail fast on misconfiguration.
        """
        missing = []
        if not self.WATSONX_API_KEY:
            missing.append("WATSONX_API_KEY")
        if not self.WATSONX_PROJECT_ID:
            missing.append("WATSONX_PROJECT_ID")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in your IBM Cloud credentials."
            )


# Singleton instance — import this everywhere
settings = Settings()
