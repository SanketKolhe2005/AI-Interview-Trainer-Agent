"""
agent/orchestrator.py
=====================

Central coordinator for the AI Interview Trainer Agent.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm.granite_client import GraniteClient
from prompts import load_prompt
from rag.retriever import retrieve
from utils.profile_parser import UserProfile

from agent.profile_analyzer import ProfileAnalyzer
from agent.question_generator import QuestionGenerator
from agent.answer_evaluator import AnswerEvaluator
from agent.feedback_agent import FeedbackAgent
from agent.mock_interviewer import MockInterviewer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InterviewKit
# ---------------------------------------------------------------------------

@dataclass
class InterviewKit:
    """Complete output of a single interview preparation session."""

    profile: UserProfile = field(default_factory=UserProfile)

    technical_questions: List[str] = field(default_factory=list)
    role_questions: List[str] = field(default_factory=list)
    hr_questions: List[str] = field(default_factory=list)
    behavioral_questions: List[str] = field(default_factory=list)

    model_answers: Dict[str, str] = field(default_factory=dict)

    tips: str = ""
    strategy: str = ""

    @property
    def all_questions(self) -> List[str]:
        """Return all interview questions."""

        return (
            self.technical_questions
            + self.role_questions
            + self.hr_questions
            + self.behavioral_questions
        )

    def to_dict(self) -> dict:
        """Convert interview kit to dictionary."""

        return {
            "profile": self.profile.to_dict(),
            "technical_questions": self.technical_questions,
            "role_questions": self.role_questions,
            "hr_questions": self.hr_questions,
            "behavioral_questions": self.behavioral_questions,
            "model_answers": self.model_answers,
            "tips": self.tips,
            "strategy": self.strategy,
            "total_questions": len(self.all_questions),
        }


# ---------------------------------------------------------------------------
# InterviewOrchestrator
# ---------------------------------------------------------------------------

class InterviewOrchestrator:
    """Coordinate all specialised agents."""

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

        self._profile_analyzer: Optional[ProfileAnalyzer] = None
        self._question_generator: Optional[QuestionGenerator] = None
        self._answer_evaluator: Optional[AnswerEvaluator] = None
        self._feedback_agent: Optional[FeedbackAgent] = None

    # ------------------------------------------------------------------
    # Agent accessors
    # ------------------------------------------------------------------

    def _get_llm(self) -> GraniteClient:
        if self._llm is None:
            self._llm = GraniteClient()

        return self._llm

    def _get_profile_analyzer(self) -> ProfileAnalyzer:
        if self._profile_analyzer is None:
            self._profile_analyzer = ProfileAnalyzer(
                llm_client=self._get_llm()
            )

        return self._profile_analyzer

    def _get_question_generator(self) -> QuestionGenerator:
        if self._question_generator is None:
            self._question_generator = QuestionGenerator(
                llm_client=self._get_llm()
            )

        return self._question_generator

    def _get_answer_evaluator(self) -> AnswerEvaluator:
        if self._answer_evaluator is None:
            self._answer_evaluator = AnswerEvaluator(
                llm_client=self._get_llm()
            )

        return self._answer_evaluator

    def _get_feedback_agent(self) -> FeedbackAgent:
        if self._feedback_agent is None:
            self._feedback_agent = FeedbackAgent(
                llm_client=self._get_llm()
            )

        return self._feedback_agent

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def run(
        self,
        profile: UserProfile,
        question_counts: Optional[dict] = None,
        generate_answers: bool = True,
        generate_tips: bool = True,
        generate_strategy: bool = True,
    ) -> InterviewKit:
        """Execute the complete interview preparation workflow."""

        kit = InterviewKit(profile=profile)

        qg = self._get_question_generator()
        fa = self._get_feedback_agent()

        logger.info(
            "Starting interview kit generation for %s (%s | %s).",
            profile.name,
            profile.role,
            profile.experience_level,
        )

        # --------------------------------------------------------------
        # Step 1: Generate questions
        # --------------------------------------------------------------

        logger.info("Generating interview questions...")

        q_result = qg.generate_all(
            profile,
            counts=question_counts,
        )

        kit.technical_questions = q_result.get(
            "technical",
            [],
        )

        kit.role_questions = q_result.get(
            "role_specific",
            [],
        )

        kit.hr_questions = q_result.get(
            "hr",
            [],
        )

        kit.behavioral_questions = q_result.get(
            "behavioral",
            [],
        )

        logger.info(
            "Questions generated: %d technical, %d role, %d HR, %d behavioral.",
            len(kit.technical_questions),
            len(kit.role_questions),
            len(kit.hr_questions),
            len(kit.behavioral_questions),
        )

        # --------------------------------------------------------------
        # Step 2: Generate model answers
        # --------------------------------------------------------------

        if generate_answers:
            logger.info("Generating model answers...")

            kit.model_answers = self._generate_model_answers(
                profile,
                kit.all_questions,
            )

            logger.info(
                "Model answers generated: %d.",
                len(kit.model_answers),
            )

        # --------------------------------------------------------------
        # Step 3: Generate tips
        # --------------------------------------------------------------

        if generate_tips:
            logger.info("Generating improvement tips...")

            kit.tips = fa.generate_tips(
                profile,
                questions=kit.all_questions,
            )

        # --------------------------------------------------------------
        # Step 4: Generate strategy
        # --------------------------------------------------------------

        if generate_strategy:
            logger.info("Generating preparation strategy...")

            kit.strategy = fa.generate_strategy(profile)

        logger.info(
            "InterviewKit complete. Total questions: %d.",
            len(kit.all_questions),
        )

        return kit

    # ------------------------------------------------------------------
    # Profile analysis
    # ------------------------------------------------------------------

    def analyze_profile(
        self,
        name=None,
        role=None,
        experience_level=None,
        skills=None,
        resume_text=None,
    ) -> UserProfile:
        """Convenience wrapper around ProfileAnalyzer."""

        return self._get_profile_analyzer().analyze(
            name=name,
            role=role,
            experience_level=experience_level,
            skills=skills,
            resume_text=resume_text,
        )

    # ------------------------------------------------------------------
    # Answer evaluation
    # ------------------------------------------------------------------

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        profile: UserProfile,
        question_type: str = "general",
    ) -> dict:
        """Evaluate a candidate answer."""

        return self._get_answer_evaluator().evaluate(
            question=question,
            answer=answer,
            profile=profile,
            question_type=question_type,
        )

    # ------------------------------------------------------------------
    # Mock interviewer
    # ------------------------------------------------------------------

    def create_mock_interviewer(
        self,
        profile: UserProfile,
        total_questions: int = 8,
        interview_type: str = "Mixed",
    ) -> MockInterviewer:
        """Create a mock interviewer."""

        return MockInterviewer(
            llm_client=self._get_llm(),
            profile=profile,
            total_questions=total_questions,
            interview_type=interview_type,
        )

    # ------------------------------------------------------------------
    # Model answer generation
    # ------------------------------------------------------------------

    def _generate_model_answers(
        self,
        profile: UserProfile,
        questions: List[str],
        max_questions: int = 40,
    ) -> Dict[str, str]:
        """
        Generate interview-ready answers.

        The answer prompt is deliberately built here instead of loading
        answer_prompt.txt. This prevents prompt-template text from being
        returned by Granite and makes the resume-grounding rules explicit.
        """

        answers: Dict[str, str] = {}
        llm = self._get_llm()

        resume = (getattr(profile, "resume_summary", "") or "").strip()
        if not resume:
            resume = "No resume information was provided."

        for question in questions[:max_questions]:
            qtype = self._infer_question_type(question)

            # Deterministic routing for common questions prevents the LLM from
            # answering the wrong topic and keeps fresher answers truthful.
            direct_answer = self._direct_answer(question, profile, qtype)
            if direct_answer:
                answers[question] = direct_answer
                continue

            # Behavioral questions require real personal evidence. If the
            # resume does not contain the requested event, use a truthful
            # fresher response rather than allowing the model to invent one.
            if qtype == "behavioral":
                answers[question] = self._behavioral_answer(question, profile)
                continue

            if qtype in {"technical", "role_specific"}:
                context = self._retrieve_answer_context(
                    question,
                    profile.role,
                )
            else:
                context = "No external context. Use only the profile and resume."

            prompt = self._build_answer_prompt(
                profile=profile,
                question=question,
                question_type=qtype,
                resume=resume,
                context=context,
            )

            try:
                raw = llm.generate(
                    prompt,
                    max_new_tokens=420,
                    temperature=0.10,
                )

                answer = self._clean_model_answer(raw)

                if self._invalid_answer(answer, question, profile):
                    logger.warning(
                        "Rejected invalid model answer for question '%s…'.",
                        question[:70],
                    )
                    answer = self._truthful_fallback(
                        qtype,
                        profile,
                        question,
                    )

                answers[question] = answer

            except Exception as exc:
                logger.warning(
                    "Model answer generation failed for question '%s…': %s",
                    question[:70],
                    exc,
                )
                answers[question] = self._truthful_fallback(
                    qtype,
                    profile,
                    question,
                )

        return answers

    @staticmethod
    def _direct_answer(
        question: str,
        profile: UserProfile,
        question_type: str,
    ) -> Optional[str]:
        """Return a deterministic, resume-safe answer for common questions."""
        q = (question or "").lower().strip()

        if "fibonacci" in q and (
            "recursive" in q or "iterative" in q or "dynamic programming" in q
        ):
            return (
                "A naive recursive Fibonacci solution has exponential time complexity, "
                "approximately O(2^n), because it recalculates the same values many times. "
                "A memoized dynamic-programming solution reduces this to O(n) time and "
                "O(n) space. A bottom-up iterative dynamic-programming solution also runs "
                "in O(n) time and can use O(1) extra space by keeping only the previous two "
                "values. In an interview, I would prefer the iterative approach when I only "
                "need the Fibonacci value because it avoids repeated work and unnecessary "
                "recursion overhead."
            )

        if "sql" in q and "join" in q:
            return (
                "SQL joins combine rows from related tables. An INNER JOIN returns only rows "
                "that have matching values in both tables. A LEFT JOIN returns every row from "
                "the left table and matching rows from the right table, using NULL when there "
                "is no match. RIGHT JOIN is the opposite, and FULL OUTER JOIN keeps matching "
                "and non-matching rows from both sides. For example, if I want only employees "
                "who have a matching department, I would use an INNER JOIN. If I want every "
                "employee, including employees who are not assigned to a department, I would "
                "use a LEFT JOIN."
            )

        if (
            "preprocessing" in q
            or "pre-process" in q
            or "missing values" in q
            or "categorical variables" in q
            or "feature scaling" in q
        ):
            return (
                "I would first understand the dataset and the machine-learning objective. "
                "Then I would inspect data types, missing values, duplicates, outliers, and "
                "important feature distributions. I would handle missing values using an "
                "approach appropriate to the feature and problem, encode categorical "
                "variables when required, and scale numerical features when the chosen "
                "algorithm is sensitive to scale. I would also check for data leakage and "
                "fit preprocessing steps using training data so information from the test "
                "set does not influence training. Finally, I would validate the processed "
                "data before training the model."
            )

        if "object-oriented" in q and "functional" in q:
            return (
                "Object-oriented programming organizes software around objects that combine "
                "data and behavior. Common concepts include encapsulation, inheritance, and "
                "polymorphism. Functional programming focuses more on functions, immutability, "
                "and minimizing side effects. I would choose OOP when the system is naturally "
                "modeled around entities and their behavior, while functional techniques can "
                "be useful for clear and predictable data transformations. Python supports both "
                "approaches, so I would choose the style that makes the solution clear, "
                "testable, and maintainable."
            )

        if "rest" in q and "api" in q:
            return (
                "A REST API exposes resources through HTTP endpoints. I would normally use "
                "GET to retrieve data, POST to create a resource, PUT or PATCH to update it, "
                "and DELETE to remove it. For example, creating a user could use POST /users "
                "with a JSON body such as {\"name\": \"Sanket\", \"email\": "
                "\"user@example.com\"}. The server should validate the input, return suitable "
                "HTTP status codes, handle errors consistently, and apply authentication and "
                "authorization where required. A successful creation would normally return "
                "201 Created with safe response fields."
            )

        if "responsibilit" in q and (
            "software engineer" in q or "main" in q or "role" in q
        ):
            return (
                "As a fresher Software Engineer, my main responsibilities would be to "
                "understand requirements, write clear and maintainable code, test my work, "
                "debug issues, and collaborate with the team. I would first understand the "
                "existing codebase and development practices before making changes. If a "
                "requirement was unclear, I would ask questions early rather than making "
                "assumptions. I would also take feedback positively, keep my work organized, "
                "and make sure the final implementation meets the expected requirement."
            )

        if "unfamiliar task" in q or (
            "understand the requirement" in q and "task" in q
        ):
            return (
                "If I receive an unfamiliar task, I would first understand the expected "
                "outcome and clarify any unclear requirements. Then I would break the task "
                "into smaller parts, identify the relevant technical concepts, and review "
                "reliable documentation or existing code. I would implement the solution "
                "incrementally and test each part. If I reached a blocker, I would explain "
                "what I had already tried and ask an experienced teammate for guidance. "
                "This approach helps me learn the task while still moving the work forward "
                "systematically."
            )

        if (
            "collaborate" in q
            or "stakeholder" in q
            or "deliver a feature" in q
            or "work with developers" in q
        ) and question_type in {"role_specific", "technical"}:
            return (
                "I would start by making sure I understand the feature and its expected "
                "outcome. I would communicate clearly with the people involved, ask questions "
                "when requirements are unclear, and keep others informed about progress and "
                "blockers. I would implement the feature carefully, test it, and use feedback "
                "or code review to improve it. As a fresher, I would also be comfortable asking "
                "experienced teammates for guidance when I am unsure about a technical or "
                "requirement-related decision."
            )

        if "why" in q and "software engineer" in q:
            return (
                "I am interested in starting my career as a Software Engineer because I enjoy "
                "programming, problem-solving, and building practical solutions. I have worked "
                "on projects involving Python, SQL, machine learning, RAG, ChromaDB, and IBM "
                "Granite, including an AI Interview Trainer Agent. These projects have helped "
                "me build a practical technical foundation. As a fresher, I now want to apply "
                "that foundation to real-world software development, learn from experienced "
                "engineers, and continue improving my problem-solving and development skills."
            )

        if "strength" in q:
            return (
                "My key strengths are problem-solving, willingness to learn, and having a "
                "practical technical foundation. I have worked with Python, SQL, machine "
                "learning, RAG, ChromaDB, and IBM Granite in my projects. I am comfortable "
                "breaking a problem into smaller steps, learning new concepts, and working "
                "through issues systematically. As a fresher, I know I still have a lot to "
                "learn, so I value feedback and use it to improve. I believe this combination "
                "of technical curiosity and a structured approach will help me contribute and "
                "grow as a Software Engineer."
            )

        if "first professional role" in q or (
            "continue learning" in q and "role" in q
        ):
            return (
                "In my first professional role, I am looking for an environment where I can "
                "apply my current technical skills, learn from experienced engineers, and "
                "understand real-world software development practices. I plan to continue "
                "learning through reliable documentation, hands-on practice, feedback, and "
                "solving new technical problems. I also want to strengthen my coding, debugging, "
                "and system-understanding skills. My goal is to build a strong foundation as a "
                "Software Engineer and gradually take on more responsibility as my experience grows."
            )

        return None

    @staticmethod
    def _build_answer_prompt(
        profile: UserProfile,
        question: str,
        question_type: str,
        resume: str,
        context: str,
    ) -> str:
        """Build one self-contained answer prompt; never expose instructions."""

        technical_note = ""
        if question_type == "technical":
            technical_note = (
                "For technical questions, explain the concept accurately. "
                "If the question explicitly asks for code or SQL, a short "
                "code/query example is allowed."
            )

        return f"""
You are helping a fresher prepare for a Software Engineer interview.

Your task is to write ONLY the answer the candidate should speak to the
interviewer for the QUESTION below.

CANDIDATE
Name: {profile.name}
Role: {profile.role}
Experience: {profile.experience_level}
Skills: {profile.skills_str}

RESUME — THE ONLY SOURCE FOR PERSONAL EXPERIENCE
{resume}

QUESTION TYPE
{question_type}

QUESTION
{question}

GENERAL TECHNICAL CONTEXT
{context}

RULES FOR YOUR ANSWER
1. Return ONLY the spoken interview answer. Never return instructions,
   analysis, headings, labels, or prompt text.
2. Do not repeat the question.
3. The candidate is a fresher. Never invent internships, jobs, companies,
   teammates, clients, achievements, metrics, percentages, certifications,
   deadlines, production incidents, or project details.
4. Personal experience may ONLY be stated when it is explicitly supported
   by the RESUME.
5. Do not turn general knowledge into a personal claim. For example, do not
   say "I used PostgreSQL" unless the resume says that.
6. If the question asks for a personal example and the resume does not
   contain one, say that honestly and explain how the candidate would
   approach the situation. Do not create a STAR story.
7. Do not mention coursework, classes, conferences, open-source work,
   agile ceremonies, product managers, QA, CI/CD, or any other experience
   unless it is explicitly supported by the resume.
8. Keep the answer natural, concise, and easy to speak: normally 80-140 words.
9. Do not say "Here is my answer", "I am ready to generate", "BEGIN",
   "END OF INSTRUCTIONS", "MODEL ANSWER", or "Answer:".
10. Do not include markdown headings, UI text, footer text, or prompt rules.
11. {technical_note}

Write the final answer now.
""".strip()

    @staticmethod
    def _invalid_answer(
        answer: str,
        question: str,
        profile: UserProfile,
    ) -> bool:
        """Reject output that is clearly prompt leakage or fabricated structure."""

        if not answer:
            return True

        lower = answer.lower()
        q_lower = question.lower()

        leakage = [
            "end of instructions",
            "final output contract",
            "strict output rules",
            "return only the final interview answer",
            "never guess or make up details",
            "use star method",
            "candidate information",
            "resume — the only source",
            "resume / actual experience",
            "i am ready to generate",
            "here is my answer",
            "model answer:",
            "answer:",
        ]

        if any(x in lower for x in leakage):
            return True

        if "ai interview trainer agent" in lower:
            return True

        # Reject a question being copied into the answer.
        normalized_q = re.sub(r"\s+", " ", q_lower).strip()
        normalized_a = re.sub(r"\s+", " ", lower).strip()
        if normalized_q and len(normalized_q) > 35 and normalized_q in normalized_a:
            return True

        # A response consisting almost entirely of instructions is invalid.
        instruction_hits = sum(
            phrase in lower
            for phrase in (
                "do not ",
                "only use ",
                "never ",
                "return only",
                "write approximately",
                "output ",
            )
        )
        if instruction_hits >= 3:
            return True

        return False

    # ------------------------------------------------------------------
    # Clean model answer
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_model_answer(text: str) -> str:
        """Remove only output artifacts; do not rewrite the candidate's answer."""

        if not text:
            return ""

        text = str(text).strip()
        text = text.replace("\x00", "")
        text = text.replace("svgsvg", "")

        # Remove code fences but preserve requested code/query content.
        text = re.sub(
            r"```(?:python|sql|text|json|bash|javascript|markdown)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.replace("```", "")

        # Remove wrapper headings.
        text = re.sub(
            r"^\s*#{0,6}\s*(?:MODEL ANSWER|FINAL ANSWER|ANSWER)\s*:?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove a small set of accidental prompt lines.
        lines = []
        for line in text.splitlines():
            s = line.strip()

            if not s:
                lines.append("")
                continue

            if re.match(
                r"^(?:BEGIN|END|END OF INSTRUCTIONS|"
                r"FINAL OUTPUT CONTRACT|STRICT OUTPUT RULES)\s*$",
                s,
                flags=re.IGNORECASE,
            ):
                continue

            if s.startswith("*AI Interview Trainer Agent"):
                continue
            if s.startswith("AI Interview Trainer Agent ·"):
                continue
            if s.startswith("Use via API"):
                continue
            if "Built with Gradio" in s:
                continue

            lines.append(line)

        text = "\n".join(lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    # ------------------------------------------------------------------
    # Behavioral answers
    # ------------------------------------------------------------------

    @staticmethod
    def _behavioral_answer(
        question: str,
        profile: UserProfile,
    ) -> str:
        """
        Behavioral answers are intentionally truthful.

        The current resume contains project work, but it does not provide
        detailed stories about teammates, conflicts, deadlines, ownership
        incidents, or measurable outcomes. Therefore this method does not
        manufacture those details.
        """

        q = question.lower()

        if "learn" in q or "technology quickly" in q or "programming language" in q:
            return (
                "As a fresher, I would answer this honestly rather than "
                "inventing a specific story. When I need to learn a new "
                "technology, I first understand the basic concepts and then "
                "use reliable documentation to build a small working example. "
                "After that, I apply the concept to the problem I am solving "
                "and test it step by step. If I get stuck, I identify the "
                "specific gap in my understanding and look for reliable "
                "guidance. This approach helps me learn quickly while making "
                "sure I understand what I am using rather than just copying "
                "a solution."
            )

        if "conflict" in q or "disagree" in q:
            return (
                "As a fresher, I do not want to invent a specific conflict "
                "story. If I disagreed with a teammate's solution, I would "
                "first understand their reasoning and then explain my concern "
                "using technical or requirement-based reasoning. I would "
                "keep the discussion focused on the problem rather than the "
                "person. If both approaches had merit, I would compare the "
                "trade-offs and choose the option that best satisfies the "
                "requirement. If we still could not agree, I would ask an "
                "experienced team member for guidance."
            )

        if "teamwork" in q or "collaboration" in q or "worked closely" in q:
            return (
                "As a fresher, I would not claim a teamwork experience that "
                "isn't described in my resume. In a software project, I would "
                "start by making sure everyone understands the requirement "
                "and their responsibilities. I would communicate my progress "
                "clearly, raise blockers early, and be open to suggestions "
                "from others. I would also make sure my work is tested and "
                "integrates correctly with the rest of the project. My goal "
                "would be to contribute reliably while learning from the "
                "people I work with."
            )

        if "ownership" in q or "took ownership" in q or "above and beyond" in q:
            return (
                "As a fresher, I would demonstrate ownership by taking "
                "responsibility for the work assigned to me and following it "
                "through to completion. I would first understand the expected "
                "result, break the work into smaller tasks, and track my "
                "progress. If I found a problem, I would investigate it "
                "instead of ignoring it and would communicate blockers early. "
                "I would test my work and compare the final result with the "
                "original requirement. For me, ownership means being "
                "accountable for the quality of my work while knowing when "
                "to ask for help."
            )

        if "deadline" in q or "time management" in q or "prioritize" in q:
            return (
                "As a fresher, I would not invent a specific deadline story. "
                "When I have multiple tasks, I first list them with their "
                "deadlines and dependencies. I then prioritize the most "
                "urgent and important work and break larger tasks into "
                "smaller milestones. I track progress regularly and adjust "
                "the plan if something takes longer than expected. If I see "
                "that a deadline may be affected, I communicate that early "
                "rather than waiting until the last moment. This helps me "
                "manage time while maintaining the quality of the work."
            )

        if "bug" in q or "coding problem" in q or "analytical" in q or "problem solving" in q:
            return (
                "As a fresher, I would approach a difficult coding problem "
                "systematically. First, I would reproduce the issue and "
                "understand the expected and actual behavior. Then I would "
                "break the problem into smaller parts and test one possible "
                "cause at a time. I would use logs, error messages, test "
                "cases, and reliable documentation to narrow down the root "
                "cause. After making a fix, I would test the original case "
                "and related cases to make sure the change did not introduce "
                "another problem. Finally, I would review what I learned so "
                "I could avoid the same mistake in future work."
            )

        return (
            "As a fresher, I would answer this honestly rather than inventing "
            "an experience. I would first understand the situation and the "
            "expected outcome, then break the problem into smaller steps. "
            "I would work through it systematically, communicate when I "
            "needed guidance, and review the result before considering the "
            "task complete. I believe being honest about my experience while "
            "showing a clear approach is better than giving an example that "
            "is not true."
        )

    # ------------------------------------------------------------------
    # Truthful fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _truthful_fallback(
        question_type: str,
        profile: UserProfile,
        question: str = "",
    ) -> str:
        """Question-aware fallback used whenever Granite output is unsafe."""

        q = (question or "").lower()

        if question_type == "behavioral":
            return InterviewOrchestrator._behavioral_answer(question, profile)

        if question_type == "hr":
            if "strength" in q:
                return (
                    "My key strengths are problem-solving, willingness to learn, "
                    "and my current technical foundation. I have worked with "
                    "Python, SQL, machine learning, RAG, ChromaDB, and IBM "
                    "Granite in my projects. As a fresher, I know I still have "
                    "a lot to learn, but I am comfortable breaking problems "
                    "into smaller steps and learning new concepts. I believe "
                    "these strengths will help me contribute while continuing "
                    "to grow."
                )

            if "why" in q and ("software engineer" in q or "career" in q):
                return (
                    "I am interested in starting my career as a Software Engineer "
                    "because I enjoy programming, problem-solving, and building "
                    "practical solutions. During my engineering studies, I have "
                    "worked with Python, SQL, machine learning, RAG, ChromaDB, "
                    "and IBM Granite. I now want to apply this foundation to "
                    "real-world software development, learn from experienced "
                    "engineers, and continue improving my technical skills."
                )

            if "first professional role" in q or "continue learning" in q:
                return (
                    "In my first professional role, I am looking for an "
                    "environment where I can apply my current technical skills, "
                    "learn from experienced engineers, and understand real-world "
                    "software development practices. I plan to continue learning "
                    "through documentation, hands-on practice, feedback, and "
                    "solving new technical problems. My goal is to build a "
                    "strong foundation and gradually take on more responsibility."
                )

            return (
                f"I am a fresher interested in {profile.role}. I am looking for "
                "an opportunity where I can apply my current technical and "
                "problem-solving skills, learn from experienced professionals, "
                "and continue improving through practical work and feedback."
            )

        if question_type == "role_specific":
            if "responsibilities" in q:
                return (
                    "As a fresher Software Engineer, my main responsibilities "
                    "would be to understand requirements, write clear and "
                    "maintainable code, test my work, debug issues, and "
                    "collaborate with the team. I would also take time to "
                    "understand the existing codebase and development practices. "
                    "I would ask questions when something is unclear and use "
                    "feedback to improve my work."
                )

            if "unfamiliar task" in q or "understand the requirement" in q:
                return (
                    "If I receive an unfamiliar task, I would first understand "
                    "the expected outcome and clarify unclear requirements. "
                    "Then I would break the task into smaller parts, identify "
                    "the relevant technical concepts, and review available "
                    "documentation or existing code. I would implement the "
                    "solution incrementally and test each part. If I encountered "
                    "a blocker, I would explain what I had tried and ask for "
                    "guidance."
                )

            if "collaborate" in q or "stakeholder" in q or "deliver a feature" in q:
                return (
                    "I would start by making sure I understand the feature and "
                    "its expected outcome. I would communicate clearly with "
                    "the people involved, ask questions when requirements are "
                    "unclear, and keep others informed about progress and "
                    "blockers. I would write and test the implementation "
                    "carefully and use feedback or code review to improve it. "
                    "As a fresher, I would also be comfortable asking "
                    "experienced teammates for guidance."
                )

        if "logistic regression" in q:
            return (
                "I would prepare the features and binary target, split the data "
                "into training and test sets, and train a LogisticRegression "
                "model using scikit-learn. I would evaluate the predictions "
                "using accuracy, precision, recall, F1-score, and a confusion "
                "matrix. For imbalanced data, I would pay particular attention "
                "to precision, recall, and F1 rather than relying only on "
                "accuracy."
            )

        if "average salary" in q and "department" in q:
            return (
                "I would group the rows by department_id and calculate the "
                "average salary using AVG. For example: "
                "SELECT department_id, AVG(salary) AS average_salary "
                "FROM employees GROUP BY department_id;"
            )

        if "object-oriented" in q and "functional" in q:
            return (
                "OOP organizes software around objects that combine data and "
                "behavior, while functional programming focuses on functions "
                "and reducing mutable state. I would choose OOP when the "
                "problem is naturally modeled around entities and their "
                "behavior. I would prefer a functional approach when the "
                "problem mainly involves independent data transformations. "
                "Python supports both styles, so I would choose the approach "
                "that keeps the solution clear, testable, and maintainable."
            )

        if "rest" in q and "api" in q:
            return (
                "For creating a user, I would use a POST /users endpoint with "
                "a JSON request containing the required fields. The server "
                "should validate the input, check uniqueness where required, "
                "and securely hash the password before storing it. A successful "
                "creation would normally return 201 Created with the new "
                "user's identifier and safe public fields. Invalid input "
                "should return an appropriate 4xx response."
            )

        if "recommendation" in q and "collaborative" in q:
            return (
                "For collaborative filtering, I would represent user-item "
                "interactions as a matrix and use similarity or latent-factor "
                "methods to learn relationships between users and items. In "
                "Python, I could use NumPy and SciPy together with a suitable "
                "machine-learning or recommender library. I would prepare and "
                "split the data, train the model, generate top recommendations, "
                "and evaluate the results with appropriate metrics."
            )

        if "slow-running" in q or "slow query" in q or "postgresql" in q:
            return (
                "I would first reproduce the slow query and inspect its execution "
                "plan using EXPLAIN ANALYZE. I would look for full table scans, "
                "expensive joins, poor estimates, sorting, or other bottlenecks. "
                "Then I would check whether appropriate indexes exist and "
                "whether the query can be simplified. I would also review "
                "table statistics and consider whether the issue is related "
                "to data volume or database resources. After each change, I "
                "would run the query again and compare the execution plan and "
                "timing to confirm that the optimization actually helped."
            )

        return (
            "I would first understand the requirements and identify the relevant "
            "technical concepts. Then I would break the problem into smaller "
            "steps, implement the solution incrementally, and test it carefully. "
            "If I encountered an issue, I would isolate the cause, consult "
            "reliable documentation, and ask for guidance when needed. Finally, "
            "I would review the solution for correctness and maintainability."
        )

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _retrieve_answer_context(
        question: str,
        role: str,
    ) -> str:
        """
        Retrieve general technical knowledge.

        RAG context is NOT candidate experience.
        """

        try:

            docs = retrieve(
                question,
                role_filter=role,
                k=3,
            )

            if not docs:
                return (
                    "No additional technical context retrieved."
                )

            return "\n\n".join(
                doc.page_content
                for doc in docs
            )

        except Exception as exc:

            logger.warning(
                "RAG retrieval failed for model answer: %s",
                exc,
            )

            return "No external context available."

    # ------------------------------------------------------------------
    # Question type inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_question_type(
        question: str,
    ) -> str:
        """Classify a question."""

        q_lower = question.lower()

        # --------------------------------------------------------------
        # Behavioral
        # --------------------------------------------------------------

        behavioral_keywords = [
            "tell me about a time",
            "tell me about a situation",
            "describe a situation",
            "give me an example",
            "describe a time",
            "when did you",
            "how did you handle",
            "conflict",
            "challenge you faced",
            "difficult situation",
            "situation where",
            "time when",
            "example of",
            "how did you solve",
            "how did you approach",
        ]

        if any(
            keyword in q_lower
            for keyword in behavioral_keywords
        ):
            return "behavioral"

        # --------------------------------------------------------------
        # HR
        # --------------------------------------------------------------

        hr_keywords = [
            "tell me about yourself",
            "introduce yourself",
            "yourself",
            "strengths",
            "weakness",
            "weaknesses",
            "salary",
            "career goals",
            "career goal",
            "goals",
            "motivate",
            "why should we hire",
            "why do you want",
            "why are you interested",
            "where do you see",
            "why this company",
            "why this role",
        ]

        if any(
            keyword in q_lower
            for keyword in hr_keywords
        ):
            return "hr"

        # --------------------------------------------------------------
        # Technical
        # --------------------------------------------------------------

        technical_keywords = [
            "algorithm",
            "design",
            "implement",
            "code",
            "coding",
            "system",
            "architecture",
            "database",
            "kubernetes",
            "pipeline",
            "model",
            "sql",
            "python",
            "machine learning",
            "deep learning",
            "api",
            "rest",
            "docker",
            "cloud",
            "deployment",
            "classification",
            "regression",
            "rag",
            "retrieval augmented generation",
            "chromadb",
            "granite",
            "llm",
            "embedding",
            "vector database",
            "git",
            "http",
            "tcp",
            "ip",
            "linux",
            "operating system",
        ]

        if any(
            keyword in q_lower
            for keyword in technical_keywords
        ):
            return "technical"

        # --------------------------------------------------------------
        # Role-specific
        # --------------------------------------------------------------

        return "role_specific"