"""
tests/test_agent.py
===================
Unit tests for all agent modules.

The IBM Granite client is mocked throughout so no live credentials are needed.
RAG retrieval is also mocked to keep tests fast and deterministic.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, ".")

from utils.profile_parser import UserProfile, ProfileParser
from utils.formatter import (
    format_questions,
    format_all_questions,
    format_model_answers,
    format_tips,
    format_strategy,
    format_feedback,
    format_interview_kit_summary,
)
from agent.profile_analyzer import ProfileAnalyzer
from agent.question_generator import QuestionGenerator
from agent.answer_evaluator import AnswerEvaluator
from agent.feedback_agent import FeedbackAgent
from agent.mock_interviewer import MockInterviewer
from agent.orchestrator import InterviewOrchestrator, InterviewKit


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_profile(**kwargs) -> UserProfile:
    """Create a test UserProfile with sensible defaults."""
    defaults = dict(
        name="Test Candidate",
        role="Software Engineer",
        experience_level="Mid-level (3-6 years)",
        skills=["Python", "System Design", "SQL"],
        resume_summary="Backend engineer with 4 years experience.",
    )
    defaults.update(kwargs)
    return UserProfile(**defaults)


def _make_llm(response: str = "Mocked LLM response") -> MagicMock:
    """Return a mock GraniteClient whose generate() returns *response*."""
    mock = MagicMock()
    mock.generate.return_value = response
    return mock


def _make_retriever_patch(texts: list = None):
    """Return a patch for rag.retriever.retrieve returning mock Documents."""
    from langchain_core.documents import Document
    docs = [
        Document(page_content=t, metadata={"role": "Software Engineer", "category": "technical_questions"})
        for t in (texts or ["Mock retrieved context for testing."])
    ]
    return docs


# ---------------------------------------------------------------------------
# 1. UserProfile and ProfileParser
# ---------------------------------------------------------------------------

class TestUserProfile(unittest.TestCase):

    def test_default_profile(self):
        p = UserProfile()
        self.assertEqual(p.name, "Candidate")
        self.assertEqual(p.role, "Software Engineer")
        self.assertIsInstance(p.skills, list)

    def test_skills_str_property(self):
        p = UserProfile(skills=["Python", "Django"])
        self.assertEqual(p.skills_str, "Python, Django")

    def test_skills_str_empty(self):
        p = UserProfile(skills=[])
        self.assertEqual(p.skills_str, "Not specified")

    def test_to_dict(self):
        p = _make_profile()
        d = p.to_dict()
        self.assertIn("role", d)
        self.assertIn("skills", d)
        self.assertIn("skills_str", d)
        self.assertEqual(d["name"], "Test Candidate")


class TestProfileParser(unittest.TestCase):

    def setUp(self):
        self.parser = ProfileParser()

    def test_parse_valid_input(self):
        profile = self.parser.parse(
            name="Alice",
            role="Data Scientist",
            experience_level="Junior (1-3 years)",
            skills="Python, ML, SQL",
            resume_summary="Data science enthusiast.",
        )
        self.assertEqual(profile.name, "Alice")
        self.assertEqual(profile.role, "Data Scientist")
        self.assertIn("Python", profile.skills)
        self.assertIn("ML", profile.skills)

    def test_parse_empty_input_uses_defaults(self):
        profile = self.parser.parse()
        self.assertIsNotNone(profile.role)
        self.assertIsNotNone(profile.experience_level)

    def test_role_normalisation_partial_match(self):
        profile = self.parser.parse(role="devops")
        self.assertEqual(profile.role, "DevOps Engineer")

    def test_skills_from_list(self):
        profile = self.parser.parse(skills=["Java", "Spring", "Kubernetes"])
        self.assertEqual(len(profile.skills), 3)

    def test_skills_from_semicolon_string(self):
        profile = self.parser.parse(skills="Java; Spring; Kubernetes")
        self.assertIn("Java", profile.skills)
        self.assertIn("Spring", profile.skills)

    def test_skills_deduplication(self):
        profile = self.parser.parse(skills="Python, python, PYTHON")
        self.assertEqual(len(profile.skills), 1)

    def test_experience_normalisation(self):
        profile = self.parser.parse(experience_level="senior")
        self.assertIn("Senior", profile.experience_level)

    def test_unknown_role_falls_back_to_default(self):
        profile = self.parser.parse(role="Quantum Computing Wizard")
        self.assertEqual(profile.role, "Software Engineer")  # first supported role


# ---------------------------------------------------------------------------
# 2. Formatter utilities
# ---------------------------------------------------------------------------

class TestFormatter(unittest.TestCase):

    def test_format_questions_numbered(self):
        qs = ["What is Python?", "Explain OOP."]
        out = format_questions(qs, title="Test Questions")
        self.assertIn("1.", out)
        self.assertIn("2.", out)
        self.assertIn("Test Questions", out)

    def test_format_questions_empty(self):
        out = format_questions([])
        self.assertIn("No questions generated", out)

    def test_format_all_questions_structure(self):
        out = format_all_questions(["T1"], ["R1"], ["H1"], ["B1"])
        self.assertIn("Technical", out)
        self.assertIn("HR", out)
        self.assertIn("Behavioral", out)

    def test_format_model_answers(self):
        answers = {"What is OOP?": "OOP is object-oriented programming."}
        out = format_model_answers(answers)
        self.assertIn("What is OOP?", out)
        self.assertIn("OOP is object-oriented", out)

    def test_format_tips_non_empty(self):
        out = format_tips("Practise coding daily.")
        self.assertIn("Practise coding", out)

    def test_format_strategy_non_empty(self):
        out = format_strategy("Day 1: Review algorithms.")
        self.assertIn("Day 1", out)

    def test_format_feedback_non_empty(self):
        out = format_feedback("Your performance was strong.")
        self.assertIn("Your performance", out)

    def test_interview_kit_summary(self):
        kit_dict = {
            "profile": {"role": "Software Engineer", "experience_level": "Mid-level (3-6 years)"},
            "technical_questions": ["Q1", "Q2"],
            "role_questions": ["Q3"],
            "hr_questions": ["Q4"],
            "behavioral_questions": ["Q5"],
        }
        summary = format_interview_kit_summary(kit_dict)
        self.assertIn("Software Engineer", summary)
        self.assertIn("5 questions", summary)


# ---------------------------------------------------------------------------
# 3. ProfileAnalyzer
# ---------------------------------------------------------------------------

class TestProfileAnalyzer(unittest.TestCase):

    def test_analyze_without_llm(self):
        analyzer = ProfileAnalyzer(llm_client=None)
        profile = analyzer.analyze(
            name="Bob",
            role="Product Manager",
            experience_level="Senior (6-10 years)",
            skills="Roadmapping, Jira, Metrics",
        )
        self.assertEqual(profile.name, "Bob")
        self.assertEqual(profile.role, "Product Manager")

    def test_analyze_with_llm_extracts_summary(self):
        mock_llm = _make_llm("Experienced product manager with 8 years in SaaS.")
        analyzer = ProfileAnalyzer(llm_client=mock_llm)
        resume = "Bob Smith, Senior PM at Acme Corp. " * 20  # > 100 chars
        profile = analyzer.analyze(
            name="Bob",
            role="Product Manager",
            resume_text=resume,
        )
        # Granite was called for summary extraction
        mock_llm.generate.assert_called_once()
        self.assertIn("product manager", profile.resume_summary.lower())

    def test_analyze_llm_failure_falls_back_to_raw(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("API down")
        analyzer = ProfileAnalyzer(llm_client=mock_llm)
        resume = "Short but valid resume text for the candidate. " * 5
        profile = analyzer.analyze(name="Carol", role="Data Scientist", resume_text=resume)
        # Should not raise; resume_summary should be the truncated raw text
        self.assertTrue(len(profile.resume_summary) > 0)

    def test_get_supported_roles(self):
        roles = ProfileAnalyzer.get_supported_roles()
        self.assertIn("Software Engineer", roles)
        self.assertIn("Data Scientist", roles)

    def test_get_experience_levels(self):
        levels = ProfileAnalyzer.get_experience_levels()
        self.assertTrue(len(levels) >= 4)


# ---------------------------------------------------------------------------
# 4. QuestionGenerator
# ---------------------------------------------------------------------------

MOCK_QUESTIONS_TEXT = """1. Explain the difference between a process and a thread.
2. What is the CAP theorem?
3. Describe a binary search tree and its time complexity.
4. What is dependency injection?
5. How do hash maps handle collisions?"""


class TestQuestionGenerator(unittest.TestCase):

    def setUp(self):
        self.llm = _make_llm(MOCK_QUESTIONS_TEXT)
        self.profile = _make_profile()
        self.qg = QuestionGenerator(llm_client=self.llm)

    @patch("agent.question_generator.retrieve", return_value=_make_retriever_patch())
    def test_generate_technical_returns_list(self, mock_ret):
        questions = self.qg.generate_technical(self.profile, num_questions=5)
        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)
        self.llm.generate.assert_called_once()

    @patch("agent.question_generator.retrieve", return_value=_make_retriever_patch())
    def test_generate_hr_returns_list(self, mock_ret):
        questions = self.qg.generate_hr(self.profile, num_questions=5)
        self.assertIsInstance(questions, list)

    @patch("agent.question_generator.retrieve", return_value=_make_retriever_patch())
    def test_generate_behavioral_returns_list(self, mock_ret):
        questions = self.qg.generate_behavioral(self.profile, num_questions=5)
        self.assertIsInstance(questions, list)

    @patch("agent.question_generator.retrieve", return_value=_make_retriever_patch())
    def test_generate_role_specific_returns_list(self, mock_ret):
        questions = self.qg.generate_role_specific(self.profile, num_questions=5)
        self.assertIsInstance(questions, list)

    @patch("agent.question_generator.retrieve", return_value=_make_retriever_patch())
    def test_generate_all_returns_all_keys(self, mock_ret):
        result = self.qg.generate_all(self.profile)
        self.assertIn("technical", result)
        self.assertIn("role_specific", result)
        self.assertIn("hr", result)
        self.assertIn("behavioral", result)

    def test_parse_numbered_list_standard(self):
        text = "1. First question\n2. Second question\n3. Third question"
        result = QuestionGenerator._parse_numbered_list(text, 5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First question")

    def test_parse_numbered_list_fallback(self):
        text = "No numbers here\nJust lines\nLike this"
        result = QuestionGenerator._parse_numbered_list(text, 5)
        self.assertEqual(len(result), 3)

    def test_infer_difficulty_fresher(self):
        d = QuestionGenerator._infer_difficulty("Fresher (0-1 years)")
        self.assertEqual(d, "basic")

    def test_infer_difficulty_senior(self):
        d = QuestionGenerator._infer_difficulty("Senior (6-10 years)")
        self.assertEqual(d, "advanced")


# ---------------------------------------------------------------------------
# 5. AnswerEvaluator
# ---------------------------------------------------------------------------

MOCK_EVALUATION = """OVERALL SCORE: [7/10] — Good answer with minor gaps.

STRENGTHS:
- Clear explanation of the concept
- Used a practical example

WEAKNESSES:
- Did not mention time complexity
- Could elaborate on edge cases

MISSING POINTS:
- Big-O analysis
- Space complexity

TECHNICAL ACCURACY:
The technical content was mostly correct. Minor inaccuracy on hash collision resolution.

COMMUNICATION FEEDBACK:
Well structured and clear. Could be more concise.

IMPROVEMENT SUGGESTIONS:
1. Always include time complexity in technical answers.
2. Practise with concrete code examples.
3. Mention edge cases proactively.

MODEL ANSWER:
A hash map works by applying a hash function to the key to compute a bucket index..."""


class TestAnswerEvaluator(unittest.TestCase):

    def setUp(self):
        self.llm = _make_llm(MOCK_EVALUATION)
        self.profile = _make_profile()
        self.evaluator = AnswerEvaluator(llm_client=self.llm)

    @patch("agent.answer_evaluator.retrieve", return_value=_make_retriever_patch())
    def test_evaluate_returns_dict(self, mock_ret):
        result = self.evaluator.evaluate(
            question="How does a hash map work?",
            answer="A hash map uses a key-value structure with hashing.",
            profile=self.profile,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("strengths", result)
        self.assertIn("weaknesses", result)
        self.assertIn("model_answer", result)

    @patch("agent.answer_evaluator.retrieve", return_value=_make_retriever_patch())
    def test_evaluate_parses_score(self, mock_ret):
        result = self.evaluator.evaluate(
            question="Test question",
            answer="Test answer",
            profile=self.profile,
        )
        self.assertEqual(result["score"], "7/10")

    @patch("agent.answer_evaluator.retrieve", return_value=_make_retriever_patch())
    def test_evaluate_parses_strengths_list(self, mock_ret):
        result = self.evaluator.evaluate("Q", "A", self.profile)
        self.assertIsInstance(result["strengths"], list)
        self.assertGreater(len(result["strengths"]), 0)

    def test_evaluate_empty_answer_returns_empty(self):
        result = self.evaluator.evaluate("Some question?", "", self.profile)
        self.assertEqual(result["score"], "N/A")

    @patch("agent.answer_evaluator.retrieve", return_value=_make_retriever_patch())
    def test_evaluate_batch_equal_length(self, mock_ret):
        questions = ["Q1?", "Q2?"]
        answers   = ["A1",  "A2"]
        results = self.evaluator.evaluate_batch(questions, answers, self.profile)
        self.assertEqual(len(results), 2)

    def test_evaluate_batch_mismatched_raises(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_batch(["Q1?"], ["A1", "A2"], self.profile)


# ---------------------------------------------------------------------------
# 6. FeedbackAgent
# ---------------------------------------------------------------------------

class TestFeedbackAgent(unittest.TestCase):

    def setUp(self):
        self.llm = _make_llm("Great feedback report here.")
        self.profile = _make_profile()
        self.fa = FeedbackAgent(llm_client=self.llm)

    @patch("agent.feedback_agent.retrieve", return_value=_make_retriever_patch())
    def test_generate_feedback_calls_llm(self, mock_ret):
        result = self.fa.generate_feedback(self.profile)
        self.llm.generate.assert_called_once()
        self.assertIsInstance(result, str)
        self.assertIn("feedback", result.lower())

    @patch("agent.feedback_agent.retrieve", return_value=_make_retriever_patch())
    def test_generate_tips_returns_string(self, mock_ret):
        result = self.fa.generate_tips(self.profile, questions=["Q1?", "Q2?"])
        self.assertIsInstance(result, str)

    @patch("agent.feedback_agent.retrieve", return_value=_make_retriever_patch())
    def test_generate_strategy_returns_string(self, mock_ret):
        result = self.fa.generate_strategy(self.profile)
        self.assertIsInstance(result, str)

    def test_build_performance_context_with_evaluations(self):
        evals = [
            {"score": "7/10", "weaknesses": ["Did not mention complexity"]},
            {"score": "5/10", "weaknesses": ["Too vague"]},
        ]
        ctx = FeedbackAgent._build_performance_context(
            questions=["Q1?", "Q2?"],
            evaluations=evals,
            mock_history=None,
        )
        self.assertIn("7/10", ctx)
        self.assertIn("Did not mention", ctx)

    def test_build_performance_context_empty(self):
        ctx = FeedbackAgent._build_performance_context(None, None, None)
        self.assertIn("No detailed", ctx)


# ---------------------------------------------------------------------------
# 7. MockInterviewer
# ---------------------------------------------------------------------------

MOCK_FIRST_QUESTION = "[NEXT QUESTION]\nQuestion 1: Explain the difference between a list and a tuple in Python."
MOCK_EVALUATION_TURN = (
    "[EVALUATION]\nQuick Score: 8/10\n"
    "Well done: Clear and accurate.\nTo improve: Add time complexity.\n\n"
    "[NEXT QUESTION]\nQuestion 2: What is a decorator in Python?"
)
MOCK_DEBRIEF = (
    "[SESSION COMPLETE]\nOverall Score: 7.5/10\n"
    "Top Strengths: Technical depth | Clear examples\n"
    "Areas to Work On: Time complexity | Edge cases\n"
    "You're doing great!"
)


class TestMockInterviewer(unittest.TestCase):

    def _make_interviewer(self, responses=None) -> MockInterviewer:
        llm = MagicMock()
        if responses:
            llm.generate.side_effect = responses
        else:
            llm.generate.return_value = MOCK_FIRST_QUESTION
        profile = _make_profile()
        return MockInterviewer(llm_client=llm, profile=profile, total_questions=3)

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_start_session_returns_string(self, mock_ret):
        mi = self._make_interviewer()
        result = mi.start_session()
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_start_session_twice_raises(self, mock_ret):
        mi = self._make_interviewer()
        mi.start_session()
        with self.assertRaises(RuntimeError):
            mi.start_session()

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_submit_answer_before_start_raises(self, mock_ret):
        mi = self._make_interviewer()
        with self.assertRaises(RuntimeError):
            mi.submit_answer("My answer")

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_submit_answer_returns_dict(self, mock_ret):
        mi = self._make_interviewer(
            responses=[MOCK_FIRST_QUESTION, MOCK_EVALUATION_TURN]
        )
        mi.start_session()
        result = mi.submit_answer("Tuples are immutable, lists are mutable.")
        self.assertIsInstance(result, dict)
        self.assertIn("feedback", result)
        self.assertIn("is_complete", result)

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_session_completes_after_all_questions(self, mock_ret):
        responses = [
            MOCK_FIRST_QUESTION,
            MOCK_EVALUATION_TURN,
            MOCK_EVALUATION_TURN,
            MOCK_DEBRIEF,
        ]
        mi = self._make_interviewer(responses=responses)
        mi.start_session()
        mi.submit_answer("Answer 1")
        mi.submit_answer("Answer 2")
        result = mi.submit_answer("Answer 3")
        self.assertTrue(result["is_complete"])
        self.assertIn("debrief", result)

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_history_tracks_turns(self, mock_ret):
        mi = self._make_interviewer(responses=[MOCK_FIRST_QUESTION, MOCK_EVALUATION_TURN])
        mi.start_session()
        mi.submit_answer("My first answer")
        self.assertGreater(len(mi.history), 0)

    def test_parse_score_valid(self):
        text = "[EVALUATION]\nQuick Score: 8/10\nWell done."
        score = MockInterviewer._parse_score(text)
        self.assertEqual(score, 8.0)

    def test_parse_score_fallback(self):
        score = MockInterviewer._parse_score("No score here at all.")
        self.assertEqual(score, 0.0)

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_get_average_score(self, mock_ret):
        mi = self._make_interviewer()
        mi.start_session()
        mi.history[0].score = 8.0
        self.assertEqual(mi.get_average_score(), 8.0)


# ---------------------------------------------------------------------------
# 8. InterviewOrchestrator + InterviewKit
# ---------------------------------------------------------------------------

class TestInterviewKit(unittest.TestCase):

    def test_all_questions_combines_lists(self):
        kit = InterviewKit(
            profile=_make_profile(),
            technical_questions=["T1", "T2"],
            role_questions=["R1"],
            hr_questions=["H1"],
            behavioral_questions=["B1", "B2"],
        )
        self.assertEqual(len(kit.all_questions), 6)

    def test_to_dict_keys(self):
        kit = InterviewKit(profile=_make_profile())
        d = kit.to_dict()
        for key in ["profile", "technical_questions", "hr_questions",
                    "behavioral_questions", "model_answers", "tips", "strategy"]:
            self.assertIn(key, d)


class TestInterviewOrchestrator(unittest.TestCase):

    def _make_orchestrator(self):
        llm = _make_llm(MOCK_QUESTIONS_TEXT)
        orch = InterviewOrchestrator(llm_client=llm)
        return orch, llm

    @patch("agent.question_generator.retrieve", return_value=_make_retriever_patch())
    @patch("agent.feedback_agent.retrieve",     return_value=_make_retriever_patch())
    @patch("agent.orchestrator.retrieve",       return_value=_make_retriever_patch())
    def test_run_returns_interview_kit(self, m1, m2, m3):
        orch, llm = self._make_orchestrator()
        profile = _make_profile()
        kit = orch.run(
            profile=profile,
            generate_answers=True,
            generate_tips=True,
            generate_strategy=True,
        )
        self.assertIsInstance(kit, InterviewKit)
        # Granite was called (questions, answers, tips, strategy)
        self.assertGreater(llm.generate.call_count, 0)

    @patch("agent.question_generator.retrieve", return_value=_make_retriever_patch())
    @patch("agent.feedback_agent.retrieve",     return_value=_make_retriever_patch())
    @patch("agent.orchestrator.retrieve",       return_value=_make_retriever_patch())
    def test_run_questions_populated(self, m1, m2, m3):
        orch, _ = self._make_orchestrator()
        kit = orch.run(_make_profile(), generate_answers=False,
                       generate_tips=False, generate_strategy=False)
        # QuestionGenerator was called — lists may be empty if parse fails on mock
        # but the attributes must exist
        self.assertIsNotNone(kit.technical_questions)
        self.assertIsNotNone(kit.hr_questions)

    def test_analyze_profile_returns_user_profile(self):
        orch, _ = self._make_orchestrator()
        profile = orch.analyze_profile(
            name="Eve",
            role="Business Analyst",
            skills="SQL, BPMN",
        )
        self.assertIsInstance(profile, UserProfile)
        self.assertEqual(profile.name, "Eve")

    @patch("agent.mock_interviewer.retrieve", return_value=_make_retriever_patch())
    def test_create_mock_interviewer_returns_instance(self, mock_ret):
        orch, _ = self._make_orchestrator()
        mi = orch.create_mock_interviewer(_make_profile(), total_questions=5)
        self.assertIsInstance(mi, MockInterviewer)
        self.assertEqual(mi.total_questions, 5)

    def test_infer_question_type_behavioral(self):
        q = "Tell me about a time when you led a project under pressure."
        t = InterviewOrchestrator._infer_question_type(q)
        self.assertEqual(t, "behavioral")

    def test_infer_question_type_hr(self):
        q = "What are your greatest strengths?"
        t = InterviewOrchestrator._infer_question_type(q)
        self.assertEqual(t, "hr")

    def test_infer_question_type_technical(self):
        q = "Explain how a database index improves query performance."
        t = InterviewOrchestrator._infer_question_type(q)
        self.assertEqual(t, "technical")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
