"""
tests/test_integration.py
==========================
End-to-end integration tests for the AI Interview Trainer Agent.

All tests use mocked IBM Granite (no live credentials required).
RAG retrieval uses the real ChromaDB vector store when available, and
falls back gracefully when the store has not been built.

Test marker legend
------------------
  [MOCKED]  — Granite LLM is mocked; no IBM Cloud quota consumed.
  [LIVE]    — Would require real WATSONX_API_KEY / WATSONX_PROJECT_ID.
              All live tests are skipped unless credentials are present in .env.

Running
-------
    pytest tests/test_integration.py -v          # mocked tests only
    pytest tests/test_integration.py -v -m live  # include live tests (requires .env)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(response: str = "Generated content from mock LLM.") -> MagicMock:
    m = MagicMock()
    m.generate.return_value = response
    return m


def _rag_docs(texts=None):
    """Return mock LangChain Documents for RAG patching."""
    from langchain_core.documents import Document
    return [
        Document(
            page_content=t,
            metadata={"role": "Software Engineer", "category": "technical_questions"},
        )
        for t in (texts or ["Mock retrieved context."])
    ]


_MOCK_QUESTIONS = (
    "1. Explain the difference between a process and a thread.\n"
    "2. What is the CAP theorem?\n"
    "3. Describe binary search tree traversal.\n"
    "4. What is dependency injection?\n"
    "5. How do hash maps handle collisions?"
)

_MOCK_EVALUATION = (
    "OVERALL SCORE: [8/10] — Strong answer with good depth.\n\n"
    "STRENGTHS:\n- Clear explanation\n- Good use of examples\n\n"
    "WEAKNESSES:\n- Missing time complexity\n\n"
    "MISSING POINTS:\n- Big-O analysis\n\n"
    "TECHNICAL ACCURACY:\nContent is accurate.\n\n"
    "COMMUNICATION FEEDBACK:\nWell structured.\n\n"
    "IMPROVEMENT SUGGESTIONS:\n1. Add complexity analysis.\n2. Use code examples.\n\n"
    "MODEL ANSWER:\nA hash map uses a hash function..."
)

_MOCK_FEEDBACK    = "Overall performance is strong. Keep practising system design."
_MOCK_TIPS        = "TECHNICAL PREPARATION TIPS\n1. Practise LeetCode daily."
_MOCK_STRATEGY    = "DAY 1 — Algorithms: Review sorting. (~2 hours)"
_MOCK_MOCK_Q      = "[NEXT QUESTION]\nQuestion 1: What is a thread?"
_MOCK_MOCK_EVAL   = "[EVALUATION]\nQuick Score: 7/10\nWell done: Clear.\nTo improve: Add examples.\n\n[NEXT QUESTION]\nQuestion 2: Explain OOP."


# ---------------------------------------------------------------------------
# 1. Application bootstrap integration
# ---------------------------------------------------------------------------

class TestApplicationBootstrap(unittest.TestCase):
    """Verify the full application stack can be imported and initialised."""

    def test_all_layers_importable(self):
        """[MOCKED] Every module in every layer must be importable."""
        from config.settings import settings                     # noqa: F401
        from llm.granite_client import GraniteClient             # noqa: F401
        from rag.loader import load_all_documents                # noqa: F401
        from rag.chunker import chunk_documents                  # noqa: F401
        from rag.embedder import get_embedder                    # noqa: F401
        from rag.vector_store import load_store                  # noqa: F401
        from rag.retriever import retrieve                       # noqa: F401
        from prompts import load_prompt                          # noqa: F401
        from utils.profile_parser import UserProfile, ProfileParser  # noqa: F401
        from utils.formatter import format_questions             # noqa: F401
        from agent.profile_analyzer import ProfileAnalyzer       # noqa: F401
        from agent.question_generator import QuestionGenerator   # noqa: F401
        from agent.answer_evaluator import AnswerEvaluator       # noqa: F401
        from agent.feedback_agent import FeedbackAgent           # noqa: F401
        from agent.mock_interviewer import MockInterviewer       # noqa: F401
        from agent.orchestrator import InterviewOrchestrator, InterviewKit  # noqa: F401

    def test_app_py_importable(self):
        """[MOCKED] app.py must be importable without executing the server."""
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIsNotNone(mod)

    def test_gradio_blocks_build(self):
        """[MOCKED] build_ui() must return a valid Gradio Blocks object."""
        import importlib, gradio as gr
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        demo = mod.build_ui()
        self.assertIsInstance(demo, gr.Blocks)

    def test_settings_loaded(self):
        """[MOCKED] settings singleton must expose all required constants."""
        self.assertIsNotNone(settings.SUPPORTED_ROLES)
        self.assertIsNotNone(settings.EXPERIENCE_LEVELS)
        self.assertIsNotNone(settings.EMBEDDING_MODEL)
        self.assertIsNotNone(settings.GRANITE_MODEL_ID)
        self.assertIsNotNone(settings.CHROMA_COLLECTION_NAME)
        self.assertIsInstance(settings.TOP_K_RETRIEVAL, int)

    def test_all_prompt_templates_loadable(self):
        """[MOCKED] All prompt template files must be readable via load_prompt()."""
        from prompts import load_prompt
        templates = [
            "system_prompt", "question_generation", "answer_evaluation",
            "feedback", "mock_interview", "technical_prompt",
            "role_specific_prompt", "hr_prompt", "behavioral_prompt",
            "answer_prompt", "tips_prompt", "strategy_prompt",
        ]
        for name in templates:
            with self.subTest(template=name):
                text = load_prompt(name)
                self.assertIsInstance(text, str)
                self.assertGreater(len(text), 50)

    def test_knowledge_base_files_exist(self):
        """[MOCKED] All expected knowledge base files must be present on disk."""
        kb = settings.KNOWLEDGE_BASE_DIR
        expected = [
            kb / "technical_questions" / "software_engineer.json",
            kb / "technical_questions" / "data_scientist.json",
            kb / "technical_questions" / "devops_engineer.json",
            kb / "technical_questions" / "product_manager.json",
            kb / "technical_questions" / "business_analyst.json",
            kb / "hr_questions" / "hr_questions.json",
            kb / "behavioral_questions" / "behavioral_questions.json",
            kb / "industry_guidelines" / "interview_prep_guide.txt",
        ]
        for path in expected:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), f"Missing: {path}")


# ---------------------------------------------------------------------------
# 2. Profile → Question Generation integration
# ---------------------------------------------------------------------------

class TestProfileToQuestionFlow(unittest.TestCase):
    """[MOCKED] Profile parsing → RAG retrieval → question generation."""

    def setUp(self):
        from utils.profile_parser import ProfileParser
        self.parser = ProfileParser()
        self.profile = self.parser.parse(
            name="Test User",
            role="Software Engineer",
            experience_level="Mid-level (3-6 years)",
            skills="Python, System Design, SQL",
            resume_summary="Backend engineer with 4 years experience.",
        )

    def test_profile_fields_flow_to_question_prompt(self):
        """[MOCKED] Profile values must appear in the rendered prompt."""
        from prompts import load_prompt
        prompt = load_prompt("technical_prompt").format(
            role=self.profile.role,
            experience=self.profile.experience_level,
            skills=self.profile.skills_str,
            num_questions=5,
            difficulty="intermediate",
            context="Mock context.",
        )
        self.assertIn("Software Engineer", prompt)
        self.assertIn("Mid-level", prompt)
        self.assertIn("Python", prompt)

    @patch("agent.question_generator.retrieve", return_value=_rag_docs())
    def test_full_question_generation_pipeline(self, mock_ret):
        """[MOCKED] Profile → RAG retrieval → Granite → parsed questions list."""
        from agent.question_generator import QuestionGenerator
        llm = _mock_llm(_MOCK_QUESTIONS)
        qg  = QuestionGenerator(llm_client=llm)

        questions = qg.generate_technical(self.profile, num_questions=5)

        # RAG was called
        mock_ret.assert_called_once()
        # LLM was called
        llm.generate.assert_called_once()
        # Returned a list
        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)

    @patch("agent.question_generator.retrieve", return_value=_rag_docs())
    def test_generate_all_produces_four_types(self, mock_ret):
        """[MOCKED] generate_all() must return all four question type keys."""
        from agent.question_generator import QuestionGenerator
        llm = _mock_llm(_MOCK_QUESTIONS)
        qg  = QuestionGenerator(llm_client=llm)
        result = qg.generate_all(self.profile, counts={
            "technical": 3, "role_specific": 3, "hr": 3, "behavioral": 3
        })
        for key in ("technical", "role_specific", "hr", "behavioral"):
            with self.subTest(key=key):
                self.assertIn(key, result)
                self.assertIsInstance(result[key], list)

    @patch("agent.question_generator.retrieve", return_value=_rag_docs())
    def test_role_filter_passed_to_rag(self, mock_ret):
        """[MOCKED] RAG must be called with the correct role filter."""
        from agent.question_generator import QuestionGenerator
        qg = QuestionGenerator(llm_client=_mock_llm(_MOCK_QUESTIONS))
        qg.generate_technical(self.profile, num_questions=3)
        call_kwargs = mock_ret.call_args
        # role_filter should be the profile's role
        self.assertIn("Software Engineer", str(call_kwargs))

    def test_rag_failure_returns_empty_context(self):
        """[MOCKED] If RAG raises, QuestionGenerator must continue with empty context."""
        from agent.question_generator import QuestionGenerator
        llm = _mock_llm(_MOCK_QUESTIONS)
        qg  = QuestionGenerator(llm_client=llm)
        with patch("agent.question_generator.retrieve", side_effect=RuntimeError("DB down")):
            # Should NOT raise — just proceed with empty context
            questions = qg.generate_technical(self.profile, num_questions=3)
        self.assertIsInstance(questions, list)


# ---------------------------------------------------------------------------
# 3. RAG retrieval integration
# ---------------------------------------------------------------------------

class TestRAGRetrieval(unittest.TestCase):
    """Test the RAG retrieval layer against the real ChromaDB store."""

    @classmethod
    def setUpClass(cls):
        """Check if vector store is available; skip gracefully if not."""
        from pathlib import Path
        cls.store_available = Path(settings.CHROMA_PERSIST_DIR).exists()

    def test_rag_loader_loads_documents(self):
        """[MOCKED] Knowledge base loader must return non-empty document list."""
        from rag.loader import load_all_documents
        docs = load_all_documents()
        self.assertGreater(len(docs), 0)

    def test_rag_chunker_produces_chunks(self):
        """[MOCKED] Chunker must produce at least as many chunks as input docs."""
        from rag.loader import load_all_documents
        from rag.chunker import chunk_documents
        docs   = load_all_documents()
        chunks = chunk_documents(docs)
        self.assertGreaterEqual(len(chunks), len(docs))

    def test_rag_chunker_preserves_metadata(self):
        """[MOCKED] Every chunk must retain source metadata."""
        from rag.loader import load_all_documents
        from rag.chunker import chunk_documents
        docs   = load_all_documents()
        chunks = chunk_documents(docs)
        for chunk in chunks[:10]:
            with self.subTest():
                self.assertIn("role", chunk.metadata)
                self.assertIn("category", chunk.metadata)
                self.assertIn("source", chunk.metadata)

    def test_retrieve_with_real_store(self):
        """[MOCKED/REAL] Retrieve using real ChromaDB store (skipped if not built)."""
        if not self.store_available:
            self.skipTest("ChromaDB store not built — run `python -m rag.ingest` first.")
        from rag.retriever import retrieve, invalidate_store_cache
        invalidate_store_cache()
        results = retrieve("technical interview questions Python", k=3)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(hasattr(d, "page_content") for d in results))

    def test_retrieve_with_role_filter(self):
        """[MOCKED/REAL] Role-filtered retrieval must not return wrong-role docs."""
        if not self.store_available:
            self.skipTest("ChromaDB store not built — run `python -m rag.ingest` first.")
        from rag.retriever import retrieve, invalidate_store_cache
        invalidate_store_cache()
        results = retrieve("machine learning algorithms", role_filter="Data Scientist", k=5)
        for doc in results:
            with self.subTest():
                self.assertIn(
                    doc.metadata.get("role"),
                    ["Data Scientist", "general"],
                )

    def test_retrieve_empty_query_returns_empty(self):
        """[MOCKED] Empty query must return an empty list, not raise."""
        from rag.retriever import retrieve
        results = retrieve("", k=3)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 4. Answer evaluation integration
# ---------------------------------------------------------------------------

class TestAnswerEvaluationFlow(unittest.TestCase):
    """[MOCKED] Question → candidate answer → AnswerEvaluator → structured result."""

    def setUp(self):
        from utils.profile_parser import ProfileParser
        self.profile = ProfileParser().parse(
            role="Software Engineer",
            experience_level="Mid-level (3-6 years)",
            skills="Python, SQL",
        )

    @patch("agent.answer_evaluator.retrieve", return_value=_rag_docs())
    def test_evaluation_dict_has_all_fields(self, _):
        """[MOCKED] Evaluation result must contain all 8 required fields."""
        from agent.answer_evaluator import AnswerEvaluator
        evaluator = AnswerEvaluator(llm_client=_mock_llm(_MOCK_EVALUATION))
        result = evaluator.evaluate(
            question="How does a hash map work?",
            answer="A hash map uses a hash function to map keys to buckets.",
            profile=self.profile,
        )
        required = [
            "raw_text", "score", "strengths", "weaknesses",
            "missing_points", "tech_accuracy", "communication",
            "suggestions", "model_answer",
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, result)

    @patch("agent.answer_evaluator.retrieve", return_value=_rag_docs())
    def test_score_parsed_correctly(self, _):
        """[MOCKED] Score field must be correctly extracted from Granite response."""
        from agent.answer_evaluator import AnswerEvaluator
        evaluator = AnswerEvaluator(llm_client=_mock_llm(_MOCK_EVALUATION))
        result = evaluator.evaluate("Q", "A", self.profile)
        self.assertEqual(result["score"], "8/10")

    @patch("agent.answer_evaluator.retrieve", return_value=_rag_docs())
    def test_strengths_is_list(self, _):
        """[MOCKED] Strengths must be a list of strings."""
        from agent.answer_evaluator import AnswerEvaluator
        evaluator = AnswerEvaluator(llm_client=_mock_llm(_MOCK_EVALUATION))
        result = evaluator.evaluate("Q", "A", self.profile)
        self.assertIsInstance(result["strengths"], list)
        self.assertGreater(len(result["strengths"]), 0)

    def test_empty_answer_skips_llm(self):
        """[MOCKED] Empty answer must return early without calling LLM."""
        from agent.answer_evaluator import AnswerEvaluator
        llm = _mock_llm()
        evaluator = AnswerEvaluator(llm_client=llm)
        result = evaluator.evaluate("Q?", "", self.profile)
        llm.generate.assert_not_called()
        self.assertEqual(result["score"], "N/A")

    @patch("agent.answer_evaluator.retrieve", return_value=_rag_docs())
    def test_batch_evaluation(self, _):
        """[MOCKED] Batch evaluation must process every question/answer pair."""
        from agent.answer_evaluator import AnswerEvaluator
        evaluator = AnswerEvaluator(llm_client=_mock_llm(_MOCK_EVALUATION))
        questions = ["Q1?", "Q2?", "Q3?"]
        answers   = ["A1",  "A2",  "A3"]
        results = evaluator.evaluate_batch(questions, answers, self.profile)
        self.assertEqual(len(results), 3)
        self.assertEqual(evaluator._llm.generate.call_count, 3)


# ---------------------------------------------------------------------------
# 5. Mock interview multi-step integration
# ---------------------------------------------------------------------------

class TestMockInterviewFlow(unittest.TestCase):
    """[MOCKED] Full mock interview session: start → submit × N → debrief."""

    def _make_session(self, total=3):
        from agent.mock_interviewer import MockInterviewer
        from utils.profile_parser import ProfileParser
        profile = ProfileParser().parse(
            role="Software Engineer",
            experience_level="Mid-level (3-6 years)",
        )
        responses = (
            [_MOCK_MOCK_Q]
            + [_MOCK_MOCK_EVAL] * (total - 1)
            + ["[SESSION COMPLETE]\nOverall Score: 7/10\nTop Strengths: A | B\nAreas: C | D\nGreat job!"]
        )
        llm = MagicMock()
        llm.generate.side_effect = responses
        with patch("agent.mock_interviewer.retrieve", return_value=_rag_docs()):
            mi = MockInterviewer(llm_client=llm, profile=profile, total_questions=total)
        return mi, llm

    @patch("agent.mock_interviewer.retrieve", return_value=_rag_docs())
    def test_start_session_calls_llm(self, _):
        """[MOCKED] start_session() must call Granite exactly once."""
        from agent.mock_interviewer import MockInterviewer
        from utils.profile_parser import ProfileParser
        profile = ProfileParser().parse(role="Software Engineer")
        llm = _mock_llm(_MOCK_MOCK_Q)
        mi  = MockInterviewer(llm_client=llm, profile=profile, total_questions=3)
        with patch("agent.mock_interviewer.retrieve", return_value=_rag_docs()):
            mi.start_session()
        llm.generate.assert_called_once()

    @patch("agent.mock_interviewer.retrieve", return_value=_rag_docs())
    def test_submit_answer_returns_dict(self, _):
        """[MOCKED] submit_answer() must return a dict with required keys."""
        mi, _ = self._make_session(total=3)
        with patch("agent.mock_interviewer.retrieve", return_value=_rag_docs()):
            mi.start_session()
            result = mi.submit_answer("Threads share memory; processes don't.")
        for key in ("feedback", "is_complete", "question_number"):
            with self.subTest(key=key):
                self.assertIn(key, result)

    @patch("agent.mock_interviewer.retrieve", return_value=_rag_docs())
    def test_session_completes_correctly(self, _):
        """[MOCKED] Session must be marked complete after all questions answered."""
        mi, llm = self._make_session(total=2)
        with patch("agent.mock_interviewer.retrieve", return_value=_rag_docs()):
            mi.start_session()
            mi.submit_answer("Answer 1")
            result = mi.submit_answer("Answer 2")
        self.assertTrue(result["is_complete"])
        self.assertIn("debrief", result)

    @patch("agent.mock_interviewer.retrieve", return_value=_rag_docs())
    def test_history_grows_with_turns(self, _):
        """[MOCKED] History list must grow with each turn submitted."""
        mi, _ = self._make_session(total=3)
        with patch("agent.mock_interviewer.retrieve", return_value=_rag_docs()):
            mi.start_session()
            initial_len = len(mi.history)
            mi.submit_answer("Answer 1")
            self.assertGreater(len(mi.history), initial_len)

    @patch("agent.mock_interviewer.retrieve", return_value=_rag_docs())
    def test_cannot_start_twice(self, _):
        """[MOCKED] Calling start_session() twice must raise RuntimeError."""
        mi, _ = self._make_session(total=3)
        with patch("agent.mock_interviewer.retrieve", return_value=_rag_docs()):
            mi.start_session()
            with self.assertRaises(RuntimeError):
                mi.start_session()

    @patch("agent.mock_interviewer.retrieve", return_value=_rag_docs())
    def test_average_score_computed(self, _):
        """[MOCKED] Average score must reflect the scores from session turns."""
        mi, _ = self._make_session(total=2)
        with patch("agent.mock_interviewer.retrieve", return_value=_rag_docs()):
            mi.start_session()
            mi.submit_answer("Answer 1")
        mi.history[0].score = 8.0
        self.assertEqual(mi.get_average_score(), 8.0)


# ---------------------------------------------------------------------------
# 6. Feedback generation integration
# ---------------------------------------------------------------------------

class TestFeedbackGenerationFlow(unittest.TestCase):
    """[MOCKED] Performance data → FeedbackAgent → feedback report."""

    def setUp(self):
        from utils.profile_parser import ProfileParser
        self.profile = ProfileParser().parse(
            name="Integration Test User",
            role="Data Scientist",
            experience_level="Junior (1-3 years)",
            skills="Python, ML, SQL",
        )

    @patch("agent.feedback_agent.retrieve", return_value=_rag_docs())
    def test_generate_feedback_returns_non_empty_string(self, _):
        """[MOCKED] generate_feedback() must return a non-empty string."""
        from agent.feedback_agent import FeedbackAgent
        fa = FeedbackAgent(llm_client=_mock_llm(_MOCK_FEEDBACK))
        result = fa.generate_feedback(self.profile)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    @patch("agent.feedback_agent.retrieve", return_value=_rag_docs())
    def test_generate_feedback_with_evaluations(self, _):
        """[MOCKED] generate_feedback() must incorporate evaluation data."""
        from agent.feedback_agent import FeedbackAgent
        fa = FeedbackAgent(llm_client=_mock_llm(_MOCK_FEEDBACK))
        evals = [
            {"score": "7/10", "weaknesses": ["Vague answer"]},
            {"score": "5/10", "weaknesses": ["Missing examples"]},
        ]
        result = fa.generate_feedback(
            self.profile, evaluations=evals, mock_history="Q1 history text."
        )
        self.assertIsInstance(result, str)
        # LLM must have been called once
        fa._llm.generate.assert_called_once()

    @patch("agent.feedback_agent.retrieve", return_value=_rag_docs())
    def test_generate_tips_uses_questions(self, _):
        """[MOCKED] generate_tips() must include the question list in the prompt."""
        from agent.feedback_agent import FeedbackAgent
        llm = _mock_llm(_MOCK_TIPS)
        fa  = FeedbackAgent(llm_client=llm)
        fa.generate_tips(self.profile, questions=["Q1?", "Q2?"])
        prompt_used = llm.generate.call_args[0][0]  # first positional arg
        self.assertIn("Q1?", prompt_used)

    @patch("agent.feedback_agent.retrieve", return_value=_rag_docs())
    def test_generate_strategy_returns_string(self, _):
        """[MOCKED] generate_strategy() must return a non-empty string."""
        from agent.feedback_agent import FeedbackAgent
        fa = FeedbackAgent(llm_client=_mock_llm(_MOCK_STRATEGY))
        result = fa.generate_strategy(self.profile)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)


# ---------------------------------------------------------------------------
# 7. Full orchestrator integration
# ---------------------------------------------------------------------------

class TestOrchestratorIntegration(unittest.TestCase):
    """[MOCKED] InterviewOrchestrator.run() end-to-end with all components mocked."""

    @patch("agent.question_generator.retrieve", return_value=_rag_docs())
    @patch("agent.feedback_agent.retrieve",     return_value=_rag_docs())
    @patch("agent.orchestrator.retrieve",       return_value=_rag_docs())
    def test_run_returns_complete_interview_kit(self, *mocks):
        """[MOCKED] run() must return an InterviewKit with all fields populated."""
        from agent.orchestrator import InterviewOrchestrator, InterviewKit
        from utils.profile_parser import ProfileParser
        llm = _mock_llm(_MOCK_QUESTIONS)
        # Override all LLM calls to return contextually appropriate responses
        responses = [
            _MOCK_QUESTIONS,  # technical
            _MOCK_QUESTIONS,  # role_specific
            _MOCK_QUESTIONS,  # hr
            _MOCK_QUESTIONS,  # behavioral
        ] + [_MOCK_EVALUATION] * 20 + [_MOCK_TIPS, _MOCK_STRATEGY]
        llm.generate.side_effect = responses

        orch = InterviewOrchestrator(llm_client=llm)
        profile = ProfileParser().parse(
            role="Software Engineer",
            experience_level="Mid-level (3-6 years)",
            skills="Python, SQL",
        )
        kit = orch.run(
            profile,
            question_counts={"technical": 3, "role_specific": 3, "hr": 3, "behavioral": 3},
        )

        self.assertIsInstance(kit, InterviewKit)
        self.assertIsNotNone(kit.profile)
        self.assertIsNotNone(kit.technical_questions)
        self.assertIsNotNone(kit.hr_questions)
        self.assertIsNotNone(kit.model_answers)
        self.assertIsInstance(kit.tips, str)
        self.assertIsInstance(kit.strategy, str)

    @patch("agent.question_generator.retrieve", return_value=_rag_docs())
    @patch("agent.feedback_agent.retrieve",     return_value=_rag_docs())
    @patch("agent.orchestrator.retrieve",       return_value=_rag_docs())
    def test_interview_kit_to_dict_is_serialisable(self, *mocks):
        """[MOCKED] InterviewKit.to_dict() must return a JSON-serialisable dict."""
        import json
        from agent.orchestrator import InterviewOrchestrator, InterviewKit
        from utils.profile_parser import ProfileParser
        llm = _mock_llm(_MOCK_QUESTIONS)
        llm.generate.side_effect = [_MOCK_QUESTIONS] * 10 + [_MOCK_TIPS, _MOCK_STRATEGY]
        orch = InterviewOrchestrator(llm_client=llm)
        profile = ProfileParser().parse(role="Product Manager")
        kit = orch.run(profile, generate_answers=False)
        kit_dict = kit.to_dict()
        json_str = json.dumps(kit_dict)  # must not raise
        self.assertIsInstance(json_str, str)

    def test_analyze_profile_returns_correct_role(self):
        """[MOCKED] analyze_profile() must return the correct normalised role."""
        from agent.orchestrator import InterviewOrchestrator
        orch = InterviewOrchestrator(llm_client=_mock_llm())
        profile = orch.analyze_profile(role="data scientist", skills="Python, ML")
        self.assertEqual(profile.role, "Data Scientist")

    @patch("agent.mock_interviewer.retrieve", return_value=_rag_docs())
    def test_create_mock_interviewer_is_isolated(self, _):
        """[MOCKED] Two MockInterviewer instances must not share state."""
        from agent.orchestrator import InterviewOrchestrator
        from utils.profile_parser import ProfileParser
        orch = InterviewOrchestrator(llm_client=_mock_llm(_MOCK_MOCK_Q))
        profile = ProfileParser().parse(role="Software Engineer")
        mi1 = orch.create_mock_interviewer(profile, total_questions=5)
        mi2 = orch.create_mock_interviewer(profile, total_questions=3)
        self.assertIsNot(mi1, mi2)
        self.assertEqual(mi1.total_questions, 5)
        self.assertEqual(mi2.total_questions, 3)


# ---------------------------------------------------------------------------
# 8. Missing-credentials behaviour
# ---------------------------------------------------------------------------

class TestMissingCredentialsBehaviour(unittest.TestCase):
    """[MOCKED] Graceful degradation when IBM credentials are absent."""

    def test_granite_client_raises_value_error_without_credentials(self):
        """[MOCKED] GraniteClient.generate() must raise ValueError (not crash) when no credentials."""
        from llm.granite_client import GraniteClient
        client = GraniteClient()
        with patch.object(settings, "WATSONX_API_KEY", ""), \
             patch.object(settings, "WATSONX_PROJECT_ID", ""):
            with self.assertRaises(ValueError) as ctx:
                client._get_model()
        self.assertIn("WATSONX_API_KEY", str(ctx.exception))

    def test_generate_kit_ui_handler_returns_warning(self):
        """[MOCKED] generate_kit() UI handler must return a warning string without crashing."""
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with patch.object(mod.settings, "WATSONX_API_KEY", ""), \
             patch.object(mod.settings, "WATSONX_PROJECT_ID", ""):
            result = mod.generate_kit(
                "Test", "Software Engineer", "Mid-level (3-6 years)",
                "Python", "", "Mixed", 5, "Auto (based on experience)",
            )
        self.assertIn("credentials", result[0].lower())

    def test_evaluate_answer_ui_handler_returns_warning(self):
        """[MOCKED] evaluate_answer() must return a warning without crashing."""
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with patch.object(mod.settings, "WATSONX_API_KEY", ""), \
             patch.object(mod.settings, "WATSONX_PROJECT_ID", ""):
            result = mod.evaluate_answer(
                "Software Engineer", "Mid-level (3-6 years)", "Python",
                "What is OOP?", "OOP is...",
            )
        self.assertIn("credentials", result.lower())

    def test_mock_interview_ui_returns_warning_without_credentials(self):
        """[MOCKED] start_mock_interview() must return a warning without crashing."""
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with patch.object(mod.settings, "WATSONX_API_KEY", ""), \
             patch.object(mod.settings, "WATSONX_PROJECT_ID", ""):
            result = mod.start_mock_interview(
                "Test", "Software Engineer", "Mid-level (3-6 years)",
                "Python", "", 5, "Mixed", {},
            )
        self.assertEqual(len(result), 4)
        self.assertIn("credentials", result[0].lower())


# ---------------------------------------------------------------------------
# 9. Error handling integration
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):
    """[MOCKED] Verify _safe_generate and individual agents handle errors gracefully."""

    def test_safe_generate_catches_value_error(self):
        """[MOCKED] _safe_generate must return (None, error_str) on ValueError."""
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def raises_value_error():
            raise ValueError("Bad value")

        result, err = mod._safe_generate(raises_value_error)
        self.assertIsNone(result)
        self.assertIn("Configuration error", err)

    def test_safe_generate_catches_runtime_error(self):
        """[MOCKED] _safe_generate must return (None, error_str) on RuntimeError."""
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def raises_runtime():
            raise RuntimeError("API down")

        result, err = mod._safe_generate(raises_runtime)
        self.assertIsNone(result)
        self.assertIn("Generation error", err)

    def test_question_generator_raises_on_llm_failure(self):
        """[MOCKED] QuestionGenerator must re-raise as RuntimeError on LLM failure."""
        from agent.question_generator import QuestionGenerator
        from utils.profile_parser import ProfileParser
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM unavailable")
        qg  = QuestionGenerator(llm_client=llm)
        profile = ProfileParser().parse(role="Software Engineer")
        with patch("agent.question_generator.retrieve", return_value=_rag_docs()):
            with self.assertRaises(RuntimeError):
                qg.generate_technical(profile)

    def test_answer_evaluator_raises_on_llm_failure(self):
        """[MOCKED] AnswerEvaluator must re-raise as RuntimeError on LLM failure."""
        from agent.answer_evaluator import AnswerEvaluator
        from utils.profile_parser import ProfileParser
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM unavailable")
        evaluator = AnswerEvaluator(llm_client=llm)
        profile   = ProfileParser().parse(role="Software Engineer")
        with patch("agent.answer_evaluator.retrieve", return_value=_rag_docs()):
            with self.assertRaises(RuntimeError):
                evaluator.evaluate("Q?", "A", profile)

    def test_profile_parser_never_raises_on_empty_input(self):
        """[MOCKED] ProfileParser.parse() must never raise even with all-None inputs."""
        from utils.profile_parser import ProfileParser
        parser = ProfileParser()
        profile = parser.parse(name=None, role=None, experience_level=None,
                                skills=None, resume_summary=None)
        self.assertIsNotNone(profile)
        self.assertIsNotNone(profile.role)


# ---------------------------------------------------------------------------
# 10. Live IBM Granite tests (skipped unless credentials present)
# ---------------------------------------------------------------------------

_LIVE_CREDS = bool(settings.WATSONX_API_KEY and settings.WATSONX_PROJECT_ID)

@unittest.skipUnless(_LIVE_CREDS, "Live IBM credentials not configured — skipping live tests.")
class TestLiveIBMGranite(unittest.TestCase):
    """[LIVE] Require valid WATSONX_API_KEY + WATSONX_PROJECT_ID in .env."""

    def test_granite_test_connection(self):
        """[LIVE] GraniteClient.test_connection() must return True."""
        from llm.granite_client import GraniteClient
        client = GraniteClient()
        result = client.test_connection()
        self.assertTrue(result)

    def test_live_question_generation(self):
        """[LIVE] Generate real technical questions via RAG + Granite."""
        from agent.orchestrator import InterviewOrchestrator
        from utils.profile_parser import ProfileParser
        orch    = InterviewOrchestrator()
        profile = ProfileParser().parse(
            name="Live Test",
            role="Software Engineer",
            experience_level="Mid-level (3-6 years)",
            skills="Python, System Design",
        )
        kit = orch.run(
            profile,
            question_counts={"technical": 2, "role_specific": 0,
                             "hr": 1, "behavioral": 0},
            generate_answers=False,
            generate_tips=False,
            generate_strategy=False,
        )
        self.assertGreater(len(kit.technical_questions), 0)
        self.assertGreater(len(kit.hr_questions), 0)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
