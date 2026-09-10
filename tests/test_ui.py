"""
tests/test_ui.py
================
Tests for UI-layer helpers and the app.py module.

These tests verify:
  - The app.py module imports without error.
  - The Gradio Blocks object is built correctly.
  - All UI handler functions exist and are callable.
  - Handler functions return the right number of outputs.
  - The formatter utilities behave correctly with edge-case inputs.
  - Credential checking works as expected.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")


class TestAppImport(unittest.TestCase):
    """Verify app.py can be imported without error."""

    def test_app_module_imports(self):
        """app.py must be importable as a module (no side effects at import time)."""
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, "build_ui"))
        self.assertTrue(hasattr(mod, "generate_kit"))
        self.assertTrue(hasattr(mod, "evaluate_answer"))
        self.assertTrue(hasattr(mod, "start_mock_interview"))
        self.assertTrue(hasattr(mod, "submit_mock_answer"))
        self.assertTrue(hasattr(mod, "generate_final_feedback"))

    def test_build_ui_returns_blocks(self):
        """build_ui() must return a Gradio Blocks instance."""
        import importlib, gradio as gr
        spec = importlib.util.spec_from_file_location("app", "app.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        demo = mod.build_ui()
        self.assertIsInstance(demo, gr.Blocks)


class TestBuildCounts(unittest.TestCase):
    """Verify _build_counts distributes questions correctly per interview type."""

    def setUp(self):
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_technical_only(self):
        c = self.mod._build_counts("Technical", 5)
        self.assertEqual(c["technical"], 5)
        self.assertEqual(c["hr"], 0)
        self.assertEqual(c["behavioral"], 0)

    def test_hr_only(self):
        c = self.mod._build_counts("HR", 7)
        self.assertEqual(c["hr"], 7)
        self.assertEqual(c["technical"], 0)

    def test_behavioral_only(self):
        c = self.mod._build_counts("Behavioral", 5)
        self.assertEqual(c["behavioral"], 5)
        self.assertEqual(c["technical"], 0)

    def test_role_specific_only(self):
        c = self.mod._build_counts("Role-Specific", 5)
        self.assertEqual(c["role_specific"], 5)
        self.assertEqual(c["hr"], 0)

    def test_mixed_distributes_evenly(self):
        c = self.mod._build_counts("Mixed", 8)
        total = sum(c.values())
        # Mixed should produce 4 * floor(8/4) = 8 total
        self.assertEqual(total, 8)
        self.assertGreater(c["technical"], 0)
        self.assertGreater(c["hr"], 0)
        self.assertGreater(c["behavioral"], 0)


class TestCredentialCheck(unittest.TestCase):
    """Verify _credentials_ok reads from settings correctly."""

    def setUp(self):
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_credentials_false_when_empty(self):
        with patch.object(self.mod.settings, "WATSONX_API_KEY", ""), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", ""):
            self.assertFalse(self.mod._credentials_ok())

    def test_credentials_true_when_set(self):
        with patch.object(self.mod.settings, "WATSONX_API_KEY", "fake-key"), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", "fake-proj"):
            self.assertTrue(self.mod._credentials_ok())


class TestGenerateKitHandler(unittest.TestCase):
    """Test the generate_kit handler with mocked orchestrator."""

    def setUp(self):
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_generate_kit_no_credentials(self):
        """Without credentials, generate_kit should return a warning string."""
        with patch.object(self.mod.settings, "WATSONX_API_KEY", ""), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", ""):
            result = self.mod.generate_kit(
                "Alice", "Software Engineer", "Mid-level (3-6 years)",
                "Python, SQL", "", "Mixed", 5,
                "Auto (based on experience)",
            )
        # Returns a tuple of 7 outputs
        self.assertEqual(len(result), 7)
        # First output is the warning
        self.assertIn("credentials", result[0].lower())

    def test_generate_kit_with_mocked_orchestrator(self):
        """With credentials and a mocked orchestrator, generate_kit returns a full kit."""
        from agent.orchestrator import InterviewKit
        from utils.profile_parser import UserProfile

        fake_kit = InterviewKit(
            profile=UserProfile(name="Alice", role="Software Engineer"),
            technical_questions=["Q1?", "Q2?"],
            role_questions=["R1?"],
            hr_questions=["H1?"],
            behavioral_questions=["B1?"],
            model_answers={"Q1?": "Answer to Q1."},
            tips="Tip 1.",
            strategy="Day 1: Study.",
        )

        with patch.object(self.mod.settings, "WATSONX_API_KEY", "fake-key"), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", "fake-proj"), \
             patch.object(self.mod._orchestrator, "run", return_value=fake_kit):
            result = self.mod.generate_kit(
                "Alice", "Software Engineer", "Mid-level (3-6 years)",
                "Python", "", "Mixed", 5, "Auto (based on experience)",
            )
        self.assertEqual(len(result), 7)
        # Summary should mention the candidate name
        self.assertIn("Alice", result[0])
        # Questions output should contain Q1
        self.assertIn("Q1", result[1])


class TestEvaluateAnswerHandler(unittest.TestCase):

    def setUp(self):
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_evaluate_no_credentials(self):
        with patch.object(self.mod.settings, "WATSONX_API_KEY", ""), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", ""):
            result = self.mod.evaluate_answer(
                "Software Engineer", "Mid-level (3-6 years)", "Python",
                "What is OOP?", "OOP stands for...",
            )
        self.assertIn("credentials", result.lower())

    def test_evaluate_empty_question(self):
        with patch.object(self.mod.settings, "WATSONX_API_KEY", "key"), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", "proj"):
            result = self.mod.evaluate_answer(
                "Software Engineer", "Mid-level (3-6 years)", "Python",
                "", "Some answer",
            )
        self.assertIn("question", result.lower())

    def test_evaluate_empty_answer(self):
        with patch.object(self.mod.settings, "WATSONX_API_KEY", "key"), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", "proj"):
            result = self.mod.evaluate_answer(
                "Software Engineer", "Mid-level (3-6 years)", "Python",
                "What is OOP?", "",
            )
        self.assertIn("answer", result.lower())

    def test_evaluate_with_mocked_orchestrator(self):
        mock_eval = {
            "score": "8/10",
            "strengths": ["Clear explanation"],
            "weaknesses": ["Missing examples"],
            "missing_points": [],
            "tech_accuracy": "Accurate.",
            "communication": "Concise.",
            "suggestions": ["Add examples."],
            "model_answer": "OOP is...",
        }
        with patch.object(self.mod.settings, "WATSONX_API_KEY", "key"), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", "proj"), \
             patch.object(self.mod._orchestrator, "evaluate_answer", return_value=mock_eval):
            result = self.mod.evaluate_answer(
                "Software Engineer", "Mid-level (3-6 years)", "Python",
                "What is OOP?", "OOP is object-oriented programming.",
            )
        self.assertIn("8/10", result)
        self.assertIn("Clear explanation", result)


class TestMockInterviewHandlers(unittest.TestCase):

    def setUp(self):
        import importlib
        spec = importlib.util.spec_from_file_location("app", "app.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_start_mock_no_credentials(self):
        with patch.object(self.mod.settings, "WATSONX_API_KEY", ""), \
             patch.object(self.mod.settings, "WATSONX_PROJECT_ID", ""):
            result = self.mod.start_mock_interview(
                "Alice", "Software Engineer", "Mid-level (3-6 years)",
                "Python", "", 8, "Mixed", {},
            )
        # 4 outputs
        self.assertEqual(len(result), 4)
        self.assertIn("credentials", result[0].lower())

    def test_submit_answer_without_session(self):
        result = self.mod.submit_mock_answer("My answer", {})
        # 4 outputs
        self.assertEqual(len(result), 4)
        self.assertIn("No active", result[0])

    def test_submit_empty_answer(self):
        mock_mi = MagicMock()
        state = {"interviewer": mock_mi}
        result = self.mod.submit_mock_answer("", state)
        self.assertEqual(len(result), 4)
        self.assertIn("answer", result[0].lower())


class TestFormatterIntegration(unittest.TestCase):
    """Integration-level checks on formatters called by the UI."""

    def test_format_questions_strips_existing_numbering(self):
        from utils.formatter import format_questions
        qs = ["1. Already numbered", "2) Also numbered"]
        out = format_questions(qs, "Test")
        # Should not have "1. 1." double numbering
        self.assertNotIn("1. 1.", out)
        self.assertNotIn("2. 2)", out)

    def test_format_interview_kit_summary_with_empty_kit(self):
        from utils.formatter import format_interview_kit_summary
        kit_dict = {
            "profile": {"role": "Data Scientist", "experience_level": "Junior (1-3 years)"},
            "technical_questions": [],
            "role_questions": [],
            "hr_questions": [],
            "behavioral_questions": [],
        }
        summary = format_interview_kit_summary(kit_dict)
        self.assertIn("Data Scientist", summary)
        self.assertIn("0 questions", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
