import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas import Vacancy, CandidateReport, AgentFinding
from src.report import render_markdown, render_html, save_report

VACANCY = Vacancy(
    id="v1", title="Test Vacancy",
    description="описание",
    required_skills=["Python"],
    min_experience_years=2,
)

REPORT = CandidateReport(
    candidate_name="Тестов Тест",
    resume_id="test_1",
    vacancy_title=VACANCY.title,
    similarity_score=88.5,
    experience_years=5.0,
    matched_skills=["Python"],
    missing_skills=[],
    findings=[
        AgentFinding("Технический Скринер", 90.0, ["Плюс 1"], ["Минус 1"], [], "Заметка тех. скринера"),
        AgentFinding("HR-Аналитик", 80.0, ["Плюс 2"], [], ["Риск 1"], "Заметка HR"),
    ],
    final_score=91.2,
    verdict="Рекомендован к приглашению",
    debate_notes="Агенты сходятся во мнении, значимых противоречий не выявлено.",
)


class TestReportRendering(unittest.TestCase):
    def test_markdown_contains_key_fields(self):
        md = render_markdown(VACANCY, [REPORT])
        self.assertIn("Тестов Тест", md)
        self.assertIn("91.2", md)
        self.assertIn("Рекомендован к приглашению", md)
        self.assertIn("Минус 1", md)
        self.assertIn("сходятся", md.lower())

    def test_html_contains_key_fields(self):
        html = render_html(VACANCY, [REPORT])
        self.assertIn("Тестов Тест", html)
        self.assertIn("Test Vacancy", html)

    def test_save_report_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_report(VACANCY, [REPORT], tmp)
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(os.path.isfile(p))
                self.assertGreater(os.path.getsize(p), 0)


if __name__ == "__main__":
    unittest.main()
