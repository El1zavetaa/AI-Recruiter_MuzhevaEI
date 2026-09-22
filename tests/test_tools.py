"""
Юнит-тесты для библиотеки инструментов (src/tools.py) и векторного поиска.

Запуск:
    python -m unittest discover -s tests -v
(pytest тоже подойдёт: pytest tests/ -v)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import tools
from src.schemas import Vacancy, ResumeChunk
from src.embeddings import TfidfEmbeddingModel
from src.vector_store import VectorStore


class TestExtractExperienceYears(unittest.TestCase):
    def test_single_range(self):
        text = "2018 - 2021 Разработчик Python"
        self.assertAlmostEqual(tools.extract_experience_years(text, current_year=2026), 3.0)

    def test_open_ended_range_uses_current_year(self):
        text = "2020 – н.в. MLOps-инженер"
        self.assertAlmostEqual(tools.extract_experience_years(text, current_year=2026), 6.0)

    def test_multiple_non_overlapping_ranges_are_summed(self):
        text = "2015 - 2017 Junior\n2018 - 2022 Middle"
        self.assertAlmostEqual(tools.extract_experience_years(text, current_year=2026), 6.0)

    def test_no_dates_returns_zero(self):
        self.assertEqual(tools.extract_experience_years("Опытный специалист", current_year=2026), 0.0)

    def test_duplicate_range_not_double_counted(self):
        text = "2018 - 2021 Backend-разработчик. Проект длился 2018 - 2021."
        self.assertAlmostEqual(tools.extract_experience_years(text, current_year=2026), 3.0)


class TestCheckMandatorySkills(unittest.TestCase):
    def test_matched_and_missing(self):
        resume = "Опыт работы с Kubernetes и Docker в высоконагруженных проектах."
        result = tools.check_mandatory_skills(resume, ["Kubernetes", "Docker", "Terraform"])
        self.assertEqual(sorted(result["matched"]), ["Docker", "Kubernetes"])
        self.assertEqual(result["missing"], ["Terraform"])

    def test_case_insensitive(self):
        resume = "Работал с PYTHON и golang"
        result = tools.check_mandatory_skills(resume, ["Python", "Go"])
        self.assertIn("Python", result["matched"])

    def test_empty_required_skills(self):
        result = tools.check_mandatory_skills("любой текст", [])
        self.assertEqual(result, {"matched": [], "missing": []})


class TestFindKeywordEvidence(unittest.TestCase):
    def test_finds_context_window(self):
        resume = "Ранее был тимлидом команды из 5 backend-разработчиков в стартапе."
        evidence = tools.find_keyword_evidence(resume, ["тимлид"])
        self.assertIn("тимлид", evidence)
        self.assertIn("тимлид", evidence["тимлид"].lower())

    def test_keyword_not_found(self):
        evidence = tools.find_keyword_evidence("обычное резюме без ключевых слов", ["cto"])
        self.assertEqual(evidence, {})


class TestSemanticSearch(unittest.TestCase):
    """Проверяем, что векторный поиск ранжирует более релевантное резюме выше
    и корректно агрегирует несколько чанков одного кандидата по максимуму."""

    def setUp(self):
        chunks = [
            ResumeChunk("cand_1", "Кандидат А", "cand_1_0",
                        "Опыт разработки на Python, Kubernetes, Docker, CI/CD пайплайны."),
            ResumeChunk("cand_1", "Кандидат А", "cand_1_1",
                        "Также занимался версткой сайтов на HTML и CSS."),
            ResumeChunk("cand_2", "Кандидат Б", "cand_2_0",
                        "Пять лет опыта в маркетинге и продажах, работа с клиентами."),
        ]
        self.store = VectorStore(embedding_model=TfidfEmbeddingModel())
        self.store.build_index(chunks)

    def test_relevant_candidate_ranks_first(self):
        vacancy = Vacancy(
            id="v1", title="DevOps Engineer",
            description="Ищем инженера с опытом Kubernetes и Docker",
            required_skills=["Kubernetes", "Docker"],
        )
        results = tools.semantic_search(self.store, vacancy, top_k=2)
        self.assertEqual(results[0].resume_id, "cand_1")

    def test_top_k_limits_results(self):
        vacancy = Vacancy(id="v1", title="X", description="Kubernetes Docker")
        results = tools.semantic_search(self.store, vacancy, top_k=1)
        self.assertEqual(len(results), 1)

    def test_aggregates_by_candidate_not_by_chunk(self):
        # У кандидата 2 чанка, но в выдаче он должен встретиться один раз
        vacancy = Vacancy(id="v1", title="X", description="Python Kubernetes Docker CI/CD")
        results = tools.semantic_search(self.store, vacancy, top_k=10)
        resume_ids = [r.resume_id for r in results]
        self.assertEqual(len(resume_ids), len(set(resume_ids)))


class TestSemanticRecheckSkill(unittest.TestCase):
    """Проверяем дозапрос отдельного навыка внутри одного конкретного
    резюме (agentic tool-calling, используется агентами)."""

    def setUp(self):
        chunks = [
            ResumeChunk("cand_1", "Кандидат А", "cand_1_0",
                        "Контейнеризация сервисов и оркестрация в высоконагруженном проде."),
            ResumeChunk("cand_2", "Кандидат Б", "cand_2_0",
                        "Пять лет опыта в маркетинге и продажах, работа с клиентами."),
        ]
        self.store = VectorStore(embedding_model=TfidfEmbeddingModel())
        self.store.build_index(chunks)

    def test_finds_paraphrased_mention_above_threshold(self):
        evidence = tools.semantic_recheck_skill(self.store, "cand_1", "Kubernetes", threshold=0.0)
        self.assertIsNotNone(evidence)

    def test_returns_none_for_irrelevant_candidate(self):
        evidence = tools.semantic_recheck_skill(self.store, "cand_2", "Kubernetes", threshold=0.3)
        self.assertIsNone(evidence)

    def test_query_within_resume_only_searches_that_candidate(self):
        results = self.store.query_within_resume("cand_1", "оркестрация", top_k=5)
        self.assertTrue(all(r.resume_id == "cand_1" for r in results))


if __name__ == "__main__":
    unittest.main()
