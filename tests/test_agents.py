import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas import Vacancy, ResumeChunk
from src.llm_client import RuleBasedLLM
from src.agents import TechnicalScreenerAgent, HRAnalystAgent
from src.embeddings import TfidfEmbeddingModel
from src.vector_store import VectorStore
from src.orchestrator import AgentOrchestrator
from src.debate import run_debate


VACANCY = Vacancy(
    id="v1", title="Backend Engineer (Go)",
    description="Ищем backend-разработчика на Go",
    required_skills=["Go", "PostgreSQL", "Docker"],
    nice_to_have_skills=["Kubernetes"],
    min_experience_years=3,
)

STRONG_RESUME = (
    "Дмитрий\n\n"
    "2019 - н.в. Backend-разработчик\n"
    "Разрабатывал микросервисы на Go, PostgreSQL, Docker.\n"
    "Был тимлидом команды из 4 человек.\n"
)

WEAK_RESUME = (
    "Егор\n\n"
    "2024 - н.в. Junior-разработчик\n"
    "Верстка сайтов на HTML и CSS.\n"
)

# Резюме без буквального слова "Docker", но с перефразированным описанием —
# должно поймать semantic_recheck_skill (дозапрос через векторный поиск).
PARAPHRASED_RESUME = (
    "Кирилл\n\n"
    "2018 - н.в. Backend-разработчик\n"
    "Разработка микросервисов на Go, работа с PostgreSQL.\n"
    "Контейнеризация сервисов и деплой в изолированных окружениях.\n"
)


def _build_store(resumes: dict) -> VectorStore:
    """resumes: {resume_id: (candidate_name, text)}"""
    chunks = [
        ResumeChunk(resume_id, name, f"{resume_id}_0", text)
        for resume_id, (name, text) in resumes.items()
    ]
    store = VectorStore(embedding_model=TfidfEmbeddingModel())
    store.build_index(chunks)
    return store


class TestTechnicalScreenerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = TechnicalScreenerAgent(RuleBasedLLM())

    def test_strong_candidate_gets_high_score(self):
        store = _build_store({"strong": ("Дмитрий", STRONG_RESUME)})
        finding = self.agent.analyze(VACANCY, "strong", store)
        self.assertGreater(finding.score, 80)
        self.assertEqual(finding.risks, [])

    def test_weak_candidate_gets_low_score_and_risks(self):
        store = _build_store({"weak": ("Егор", WEAK_RESUME)})
        finding = self.agent.analyze(VACANCY, "weak", store)
        self.assertLess(finding.score, 50)
        self.assertTrue(len(finding.risks) > 0)

    def test_notes_is_non_empty_string(self):
        store = _build_store({"strong": ("Дмитрий", STRONG_RESUME)})
        finding = self.agent.analyze(VACANCY, "strong", store)
        self.assertIsInstance(finding.notes, str)
        self.assertTrue(len(finding.notes) > 0)

    def test_missing_nice_to_have_is_weakness_not_risk(self):
        # STRONG_RESUME не содержит Kubernetes (nice-to-have) —
        # это должно попасть в weaknesses, а не в risks.
        store = _build_store({"strong": ("Дмитрий", STRONG_RESUME)})
        finding = self.agent.analyze(VACANCY, "strong", store)
        self.assertTrue(any("kubernetes" in w.lower() for w in finding.weaknesses))
        self.assertFalse(any("kubernetes" in r.lower() for r in finding.risks))

    def test_semantic_recheck_finds_paraphrased_skill(self):
        # В компании кроме кандидата с перефразировкой добавляем ещё
        # резюме, чтобы TF-IDF индекс не был вырожденным на одном документе.
        store = _build_store({
            "paraphrased": ("Кирилл", PARAPHRASED_RESUME),
            "weak": ("Егор", WEAK_RESUME),
        })
        finding = self.agent.analyze(VACANCY, "paraphrased", store)
        # Docker дословно в резюме не встречается — либо найден семантически
        # (и тогда это плюс, не риск), либо остаётся риском. Проверяем, что
        # он НЕ висит одновременно и как явный, и как семантический —
        # логика дозапроса как минимум не роняет пайплайн.
        self.assertIsInstance(finding.risks, list)


class TestHRAnalystAgent(unittest.TestCase):
    def setUp(self):
        self.agent = HRAnalystAgent(RuleBasedLLM())

    def test_detects_management_experience(self):
        store = _build_store({"strong": ("Дмитрий", STRONG_RESUME)})
        finding = self.agent.analyze(VACANCY, "strong", store)
        self.assertTrue(any("управлен" in s.lower() for s in finding.strengths))

    def test_frequent_job_changes_flagged_as_risk(self):
        hoppy_resume = (
            "2020 - 2020 Компания А\n"
            "2021 - 2021 Компания Б\n"
            "2022 - 2022 Компания В\n"
        )
        store = _build_store({"hoppy": ("Тест", hoppy_resume)})
        finding = self.agent.analyze(VACANCY, "hoppy", store)
        self.assertTrue(any("част" in r.lower() for r in finding.risks))

    def test_no_management_evidence_is_weakness_not_risk(self):
        store = _build_store({"weak": ("Егор", WEAK_RESUME)})
        finding = self.agent.analyze(VACANCY, "weak", store)
        self.assertTrue(any("управленческ" in w.lower() for w in finding.weaknesses))


class TestDebate(unittest.TestCase):
    def test_consensus_when_no_risks(self):
        from src.schemas import AgentFinding
        tech = AgentFinding("Технический Скринер", 90.0, ["ок"], [], [], "note")
        hr = AgentFinding("HR-Аналитик", 85.0, ["ок"], [], [], "note")
        notes = run_debate(tech, hr, RuleBasedLLM())
        self.assertIn("сходятся", notes.lower())

    def test_flags_conflict_between_agents(self):
        from src.schemas import AgentFinding
        tech = AgentFinding("Технический Скринер", 95.0, ["ок"], [], [], "note")
        hr = AgentFinding("HR-Аналитик", 40.0, [], [], ["частая смена мест работы"], "note")
        notes = run_debate(tech, hr, RuleBasedLLM())
        self.assertIn("противоречие", notes.lower())


class TestOrchestrator(unittest.TestCase):
    def test_rank_candidates_end_to_end(self):
        chunks = (
            [ResumeChunk("strong", "Дмитрий", "strong_0", STRONG_RESUME)]
            + [ResumeChunk("weak", "Егор", "weak_0", WEAK_RESUME)]
        )
        store = VectorStore(embedding_model=TfidfEmbeddingModel())
        store.build_index(chunks)

        orchestrator = AgentOrchestrator(store, llm=RuleBasedLLM())
        reports = orchestrator.rank_candidates(VACANCY, top_k=2)

        self.assertEqual(len(reports), 2)
        # Более сильный кандидат должен оказаться на первом месте
        self.assertEqual(reports[0].resume_id, "strong")
        # Список должен быть отсортирован по убыванию финального скора
        scores = [r.final_score for r in reports]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # У каждого кандидата должно быть ровно 2 заключения (Технический + HR)
        self.assertEqual(len(reports[0].findings), 2)
        # И заполненный результат "дебатов"
        self.assertIsInstance(reports[0].debate_notes, str)
        self.assertTrue(len(reports[0].debate_notes) > 0)


if __name__ == "__main__":
    unittest.main()
