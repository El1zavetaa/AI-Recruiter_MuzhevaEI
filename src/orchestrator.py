"""
Оркестратор агентов — центральное ядро системы.

Пайплайн:
1) semantic_search находит top-N кандидатов, ближайших к вакансии по смыслу;
2) для каждого из них по очереди отрабатывают Технический Скринер,
   затем HR-Аналитик (агенты могут пользоваться инструментами из tools.py
   для уточнения деталей — extract_experience_years, check_mandatory_skills);
3) результаты агрегируются в финальный взвешенный скор и карточку кандидата.
"""
from __future__ import annotations
from typing import List

from .schemas import Vacancy, CandidateReport
from .vector_store import VectorStore
from .agents import TechnicalScreenerAgent, HRAnalystAgent
from .llm_client import LLMClient, get_default_llm_client
from .debate import run_debate
from . import tools

# Веса финального скора: приоритет — смысловая близость и хард-скиллы,
# HR-оценка — модифицирующий (риск-корректирующий) фактор.
WEIGHTS = {
    "similarity": 0.30,
    "technical": 0.45,
    "hr": 0.25,
}


def _verdict(final_score: float) -> str:
    if final_score >= 75:
        return "Рекомендован к приглашению"
    if final_score >= 50:
        return "В резерв"
    return "Отказ"


class AgentOrchestrator:
    def __init__(self, store: VectorStore, llm: LLMClient = None):
        self.store = store
        self.llm = llm or get_default_llm_client()
        self.technical_agent = TechnicalScreenerAgent(self.llm)
        self.hr_agent = HRAnalystAgent(self.llm)

    def rank_candidates(self, vacancy: Vacancy, top_k: int = 10) -> List[CandidateReport]:
        search_results = tools.semantic_search(self.store, vacancy, top_k=top_k)

        reports: List[CandidateReport] = []
        for result in search_results:
            # Оба агента получают доступ к store и сами решают, когда
            # дозапросить дополнительные чанки (semantic_recheck_skill) —
            # вместо того, чтобы сразу видеть весь текст резюме целиком.
            technical_finding = self.technical_agent.analyze(vacancy, result.resume_id, self.store)
            hr_finding = self.hr_agent.analyze(vacancy, result.resume_id, self.store)

            # "Дебаты": модератор видит оба независимых заключения и
            # явно называет согласие/противоречие между ними.
            debate_notes = run_debate(technical_finding, hr_finding, self.llm)

            resume_text = self.store.get_full_text(result.resume_id)
            skills_check = tools.check_mandatory_skills(resume_text, vacancy.required_skills)
            exp_years = tools.extract_experience_years(resume_text)

            final_score = round(
                WEIGHTS["similarity"] * result.similarity_score * 100
                + WEIGHTS["technical"] * technical_finding.score
                + WEIGHTS["hr"] * hr_finding.score,
                1,
            )

            report = CandidateReport(
                candidate_name=result.candidate_name,
                resume_id=result.resume_id,
                vacancy_title=vacancy.title,
                similarity_score=round(result.similarity_score * 100, 1),
                experience_years=exp_years,
                matched_skills=skills_check["matched"],
                missing_skills=skills_check["missing"],
                findings=[technical_finding, hr_finding],
                final_score=final_score,
                verdict=_verdict(final_score),
                debate_notes=debate_notes,
            )
            reports.append(report)

        reports.sort(key=lambda r: r.final_score, reverse=True)
        return reports
