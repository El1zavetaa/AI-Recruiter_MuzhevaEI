"""
Роли агентов.

BaseAgent задаёт общий контракт:
    agent.analyze(vacancy, resume_id, store) -> AgentFinding

Каждый агент использует инструменты из tools.py, чтобы получить факты —
включая, при необходимости, ДОЗАПРОС дополнительных чанков через
semantic_recheck_skill (agentic tool-calling: агент сам решает, когда
недостаточно точного текстового совпадения и нужно свериться с векторным
поиском) — а затем LLM (llm_client.py), чтобы превратить факты в связный
текст. Такое разделение (tools = факты, LLM = формулировка) снижает
галлюцинации: LLM не придумывает цифры и не видит сырой текст резюме
напрямую, только структурированную сводку фактов.
"""
from __future__ import annotations
import re
from typing import List

from .schemas import Vacancy, AgentFinding
from .llm_client import LLMClient
from .vector_store import VectorStore
from . import tools


class BaseAgent:
    name: str = "BaseAgent"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(self, vacancy: Vacancy, resume_id: str, store: VectorStore) -> AgentFinding:
        raise NotImplementedError


class TechnicalScreenerAgent(BaseAgent):
    """Анализирует стек, хард-скиллы и масштаб проектов кандидата."""
    name = "Технический Скринер"

    SYSTEM_PROMPT = (
        "Ты — технический скринер в ИТ-компании. Оцени соответствие "
        "хард-скиллов кандидата требованиям вакансии на основе ФАКТОВ ниже. "
        "Не придумывай навыки, которых нет в фактах."
    )

    def analyze(self, vacancy: Vacancy, resume_id: str, store: VectorStore) -> AgentFinding:
        resume_text = store.get_full_text(resume_id)
        skills_check = tools.check_mandatory_skills(resume_text, vacancy.required_skills)
        nice_check = tools.check_mandatory_skills(resume_text, vacancy.nice_to_have_skills)
        exp_years = tools.extract_experience_years(resume_text)

        # --- Agentic tool-calling: по каждому формально отсутствующему
        # обязательному навыку агент ДОЗАПРАШИВАЕТ у векторного индекса,
        # нет ли перефразированного упоминания (например, "оркестрация
        # контейнеров" вместо буквального "Kubernetes"), прежде чем
        # окончательно считать навык отсутствующим. ---
        confirmed_missing: List[str] = []
        semantically_found: List[str] = []
        for skill in skills_check["missing"]:
            evidence = tools.semantic_recheck_skill(store, resume_id, skill)
            if evidence:
                semantically_found.append(skill)
            else:
                confirmed_missing.append(skill)

        matched_skills = skills_check["matched"] + semantically_found

        user_prompt = (
            f"Вакансия: {vacancy.title}\n"
            f"Совпавшие навыки (точное совпадение): {', '.join(skills_check['matched']) or '—'}\n"
            f"Найдено семантическим дозапросом (перефразировано): {', '.join(semantically_found) or '—'}\n"
            f"Отсутствующие навыки: {', '.join(confirmed_missing) or '—'}\n"
            f"Опыт (лет): {exp_years}\n"
            f"Требуемый минимальный опыт: {vacancy.min_experience_years}\n"
        )
        narrative = self.llm.complete(self.SYSTEM_PROMPT, user_prompt)

        matched_ratio = len(matched_skills) / max(1, len(vacancy.required_skills))
        exp_ratio = min(1.0, exp_years / vacancy.min_experience_years) if vacancy.min_experience_years else 1.0
        score = round(100 * (0.7 * matched_ratio + 0.3 * exp_ratio), 1)

        strengths = [f"Подтверждён навык: {s}" for s in skills_check["matched"]]
        strengths += [f"Подтверждён навык (найден семантическим поиском, "
                      f"дословно в резюме не встречается): {s}" for s in semantically_found]

        weaknesses = [f"Отсутствует желательный (необязательный) навык: {s}"
                      for s in nice_check["missing"]]

        risks = [f"Не найдено подтверждения обязательного навыка (ни точным, "
                 f"ни семантическим поиском): {s}" for s in confirmed_missing]
        if vacancy.min_experience_years and exp_years < vacancy.min_experience_years:
            risks.append(
                f"Формальный опыт ({exp_years} лет) ниже требуемого "
                f"({vacancy.min_experience_years} лет)"
            )

        return AgentFinding(
            agent_name=self.name,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            notes=narrative,
        )


class HRAnalystAgent(BaseAgent):
    """Оценивает динамику карьерного роста, софт-скиллы, частоту смены
    работы и управленческий опыт (red flags)."""
    name = "HR-Аналитик"

    SYSTEM_PROMPT = (
        "Ты — HR-аналитик. Оцени карьерную динамику кандидата на основе "
        "ФАКТОВ ниже: частоту смены мест работы, разрывы в стаже, признаки "
        "управленческого опыта. Не придумывай факты, которых нет во входных данных."
    )

    _JOB_TITLE_MARKERS = re.compile(
        r"(?:^|\n)\s*(?:\d{4}\s*[-–—]\s*(?:\d{4}|н\.?в\.?|наст[а-я]*))",
        re.IGNORECASE,
    )
    _MANAGEMENT_KEYWORDS = [
        "тимлид", "team lead", "teamlead", "руководитель", "управлял командой",
        "руководство командой", "менеджер проекта", "cto", "head of", "тим-лид",
        "led a team", "manager",
    ]

    def analyze(self, vacancy: Vacancy, resume_id: str, store: VectorStore) -> AgentFinding:
        resume_text = store.get_full_text(resume_id)
        job_blocks = self._JOB_TITLE_MARKERS.findall(resume_text)
        num_jobs = max(1, len(job_blocks))
        exp_years = tools.extract_experience_years(resume_text)
        avg_tenure = round(exp_years / num_jobs, 1) if num_jobs else exp_years

        mgmt_evidence = tools.find_keyword_evidence(resume_text, self._MANAGEMENT_KEYWORDS)
        has_management = bool(mgmt_evidence)

        # Дозапрос: если явных ключевых слов управленческого опыта нет,
        # проверяем семантически ("руководство", "лидерство проекта" и т.п.),
        # прежде чем окончательно относить это к минусам.
        management_semantic_evidence = None
        if not has_management:
            management_semantic_evidence = tools.semantic_recheck_skill(
                store, resume_id, "управление командой, лидерство, менеджмент проекта", threshold=0.15
            )

        user_prompt = (
            f"Количество мест работы: {num_jobs}\n"
            f"Средний срок на одном месте (лет): {avg_tenure}\n"
            f"Разрывы в стаже: {'не обнаружены' if avg_tenure > 0 else 'требуют проверки'}\n"
            f"Управленческий опыт: {'да (явно)' if has_management else ('да (косвенно, по семантическому поиску)' if management_semantic_evidence else 'не найден')}\n"
        )
        narrative = self.llm.complete(self.SYSTEM_PROMPT, user_prompt)

        risks: List[str] = []
        strengths: List[str] = []
        weaknesses: List[str] = []
        score = 70.0

        if avg_tenure < 1.0 and num_jobs > 1:
            risks.append(
                f"Частая смена мест работы: в среднем ~{avg_tenure} года на одном месте"
            )
            score -= 20
        else:
            strengths.append(f"Стабильная карьера: в среднем ~{avg_tenure} лет на одном месте")
            score += 10

        if has_management:
            strengths.append("Есть явные упоминания управленческого/лидерского опыта")
            score += 15
        elif management_semantic_evidence:
            strengths.append("Есть косвенные признаки управленческого опыта (найдено "
                              "семантическим поиском, явных ключевых слов нет)")
            score += 7
        else:
            weaknesses.append("Нет явных или косвенных подтверждений управленческого опыта — "
                               "не блокер для non-lead позиций")

        score = max(0.0, min(100.0, score))

        return AgentFinding(
            agent_name=self.name,
            score=round(score, 1),
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            notes=narrative,
        )
