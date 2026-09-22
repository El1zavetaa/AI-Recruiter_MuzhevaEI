"""
Библиотека аналитических инструментов (Tools), которые вызывают агенты.

"""
from __future__ import annotations
import re
from datetime import datetime
from typing import List, Dict

from .vector_store import VectorStore
from .schemas import SearchResult, Vacancy


def semantic_search(store: VectorStore, vacancy: Vacancy, top_k: int = 10) -> List[SearchResult]:
    """Инструмент 1: векторный поиск top-K резюме, ближайших к вакансии."""
    query = f"{vacancy.title}. {vacancy.description}. " \
            f"Требуемые навыки: {', '.join(vacancy.required_skills)}."
    return store.query(query, top_k=top_k)


# Паттерн вида "2019-2022", "2019—2022", "2019 - н.в.", "янв 2020 – май 2023"
_YEAR_RANGE_RE = re.compile(
    r"(19|20)\d{2}\s*[-–—]\s*((19|20)\d{2}|н\.?в\.?|наст(оящее)?\s*время|по\s*настоящее)",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def extract_experience_years(resume_text: str, current_year: int = None) -> float:
    """Инструмент 2: грубый математический подсчёт коммерческого опыта
    по диапазонам дат, встречающимся в тексте резюме.
    """
    current_year = current_year or datetime.now().year
    total_months = 0
    seen_spans = set()

    for match in _YEAR_RANGE_RE.finditer(resume_text):
        span_text = match.group(0)
        all_years_4digit = re.findall(r"(?:19|20)\d{2}", span_text)
        start_year = int(all_years_4digit[0])

        if re.search(r"н\.?в\.?|наст", span_text, re.IGNORECASE):
            end_year = current_year
        elif len(all_years_4digit) > 1:
            end_year = int(all_years_4digit[-1])
        else:
            end_year = current_year

        if (start_year, end_year) in seen_spans:
            continue
        seen_spans.add((start_year, end_year))

        if end_year >= start_year and (end_year - start_year) <= 45:
            total_months += (end_year - start_year) * 12

    return round(total_months / 12, 1)


def check_mandatory_skills(resume_text: str, required_skills: List[str]) -> Dict[str, List[str]]:
    """Инструмент 3: быстрая проверка наличия критических навыков
    (стоп-слов) вакансии в тексте резюме.

    Возвращает {"matched": [...], "missing": [...]}.
    """
    text_lower = resume_text.lower()
    matched, missing = [], []
    for skill in required_skills:
        pattern = re.escape(skill.lower())
        if re.search(pattern, text_lower):
            matched.append(skill)
        else:
            missing.append(skill)
    return {"matched": matched, "missing": missing}


def semantic_recheck_skill(store: VectorStore, resume_id: str, skill: str,
                            threshold: float = 0.12) -> str | None:
    """Инструмент 4 (agentic tool-calling): точечный семантический дозапрос
    одного конкретного кандидата по одному конкретному навыку.

    Вызывается агентом, когда `check_mandatory_skills` НЕ нашёл точного
    текстового совпадения — чтобы проверить, не упомянут ли навык
    перефразированно (например, вакансия требует "Kubernetes", а в резюме
    написано "оркестрация контейнеров в высоконагруженном проде"). Именно
    этот шаг реализует поведение из ТЗ: "агенты могут запрашивать через
    инструменты дополнительные чанки из базы для уточнения деталей".

    Возвращает текст наиболее релевантного чанка, если сходство выше
    порога, иначе None (навык действительно не подтверждён).
    """
    results = store.query_within_resume(resume_id, skill, top_k=1)
    if results and results[0].similarity_score >= threshold:
        return results[0].matched_chunk
    return None


def find_keyword_evidence(resume_text: str, keywords: List[str], window: int = 60) -> Dict[str, str]:
    """Вспомогательный инструмент: возвращает короткую цитату вокруг
    найденного ключевого слова — агенты используют это как "доказательство"
    для отчёта (а не просто бинарный факт найдено/не найдено)."""
    evidence = {}
    text = resume_text
    lower = text.lower()
    for kw in keywords:
        idx = lower.find(kw.lower())
        if idx != -1:
            start = max(0, idx - window // 2)
            end = min(len(text), idx + len(kw) + window // 2)
            evidence[kw] = text[start:end].strip().replace("\n", " ")
    return evidence
