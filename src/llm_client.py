"""
Абстракция над LLM. Контракт: LLMClient.complete(system_prompt, user_prompt) -> str.

Почему так:
- может не быть доступа в интернет или API-ключа —
  RuleBasedLLM работает полностью офлайн и без задержек,
  но генерирует текст по тем же данным (навыки, опыт, риски), которые
  использовал бы настоящий LLM-агент.
- Если ключ есть — OpenAICompatibleLLM отправляет запрос на любой
  OpenAI-совместимый эндпоинт (OpenAI API, локальный Ollama с
  `ollama serve` + OpenAI-совместимый роут, OpenRouter и т.д.).
"""
from __future__ import annotations
import os
import textwrap


class LLMClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class RuleBasedLLM(LLMClient):
    """Детерминированный офлайн-'LLM': собирает связный текст-заключение
    из фактов, переданных в user_prompt (в виде размеченных секций
    FACTS: ... ). Не использует сеть и не требует API-ключа.

    Это позволяет полностью прогнать мультиагентный пайплайн офлайн —
    удобно для воспроизводимой проверки проекта. Архитектура при
    этом никак не завязана на эту реализацию: смените LLM_BACKEND в
    config.py на "openai", и агенты начнут использовать настоящую LLM.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        role = self._detect_role(system_prompt)
        facts = self._parse_facts(user_prompt)

        if role == "moderator":
            return self._render_moderator(facts)
        if role == "технический скринер":
            return self._render_technical(facts)
        return self._render_hr(facts)

    @staticmethod
    def _detect_role(system_prompt: str) -> str:
        sp = system_prompt.lower()
        if "модератор" in sp:
            return "moderator"
        if "техническ" in sp:
            return "технический скринер"
        return "hr-аналитик"

    @staticmethod
    def _parse_facts(user_prompt: str) -> dict:
        facts = {}
        for line in user_prompt.split("\n"):
            if ":" in line:
                k, _, v = line.partition(":")
                facts[k.strip().lower()] = v.strip()
        return facts

    def _render_technical(self, f: dict) -> str:
        matched = f.get("совпавшие навыки (точное совпадение)", "нет данных")
        semantic = f.get("найдено семантическим дозапросом (перефразировано)", "—")
        missing = f.get("отсутствующие навыки", "нет данных")
        exp = f.get("опыт (лет)", "нет данных")
        parts = [
            f"Кандидат подтверждает {matched if matched and matched != '—' else 'ограниченный набор'} "
            f"из ключевых требований вакансии точным совпадением.",
        ]
        if semantic and semantic != "—":
            parts.append(f"Дополнительно семантический дозапрос по векторному индексу нашёл "
                          f"перефразированные упоминания: {semantic} — формально в резюме "
                          f"другими словами, но по смыслу совпадает.")
        if missing and missing != "—":
            parts.append(f"Ни точным, ни семантическим поиском не подтверждены: {missing} — "
                          f"это стоит уточнить на техническом интервью.")
        parts.append(f"Формальный стаж по датам в резюме: ~{exp} лет.")
        return " ".join(parts)

    def _render_hr(self, f: dict) -> str:
        job_changes = f.get("количество мест работы", "нет данных")
        gaps = f.get("разрывы в стаже", "не обнаружены")
        mgmt = f.get("управленческий опыт", "не найден")
        parts = [
            f"Количество упомянутых мест работы: {job_changes}.",
            f"Разрывы в стаже: {gaps}.",
            f"Управленческий опыт: {mgmt}.",
        ]
        return " ".join(parts)

    def _render_moderator(self, f: dict) -> str:
        tech_score = float(f.get("скор технического скринера", 0) or 0)
        hr_score = float(f.get("скор hr-аналитика", 0) or 0)
        tech_risks = f.get("риски по мнению технического скринера", "нет")
        hr_risks = f.get("риски по мнению hr-аналитика", "нет")

        has_tech_risks = tech_risks and tech_risks != "нет"
        has_hr_risks = hr_risks and hr_risks != "нет"
        gap = abs(tech_score - hr_score)

        if not has_tech_risks and not has_hr_risks:
            return ("Оба агента сходятся во мнении: значимых рисков не выявлено ни по "
                    "техническому стеку, ни по карьерной динамике — заключения согласованы.")
        if has_tech_risks and has_hr_risks:
            return (f"Оба агента независимо отмечают риски (технические: {tech_risks}; "
                    f"HR: {hr_risks}) — заключения согласуются в том, что кандидата "
                    f"стоит внимательно проверить на интервью по обоим направлениям.")
        if has_tech_risks and gap > 25:
            return (f"Противоречие между агентами: Технический Скринер отмечает риски "
                    f"({tech_risks}) при скоре {tech_score}, тогда как HR-Аналитик не видит "
                    f"проблем в карьерной динамике (скор {hr_score}) — сильный кандидат по "
                    f"софт-скиллам, но со слабостями в стеке, стоит уточнить на техническом интервью.")
        if has_hr_risks and gap > 25:
            return (f"Противоречие между агентами: HR-Аналитик отмечает риски карьерной "
                    f"динамики ({hr_risks}) при скоре {hr_score}, тогда как Технический "
                    f"Скринер уверен в стеке (скор {tech_score}) — сильный технический "
                    f"профиль при рисках в динамике карьеры, стоит уточнить причины на интервью.")
        return (f"Один из агентов отмечает риск ({tech_risks if has_tech_risks else hr_risks}), "
                f"но расхождение в скорах некритично ({tech_score} vs {hr_score}) — существенных "
                f"противоречий между агентами нет.")


class OpenAICompatibleLLM(LLMClient):
    """Клиент для реального LLM через любой OpenAI-совместимый API.
    (для локального Ollama достаточно OPENAI_BASE_URL=http://localhost:11434/v1
    и произвольного значения ключа)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        base_url = os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


def get_default_llm_client() -> LLMClient:
    """Выбирает реализацию по переменной окружения LLM_BACKEND
    (по умолчанию — офлайн rule-based, чтобы проект гарантированно
    запускался)."""
    backend = os.environ.get("LLM_BACKEND", "rule_based")
    if backend == "openai":
        return OpenAICompatibleLLM(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    return RuleBasedLLM()
