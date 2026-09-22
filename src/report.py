"""
Модуль генерации отчёта: превращает список CandidateReport в
наглядную аналитическую карточку .md / .html и сводную таблицу по вакансии.
"""
from __future__ import annotations
import os
from typing import List
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .schemas import CandidateReport, Vacancy

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


def render_markdown(vacancy: Vacancy, reports: List[CandidateReport]) -> str:
    template = _env.get_template("report.md.j2")
    return template.render(vacancy=vacancy, reports=reports)


def render_html(vacancy: Vacancy, reports: List[CandidateReport]) -> str:
    template = _env.get_template("report.html.j2")
    return template.render(vacancy=vacancy, reports=reports)


def save_report(vacancy: Vacancy, reports: List[CandidateReport], out_dir: str,
                 formats: tuple = ("md", "html")) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() else "_" for c in vacancy.title).strip("_")
    saved = []
    if "md" in formats:
        path = os.path.join(out_dir, f"report_{safe_title}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_markdown(vacancy, reports))
        saved.append(path)
    if "html" in formats:
        path = os.path.join(out_dir, f"report_{safe_title}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_html(vacancy, reports))
        saved.append(path)
    return saved
