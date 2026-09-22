"""
Демонстрационный запуск полного цикла AI-Recruiter:

    вакансия -> векторизация базы резюме -> semantic_search (top-K) ->
    -> Технический Скринер + HR-Аналитик -> финальный скор -> отчёт .md/.html
"""
from __future__ import annotations
import argparse
import glob
import json
import os

from src.schemas import Vacancy
from src.embeddings import get_default_embedding_model
from src.vector_store import VectorStore
from src.parsing import load_resume_file
from src.orchestrator import AgentOrchestrator
from src.report import save_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VACANCY = os.path.join(BASE_DIR, "data", "vacancies", "senior_mlops.json")
RESUMES_DIR = os.path.join(BASE_DIR, "data", "resumes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def load_vacancy(path: str) -> Vacancy:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Vacancy(**data)


def build_store(resumes_dir: str) -> VectorStore:
    store = VectorStore(embedding_model=get_default_embedding_model())
    all_chunks = []
    for path in sorted(glob.glob(os.path.join(resumes_dir, "*.txt"))):
        all_chunks.extend(load_resume_file(path))
    store.build_index(all_chunks)
    print(f"[embedding] Проиндексировано резюме: "
          f"{len(set(c.resume_id for c in all_chunks))}, чанков: {len(all_chunks)}")
    return store


def main():
    parser = argparse.ArgumentParser(description="AI-Recruiter: демо полного цикла")
    parser.add_argument("--vacancy", default=DEFAULT_VACANCY, help="путь к JSON-файлу вакансии")
    parser.add_argument("--resumes-dir", default=RESUMES_DIR, help="папка с резюме (.txt/.pdf/.docx)")
    parser.add_argument("--top-k", type=int, default=5, help="сколько кандидатов анализировать глубоко")
    parser.add_argument("--out-dir", default=REPORTS_DIR, help="куда сохранить отчёт")
    args = parser.parse_args()

    vacancy = load_vacancy(args.vacancy)
    print(f"[vacancy] {vacancy.title}: обязательные навыки — {', '.join(vacancy.required_skills)}")

    store = build_store(args.resumes_dir)

    orchestrator = AgentOrchestrator(store)
    reports = orchestrator.rank_candidates(vacancy, top_k=args.top_k)

    print("\n=== Ранжированная выдача ===")
    for i, r in enumerate(reports, 1):
        print(f"{i}. {r.candidate_name:20s} | скор: {r.final_score:5.1f}% | "
              f"близость: {r.similarity_score:5.1f}% | вердикт: {r.verdict}")

    saved_paths = save_report(vacancy, reports, args.out_dir)
    print("\n[report] Сохранено:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
