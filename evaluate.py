"""
Оценка качества семантического поиска (Parsing & Embedding Pipeline +
semantic_search) на демо-базе резюме, сгенерированной generate_demo_data.py.

Идея эксперимента: у каждого сгенерированного резюме есть скрытая
"истинная метка" — трек (`mlops` / `backend_go` / `data_scientist` /
`irrelevant`), который не участвует ни в эмбеддингах, ни в поиске.
Каждая из 3 тестовых вакансий соответствует одному треку. Считаем,
насколько хорошо векторный поиск поднимает "релевантных" кандидатов
в топ выдачи — это объективная, воспроизводимая метрика качества (в
отличие от текстовых заключений агентов, которые сложно оценить
численно без разметки экспертами).

Метрики:
    Precision@K  — доля кандидатов из top-K, чей истинный трек совпадает
                   с треком вакансии.
    MRR          — 1 / позиция первого релевантного кандидата в выдаче
                   (Mean Reciprocal Rank), усреднённая по вакансиям.

Запуск:
    python evaluate.py
    python evaluate.py --top-k 5
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
from src import tools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUMES_DIR = os.path.join(BASE_DIR, "data", "resumes")
VACANCIES_DIR = os.path.join(BASE_DIR, "data", "vacancies")

# Соответствие файла вакансии её "истинному" треку в ground truth
VACANCY_TRACK = {
    "senior_mlops.json": "mlops",
    "backend_go.json": "backend_go",
    "data_scientist.json": "data_scientist",
}


def load_ground_truth(resumes_dir: str) -> dict:
    gt_path = os.path.join(resumes_dir, "_ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"{gt_path} не найден. Сначала запустите: python generate_demo_data.py"
        )
    with open(gt_path, encoding="utf-8") as f:
        return json.load(f)


def build_store(resumes_dir: str) -> VectorStore:
    store = VectorStore(embedding_model=get_default_embedding_model())
    chunks = []
    for path in sorted(glob.glob(os.path.join(resumes_dir, "*.txt"))):
        chunks.extend(load_resume_file(path))
    store.build_index(chunks)
    return store


def evaluate(top_k: int = 5):
    ground_truth = load_ground_truth(RESUMES_DIR)
    store = build_store(RESUMES_DIR)

    print(f"{'Вакансия':<28} {'Precision@' + str(top_k):<14} {'MRR':<8} детали")
    print("-" * 80)

    precisions, mrrs = [], []
    for fname, expected_track in VACANCY_TRACK.items():
        with open(os.path.join(VACANCIES_DIR, fname), encoding="utf-8") as f:
            vacancy = Vacancy(**json.load(f))

        results = tools.semantic_search(store, vacancy, top_k=top_k)
        hits = [ground_truth.get(r.resume_id) == expected_track for r in results]

        precision = sum(hits) / len(hits) if hits else 0.0
        rr = 0.0
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                rr = 1.0 / rank
                break

        precisions.append(precision)
        mrrs.append(rr)

        detail = ", ".join(
            f"{r.candidate_name}({'✓' if h else '✗'})" for r, h in zip(results, hits)
        )
        print(f"{vacancy.title:<28} {precision:<14.2f} {rr:<8.2f} {detail}")

    print("-" * 80)
    print(f"{'Среднее по 3 вакансиям':<28} {sum(precisions)/len(precisions):<14.2f} "
          f"{sum(mrrs)/len(mrrs):<8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка качества semantic_search по ground truth")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    evaluate(top_k=args.top_k)
