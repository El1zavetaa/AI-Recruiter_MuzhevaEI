"""
Векторная база данных
"""
from __future__ import annotations
import pickle
from typing import List
import numpy as np

from .schemas import ResumeChunk, SearchResult
from .embeddings import EmbeddingModel


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.chunks: List[ResumeChunk] = []
        self.matrix: np.ndarray | None = None

    def build_index(self, chunks: List[ResumeChunk]) -> None:
        """Обучает эмбеддинг-модель на корпусе и индексирует все чанки."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embedding_model.fit(texts)
        self.matrix = self.embedding_model.encode(texts)

    def add_documents(self, chunks: List[ResumeChunk]) -> None:
        """Добавляет новые документы в уже построенный индекс."""
        if self.matrix is None:
            self.build_index(chunks)
            return
        new_vectors = self.embedding_model.encode([c.text for c in chunks])
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, new_vectors])

    def query(self, query_text: str, top_k: int = 10) -> List[SearchResult]:
        """Семантический поиск top-K наиболее близких резюме по вакансии.

        """
        if self.matrix is None or len(self.chunks) == 0:
            return []
        q_vec = self.embedding_model.encode([query_text])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        best_per_candidate = {}
        for chunk, score in zip(self.chunks, sims):
            key = chunk.resume_id
            if key not in best_per_candidate or score > best_per_candidate[key][0]:
                best_per_candidate[key] = (float(score), chunk)

        ranked = sorted(best_per_candidate.values(), key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                resume_id=chunk.resume_id,
                candidate_name=chunk.candidate_name,
                similarity_score=round(score, 4),
                matched_chunk=chunk.text[:300],
            )
            for score, chunk in ranked[:top_k]
        ]

    def query_within_resume(self, resume_id: str, query_text: str, top_k: int = 3) -> List[SearchResult]:
        """Точечный поиск ВНУТРИ чанков одного конкретного кандидата.

        Это и есть инструмент, которым агенты в оркестраторе "запрашивают
        через инструменты дополнительные чанки из базы для уточнения
        деталей" (см. ТЗ) — вместо того, чтобы агент сразу видел весь текст
        резюме целиком, он может прицельно спросить "есть ли здесь что-то
        по теме X?" и получить только релевантные фрагменты с их скором
        сходства. В частности это ловит перефразированные навыки, которые
        точный текстовый поиск (check_mandatory_skills) не находит —
        например, запрос "Kubernetes" может семантически совпасть с
        фрагментом "оркестрация контейнеров в проде", даже если слово
        "Kubernetes" в резюме не встречается буквально.
        """
        own_chunks = [c for c in self.chunks if c.resume_id == resume_id]
        if not own_chunks or self.matrix is None:
            return []

        own_indices = [i for i, c in enumerate(self.chunks) if c.resume_id == resume_id]
        own_matrix = self.matrix[own_indices]

        q_vec = self.embedding_model.encode([query_text])
        sims = cosine_similarity(q_vec, own_matrix)[0]

        ranked = sorted(zip(sims, own_chunks), key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                resume_id=chunk.resume_id,
                candidate_name=chunk.candidate_name,
                similarity_score=round(float(score), 4),
                matched_chunk=chunk.text[:300],
            )
            for score, chunk in ranked[:top_k]
        ]

    def get_full_text(self, resume_id: str) -> str:
        return "\n".join(c.text for c in self.chunks if c.resume_id == resume_id)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "matrix": self.matrix,
                         "embedding_model": self.embedding_model}, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls(data["embedding_model"])
        store.chunks = data["chunks"]
        store.matrix = data["matrix"]
        return store
