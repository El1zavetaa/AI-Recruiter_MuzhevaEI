"""
Parsing & Embedding Pipeline — часть 2: превращение текста в векторы.

Интерфейс EmbeddingModel одинаков независимо от реализации, поэтому
"движок" эмбеддингов можно менять, не трогая остальной код (vector_store,
tools, orchestrator).

По умолчанию используется TfidfEmbeddingModel — работает полностью
офлайн (без интернета и без скачивания весов), детерминированна и
достаточна для семантического ранжирования в учебном прототипе.

Подключение нейросетевой модели (bge-large-en/ru,
text-embedding-3-large и т.п.) —  тот же интерфейс в классе
SentenceTransformerEmbeddingModel (см. ниже, закомментировано) и
подставьте её в config.py.
"""
from __future__ import annotations
import re
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "для",
    "этого", "чтобы", "тот", "им", "более", "всегда", "конечно", "всю",
    "между", "г", "гг", "лет", "год", "года", "работа", "работал", "работала",
}


def _tokenize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ+#\.\s]", " ", text)
    tokens = [t for t in text.split() if t not in RU_STOPWORDS and len(t) > 1]
    return " ".join(tokens)


class EmbeddingModel:
    """Абстрактный интерфейс — любая реализация должна иметь fit/encode."""

    def fit(self, corpus: List[str]) -> None:
        raise NotImplementedError

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbeddingModel(EmbeddingModel):
    """Офлайн-эмбеддинги на основе TF-IDF + косинусное сходство.
    """

    def __init__(self, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(
            preprocessor=_tokenize,
            ngram_range=(1, 2),
            max_features=max_features,
        )
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self.vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EmbeddingModel не обучен. Сначала вызовите .fit(corpus).")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray()


# --- Реальная нейросетевая модель (требует `pip install sentence-transformers`
#     и интернет при первом запуске — скачивание весов) ---
class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Обёртка над bge-large-en/ru / любой моделью с sentence-transformers.
    Реализует тот же интерфейс fit/encode, что и TfidfEmbeddingModel, —
    остальной код (VectorStore, tools.py, orchestrator.py) не знает и не
    должен знать, какая реализация используется."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbeddingModel требует пакет sentence-transformers: "
                "pip install sentence-transformers. При первом запуске также нужен "
                "интернет — модель скачивает веса с HuggingFace Hub."
            ) from e
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def fit(self, corpus: List[str]) -> None:
        # Предобученная модель, дообучение на корпусе не требуется —
        # метод сохранён только ради единого интерфейса с TfidfEmbeddingModel.
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


def get_default_embedding_model() -> EmbeddingModel:
    """Выбирает реализацию по переменной окружения EMBEDDING_BACKEND
    (по умолчанию — офлайн TF-IDF, чтобы проект гарантированно запускался
    у проверяющего без интернета и без скачивания весов).

    EMBEDDING_BACKEND=sentence_transformers [EMBEDDING_MODEL=BAAI/bge-m3]
    переключает на настоящую нейросетевую модель."""
    import os
    backend = os.environ.get("EMBEDDING_BACKEND", "tfidf")
    if backend == "sentence_transformers":
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        return SentenceTransformerEmbeddingModel(model_name=model_name)
    return TfidfEmbeddingModel()
