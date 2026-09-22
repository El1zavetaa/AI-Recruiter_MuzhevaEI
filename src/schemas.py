"""
Единый контракт передачи данных между модулями.
Все объекты — простые dataclass'ы, которые легко сериализуются в JSON.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Vacancy:
    id: str
    title: str
    description: str
    required_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    min_experience_years: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class ResumeChunk:
    """Смысловой фрагмент резюме — единица индексации в векторной базе."""
    resume_id: str
    candidate_name: str
    chunk_id: str
    text: str

    def to_dict(self):
        return asdict(self)


@dataclass
class SearchResult:
    """Результат semantic_search — одна строка выдачи векторного поиска."""
    resume_id: str
    candidate_name: str
    similarity_score: float
    matched_chunk: str


@dataclass
class AgentFinding:
    """Единый формат вывода любого агента (Технический Скринер / HR-Аналитик).

    Три категории соответствуют формату карточки из ТЗ (Плюсы / Минусы / Риски):
    strengths — подтверждённые сильные стороны; weaknesses — некритичные
    пробелы (например, отсутствует необязательный навык или явных
    подтверждений управленческого опыта нет, но это не блокер); risks —
    красные флаги, которые могут стать причиной отказа."""
    agent_name: str
    score: float  # 0..100
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class CandidateReport:
    """Финальная карточка кандидата — то, что видит рекрутер."""
    candidate_name: str
    resume_id: str
    vacancy_title: str
    similarity_score: float
    experience_years: float
    matched_skills: List[str]
    missing_skills: List[str]
    findings: List[AgentFinding]
    final_score: float
    verdict: str  # "Рекомендован к приглашению" / "В резерв" / "Отказ"
    debate_notes: Optional[str] = None  # синтез/сверка выводов агентов друг с другом

    def to_dict(self):
        d = asdict(self)
        return d
