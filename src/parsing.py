"""
Parsing & Embedding Pipeline — часть 1: извлечение чистого текста
из файлов резюме/вакансий (PDF / DOCX / TXT) и нарезка на чанки.
"""
import os
import re
from typing import List

from .schemas import ResumeChunk


def extract_text_from_pdf(path: str) -> str:
    import pypdf
    text_parts = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def extract_text_from_docx(path: str) -> str:
    import docx
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(path: str) -> str:
    """Определяет формат файла по расширению и извлекает текст."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    if ext in (".txt", ".md"):
        return extract_text_from_txt(path)
    raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, resume_id: str, candidate_name: str,
                max_chars: int = 700, overlap: int = 100) -> List[ResumeChunk]:
    """
    Нарезает текст на смысловые фрагменты (по абзацам, с укрупнением до
    max_chars и небольшим перекрытием, чтобы не терять контекст на стыках).
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    # перекрытие между соседними чанками для сохранения контекста
    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0:
            tail = chunks[i - 1][-overlap:]
            c = f"{tail} {c}"
        overlapped.append(c)

    return [
        ResumeChunk(
            resume_id=resume_id,
            candidate_name=candidate_name,
            chunk_id=f"{resume_id}_{i}",
            text=c,
        )
        for i, c in enumerate(overlapped)
    ]


def load_resume_file(path: str, candidate_name: str = None) -> List[ResumeChunk]:
    """Полный цикл: файл резюме -> чистый текст -> список чанков."""
    resume_id = os.path.splitext(os.path.basename(path))[0]
    raw = extract_text(path)
    text = clean_text(raw)
    name = candidate_name or _guess_name(text) or resume_id
    return chunk_text(text, resume_id=resume_id, candidate_name=name)


def _guess_name(text: str) -> str:
    """Первая непустая строка часто содержит имя кандидата."""
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line.split()) <= 4 and not any(ch.isdigit() for ch in line):
            return line
    return ""
