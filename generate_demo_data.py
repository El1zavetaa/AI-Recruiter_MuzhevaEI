"""
Воспроизводимый генератор демонстрационной базы резюме.

Каждое резюме собирается из пулов компонентов (имена, компании, роли,
навыки, формулировки обязанностей) случайным образом, но с ФИКСИРОВАННЫМ
seed — поэтому повторный запуск даёт побайтово идентичный результат.

Запуск:
    python generate_demo_data.py                # перезаписывает data/resumes/
    python generate_demo_data.py --seed 123 --n 15 --out data/resumes_v2

Данные полностью синтетические (имена и компании вымышлены) и не содержат
персональной информации реальных людей — сделаны специально для
воспроизводимого учебного прогона AI-Recruiter.
"""
from __future__ import annotations
import argparse
import json
import os
import random

FIRST_NAMES_M = ["Иван", "Павел", "Алексей", "Дмитрий", "Сергей", "Егор", "Андрей", "Кирилл", "Николай", "Роман"]
FIRST_NAMES_F = ["Мария", "Анна", "Ольга", "Екатерина", "Наталья", "Светлана", "Юлия", "Ирина"]
LAST_NAMES_M = ["Петров", "Соловьёв", "Волков", "Кузнецов", "Смирнов", "Морозов", "Белов", "Орлов", "Фёдоров", "Лебедев"]
LAST_NAMES_F = ["Соколова", "Кузьмина", "Никитина", "Волкова", "Морозова", "Белова", "Орлова", "Фёдорова"]

COMPANIES = [
    "ООО «ВысокоНагрузка»", "AI Cloud Solutions", "TechCorp", "ДатаСтрим",
    "ФинТех Диджитал", "Небольшая ИТ-компания", "Северный Софт", "Облачные Технологии",
    "РитейлТех", "МедиаПлатформа", "ЛогистикПро", "СтартапЛаб",
]
EDU_PLACES = [
    "СПбГУ, факультет прикладной математики", "МГУ, факультет ВМК",
    "МФТИ, физтех-школа прикладной математики и информатики",
    "Региональный технический университет, бакалавриат по информатике",
    "ИТМО, факультет информационных технологий",
    "Высшая школа экономики, программная инженерия",
]

# Каждый трек: пул навыков (для check_mandatory_skills) и шаблоны формулировок обязанностей.
# Формулировки обязанностей намеренно написаны в виде именных групп
# ("Разработка ...", "Настройка ...") без согласования по роду/лицу —
# это распространённый стиль в резюме и позволяет комбинировать буллеты
# с любым именем без грамматических ошибок.
TRACKS = {
    "mlops": {
        "titles": ["MLOps-инженер", "Senior MLOps Engineer", "ML Platform Engineer"],
        "skills": ["Python", "Kubernetes", "Docker", "CI/CD", "Triton Inference Server",
                    "Kafka", "Terraform", "Prometheus", "AWS"],
        "bullets": [
            "Разработка и поддержка архитектуры распределённых систем инференса ML-моделей.",
            "Внедрение Kubernetes-кластера для оркестрации сервисов инференса в высоконагруженном проекте.",
            "Настройка Triton Inference Server для развёртывания моделей компьютерного зрения и NLP.",
            "Проектирование CI/CD пайплайна для деплоя ML-моделей.",
            "Настройка мониторинга моделей в проде с помощью Prometheus.",
        ],
    },
    "backend_go": {
        "titles": ["Backend-разработчик", "Go Developer", "Senior Backend Engineer"],
        "skills": ["Go", "PostgreSQL", "Docker", "Kubernetes", "gRPC", "Redis", "Kafka", "MySQL"],
        "bullets": [
            "Разработка микросервисов на Go с использованием gRPC.",
            "Проектирование схемы PostgreSQL для высоконагруженного сервиса.",
            "Внедрение кэширования на Redis для снижения нагрузки на БД.",
            "Написание unit- и интеграционных тестов для критичных сервисов.",
            "Контейнеризация сервисов с помощью Docker.",
        ],
    },
    "data_scientist": {
        "titles": ["Data Scientist", "NLP Engineer", "ML Engineer"],
        "skills": ["Python", "PyTorch", "NLP", "Transformers", "SQL", "Pandas", "scikit-learn"],
        "bullets": [
            "Разработка моделей классификации текста на базе Transformers.",
            "Построение пайплайнов обучения моделей на PyTorch.",
            "Проведение A/B-тестирования ML-моделей в проде.",
            "Анализ данных с помощью Pandas и SQL.",
        ],
    },
    "irrelevant": {
        "titles": ["Менеджер по продажам", "Frontend-разработчик (верстальщик)", "SMM-специалист"],
        "skills": ["HTML", "CSS", "Excel", "1С", "Photoshop"],
        "bullets": [
            "Ведение переговоров с клиентами и заключение договоров.",
            "Вёрстка лендингов на HTML и CSS.",
            "Ведение социальных сетей компании, подготовка контент-плана.",
        ],
    },
}

MANAGEMENT_BULLETS = [
    "Тимлид команды из {n} человек.",
    "Руководство командой разработки из {n} инженеров.",
    "Менторинг {n} junior-специалистов.",
]


def _gen_person(rng: random.Random):
    gender = rng.choice(["m", "f"])
    if gender == "m":
        name = f"{rng.choice(FIRST_NAMES_M)} {rng.choice(LAST_NAMES_M)}"
    else:
        name = f"{rng.choice(FIRST_NAMES_F)} {rng.choice(LAST_NAMES_F)}"
    return name


def _gen_job_block(rng: random.Random, track: dict, start_year: int, end_year: int | None):
    title = rng.choice(track["titles"])
    company = rng.choice(COMPANIES)
    period = f"{start_year} – {'н.в.' if end_year is None else end_year}"
    bullets = rng.sample(track["bullets"], k=min(len(track["bullets"]), rng.randint(2, 4)))
    if rng.random() < 0.3:
        bullets.append(rng.choice(MANAGEMENT_BULLETS).format(n=rng.randint(2, 6)))
    lines = [f"{period} {company}, {title}"] + bullets
    return "\n".join(lines)


def generate_resume(rng: random.Random, resume_id: str):
    """Возвращает (текст_резюме, название_трека). Трек — это скрытая
    'истинная метка' (какой вакансии кандидат реально релевантен),
    используемая только для оценки качества ранжирования (evaluate.py),
    сама LLM/векторный поиск её не видят."""
    track_name = rng.choices(
        population=["mlops", "backend_go", "data_scientist", "irrelevant"],
        weights=[0.3, 0.3, 0.25, 0.15],
        k=1,
    )[0]
    track = TRACKS[track_name]
    name = _gen_person(rng)

    current_year = 2026
    num_jobs = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2], k=1)[0]
    # Иногда моделируем "частую смену работы" (риск для HR-Аналитика)
    job_hopper = rng.random() < 0.2

    jobs = []
    year_cursor = current_year
    for i in range(num_jobs):
        span = rng.choice([1, 1, 2]) if job_hopper else rng.choice([2, 3, 4, 5])
        start = year_cursor - span
        end = None if i == 0 else year_cursor
        jobs.append(_gen_job_block(rng, track, start, end))
        year_cursor = start

    edu_start = year_cursor - rng.randint(4, 5)
    education = f"{edu_start} – {year_cursor} {rng.choice(EDU_PLACES)}"

    skills_line = ", ".join(sorted(set(track["skills"])))

    parts = [
        name, "",
        "Опыт работы", "",
        "\n\n".join(jobs), "",
        "Образование",
        education, "",
        "Навыки",
        skills_line,
    ]
    return "\n".join(parts) + "\n", track_name


def main():
    parser = argparse.ArgumentParser(description="Генератор демо-базы резюме (воспроизводимо, по seed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=12, help="сколько резюме сгенерировать")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "resumes"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Чистим только сгенерированные ранее .txt и ground-truth файл,
    # чтобы не задеть случайно посторонние файлы
    for f in os.listdir(args.out):
        if f.endswith(".txt") or f == "_ground_truth.json":
            os.remove(os.path.join(args.out, f))

    ground_truth = {}
    for i in range(1, args.n + 1):
        resume_id = f"candidate_{i:02d}"
        text, track = generate_resume(rng, resume_id)
        with open(os.path.join(args.out, f"{resume_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        ground_truth[resume_id] = track

    gt_path = os.path.join(args.out, "_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"Сгенерировано {args.n} резюме в {args.out} (seed={args.seed})")
    print(f"Ground truth (для оценки качества ранжирования) сохранён в {gt_path}")


if __name__ == "__main__":
    main()
