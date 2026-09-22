# AI-Recruiter — мультиагентная система семантического ранжирования и скоринга кандидатов

Прототип, который решает проблему keyword-matching в HR-скрининге: находит
резюме, семантически близкие к вакансии (даже если формулировки не совпадают
дословно), а затем прогоняет топ-кандидатов через двух специализированных
LLM-агентов — **Технического Скринера** и **HR-Аналитика** — и выдаёт
рекрутеру структурированную карточку с процентом соответствия, плюсами,
минусами, рисками и вердиктом.

Проект настроен на запуск с **настоящей LLM через локальный Ollama** и **нейросетевыми эмбеддингами**
(`sentence-transformers`, `BAAI/bge-m3`) — см. раздел «Запуск» ниже.

Архитектура при этом не привязана к конкретному провайдеру: LLM и
эмбеддинг-модель подключаются через интерфейсы `LLMClient`
(`src/llm_client.py`) и `EmbeddingModel` (`src/embeddings.py`), выбор
реализации — через переменные окружения. Помимо `OpenAICompatibleLLM` и
`SentenceTransformerEmbeddingModel` там же tcnm офлайн-реализации тех
же интерфейсов (`RuleBasedLLM`, `TfidfEmbeddingModel`) — они ничего не
ломают и не участвуют в обычном запуске, но на них построены
детерминированные unit-тесты (`tests/`), которым не нужны сеть и запущенный
Ollama.

> Итоговый отчёт по проекту (постановка задачи, описание решения, результаты
> экспериментов с метриками, выводы) — [`report/otchet.html`](./report/otchet.html).

## Как это устроено

```
Вакансия (JSON) ──┐
                   ▼
        ┌─────────────────────┐
        │ Parsing & Embedding │  извлечение текста (PDF/DOCX/TXT) → чанкинг →
        │      Pipeline       │  эмбеддинги (sentence-transformers / TF-IDF для тестов) → VectorStore
        └──────────┬──────────┘
                   ▼
        semantic_search(vacancy, top_k)   ← Tools Library
                   ▼
        ┌─────────────────────┐
        │  Agent Orchestrator │  для каждого топ-кандидата, независимо:
        │                     │  1) Технический Скринер (стек, хард-скиллы, стаж;
        │                     │     дозапрашивает чанки через semantic_recheck_skill)
        │                     │  2) HR-Аналитик (динамика карьеры, red flags;
        │                     │     тоже дозапрашивает чанки при необходимости)
        └──────────┬──────────┘
                   ▼
        Дебаты: модератор сверяет 2 независимых заключения,
        явно называет согласие/противоречие
                   ▼
        Финальный взвешенный скор + вердикт
                   ▼
        Модуль генерации отчёта → report.md / report.html
        (Плюсы / Минусы / Риски / Дебаты агентов)
```

Ключевая идея разделения ответственности: **инструменты (tools.py) вычисляют
факты** (навыки, стаж, частота смены работы) детерминированным кодом, а
**LLM только формулирует текст на основе этих фактов** — это резко снижает
риск галлюцинаций (LLM не может "придумать" навык, которого нет в фактах).

Агенты не просто пассивно получают весь текст резюме — они **сами решают,
когда дозапросить дополнительные данные**: если точный текстовый поиск не
нашёл обязательный навык буквально, агент вызывает
`tools.semantic_recheck_skill(store, resume_id, skill)`, который ищет
семантически близкий фрагмент ТОЛЬКО внутри чанков этого кандидата (например,
"оркестрация контейнеров" может семантически совпасть с "Kubernetes", даже
если слово в резюме не встречается). Это и есть agentic tool-calling —
дозапрос чанков из базы для уточнения деталей, а не заранее заданный
единственный проход по всему тексту.

### Структура репозитория

```
ai-recruiter/
├── src/
│   ├── schemas.py       # единый JSON-контракт: Vacancy, ResumeChunk, AgentFinding, CandidateReport
│   ├── parsing.py        # извлечение текста из PDF/DOCX/TXT + чанкинг
│   ├── embeddings.py     # интерфейс EmbeddingModel: SentenceTransformer (neural, по умолчанию для запуска) или TF-IDF (офлайн, для тестов)
│   ├── vector_store.py   # лёгкая векторная БД (замена Chroma/FAISS): add_documents/query/query_within_resume
│   ├── tools.py          # semantic_search, extract_experience_years, check_mandatory_skills, semantic_recheck_skill, find_keyword_evidence
│   ├── llm_client.py     # абстракция LLM: OpenAICompatibleLLM (Ollama, по умолчанию для запуска) или офлайн RuleBasedLLM (для тестов)
│   ├── agents.py         # TechnicalScreenerAgent, HRAnalystAgent (с дозапросом чанков)
│   ├── debate.py          # шаг "дебатов": модератор сверяет заключения агентов
│   ├── orchestrator.py   # AgentOrchestrator — ранжирование + дебаты + агрегация финального скора
│   └── report.py         # рендер отчёта в .md / .html (Jinja2)
├── templates/             # report.md.j2, report.html.j2
├── data/
│   ├── vacancies/         # 3 тестовые вакансии (JSON)
│   └── resumes/            # демо-резюме, генерируются generate_demo_data.py (+ _ground_truth.json)
├── tests/                  # unit-тесты (unittest / pytest-совместимые)
├── notebooks/demo.ipynb   # полный цикл от загрузки вакансии до отчёта, с пояснениями
├── generate_demo_data.py   # воспроизводимый генератор демо-базы резюме (фиксированный seed)
├── evaluate.py              # метрики качества ранжирования (Precision@K, MRR) на ground truth
├── run_demo.py             # CLI: запуск всего пайплайна одной командой
├── report/otchet.html      # итоговый отчёт по проекту (постановка задачи, решение, метрики, выводы)
├── reports/                # сюда сохраняются сгенерированные карточки кандидатов
└── requirements.txt
```

## Установка

```bash
git clone https://github.com/El1zavetaa/AI-Recruiter_MuzhevaEI
cd ai-recruiter
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install openai sentence-transformers
```

### Ollama (локальная LLM)

```bash
# 1. установить Ollama (ollama.com) и один раз скачать модель
ollama pull llama3
```

Ollama должна быть запущена (`ollama serve`, обычно поднимается автоматически
после установки) — она поднимает OpenAI-совместимый API на
`http://localhost:11434/v1`, на который и настроен проект по умолчанию.

### Переменные окружения

**Windows / PowerShell:**

```powershell
$env:LLM_BACKEND="openai"
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_API_KEY="ollama"
$env:LLM_MODEL="llama3"

$env:EMBEDDING_BACKEND="sentence_transformers"
$env:EMBEDDING_MODEL="BAAI/bge-m3"
```

**Linux / macOS / bash:**

```bash
export LLM_BACKEND=openai
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama        
export LLM_MODEL=llama3

export EMBEDDING_BACKEND=sentence_transformers
export EMBEDDING_MODEL=BAAI/bge-m3
```

Эти переменные нужно задать один раз в текущей сессии терминала — все команды
ниже (`generate_demo_data.py`, `run_demo.py`, `evaluate.py`,
`notebooks/demo.ipynb`) читают их через `get_default_llm_client()` /
`get_default_embedding_model()` и не требуют правок кода.

> Код агентов (`agents.py`, `orchestrator.py`, `vector_store.py`) при этом не
> меняется — он работает через интерфейсы `LLMClient` и `EmbeddingModel` и не
> знает, какой провайдер за ними стоит.

## Запуск

Сгенерировать демо-базу резюме (воспроизводимо, фиксированный seed=42):

```bash
python generate_demo_data.py
```

Прогнать демо-вакансию (Senior MLOps) по демо-базе и получить отчёт (первый
запуск с `sentence-transformers` скачает веса `BAAI/bge-m3` с HuggingFace Hub
— нужен интернет один раз, дальше модель кешируется локально):

```bash
python run_demo.py
```

Другая вакансия / другое количество кандидатов для глубокого анализа:

```bash
python run_demo.py --vacancy data/vacancies/backend_go.json --top-k 3
python run_demo.py --vacancy data/vacancies/data_scientist.json --top-k 5
```

Результат:
- в консоли — ранжированный список кандидатов с итоговым скором и вердиктом;
- в `reports/` — `report_<Название_вакансии>.md` и `.html` с детальным
  разбором по каждому кандидату (Плюсы / Минусы / Риски по обоим агентам +
  итог дебатов — согласие или противоречие между ними).

Пример вывода:

```
1. Андрей Смирнов       | скор:  80.6% | близость:  39.6% | вердикт: Рекомендован к приглашению
2. Светлана Волкова     | скор:  78.9% | близость:  46.2% | вердикт: Рекомендован к приглашению
3. Екатерина Фёдорова   | скор:  78.8% | близость:  33.4% | вердикт: Рекомендован к приглашению
```

### Ноутбук с пошаговым разбором

```bash
jupyter notebook notebooks/demo.ipynb
```

Показывает каждый шаг пайплайна отдельно (индексация → semantic_search →
работа агентов → финальный скор → сохранение отчёта) с комментариями.

### Оценка качества ранжирования (метрики)

```bash
python evaluate.py            # Precision@5, MRR по всем трём вакансиям
python evaluate.py --top-k 3  # то же для top-3
```

Считается по ground truth-меткам, которые `generate_demo_data.py` сохраняет отдельно от текста
резюме (`data/resumes/_ground_truth.json`) специально для этой проверки. Подробности и
интерпретация результатов — в `report/otchet.html`.

### Свои резюме и вакансии

- Резюме: положите `.txt`, `.pdf` или `.docx` файлы в `data/resumes/` —
  `run_demo.py` подхватит `.txt` автоматически; для PDF/DOCX используйте
  `src.parsing.load_resume_file(path)` напрямую (см. ноутбук).
- Вакансия: создайте JSON по образцу `data/vacancies/senior_mlops.json`
  (`id`, `title`, `description`, `required_skills`, `nice_to_have_skills`,
  `min_experience_years`).

### Офлайн-режим (для тестов / без Ollama)

Если Ollama не установлена или нет интернета для скачивания весов
`sentence-transformers`, можно не задавать переменные окружения из раздела
«Переменные окружения» вообще — `LLM_BACKEND`/`EMBEDDING_BACKEND` по
умолчанию не заданы, и `get_default_llm_client()` /
`get_default_embedding_model()` (`src/llm_client.py`, `src/embeddings.py`)
вернут офлайн-реализации тех же интерфейсов: `RuleBasedLLM` (детерминированный
генератор текста по фактам, без сети и ключей) и `TfidfEmbeddingModel`
(TF-IDF + косинусное сходство). Именно на них построены unit-тесты
(`tests/`) — чтобы `python -m unittest discover -s tests` гарантированно
проходил без запущенной Ollama.

Код агентов (`agents.py`, `orchestrator.py`, `vector_store.py`) при этом не
меняется в обоих случаях — он работает через интерфейсы `LLMClient` и
`EmbeddingModel` и не знает, какая реализация за ними стоит.

## Тесты

```bash
python -m unittest discover -s tests -v
```

30 тестов покрывают: подсчёт опыта по датам, проверку обязательных навыков,
агрегацию semantic_search по кандидату (а не по чанку), семантический
дозапрос отдельного навыка внутри одного резюме (`semantic_recheck_skill`),
поведение обоих агентов на сильном/слабом резюме (включая разделение
находок на Плюсы/Минусы/Риски), шаг "дебатов" (согласие vs противоречие
между агентами), сортировку и агрегацию оркестратора, корректность рендера
отчёта.

## Метрики и ограничения прототипа

- **Эмбеддинги** — нейросетевая модель `sentence-transformers`
  (`BAAI/bge-m3` по умолчанию, `EMBEDDING_BACKEND=sentence_transformers`);
  для офлайн-тестов используется TF-IDF + косинусное сходство
  (`TfidfEmbeddingModel`) — см. раздел «Офлайн-режим». Интерфейс
  `EmbeddingModel` (`fit`/`encode`) один и тот же для обеих реализаций.
- **LLM** — Llama-3 через локальный Ollama (OpenAI-совместимый API,
  `LLM_BACKEND=openai` + `OPENAI_BASE_URL`);
  для офлайн-тестов используется `RuleBasedLLM` — см. раздел
  «Офлайн-режим».
- **Подсчёт опыта** — эвристика по регулярным выражениям над диапазонами дат
  в тексте резюме, не идеальна на нестандартных форматах дат.
- **Данные** — 12 демо-резюме генерируются процедурно скриптом
  `generate_demo_data.py` с фиксированным `seed=42` (полностью синтетические,
  не содержат персональных данных реальных людей); повторный запуск даёт
  побайтово идентичный результат. 3 тестовые вакансии — вручную заданные
  JSON-фикстуры (аналог тест-кейсов, не "датасет").
- **Веса финального скора** заданы в `orchestrator.py` (`WEIGHTS`):
  30% семантическая близость, 45% технический скор, 25% HR-скор — подобраны
  экспертно, не откалиброваны на размеченных данных (для прототипа это
  ожидаемо; для продакшена нужна калибровка на исторических решениях
  рекрутеров).

## Воспроизводимость

Демо-база резюме генерируется с фиксированным `seed=42`
(`generate_demo_data.py`), поэтому набор данных одинаков между запусками.
Итоговый текст в карточках кандидата при этом зависит от выбранного `LLM_BACKEND`:
с Ollama LLM генерирует текст заново при каждом
запуске (temperature=0.2), формулировки могут немного отличаться при
неизменных фактах и скорах; в офлайн-режиме (`RuleBasedLLM` + TF-IDF, без
заданных `LLM_BACKEND`/`EMBEDDING_BACKEND`) всё полностью детерминировано —
повторный запуск даёт побайтово идентичный результат, поэтому именно этот
режим используется в unit-тестах.
