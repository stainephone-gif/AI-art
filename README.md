# Теоретические рамки сознания и трудная проблема в текстах об AI-арте — пайплайн v3

Воспроизводимый пайплайн разметки корпуса из 211 русскоязычных описаний работ AI-арта
(`combined_ai_preprocessed.xlsx`: 64 текста из `explication_table.xlsx`, 147 из `mirdb_arts.xlsx`).

Что делает v3 (по ТЗ `TZ_classifier_v3`):

- **(а)** консервативная разметка теоретических рамок (COMP, IIT, PRED, GWT, ENACT, PAN, EMERG, UND) с явными критериями исключения; свидетельство засчитывается только при метафорическом переносе между областью ума и областью машины/материала и обязательном `mapping`;
- **(б)** отдельный слой трудной проблемы (`hp_level` 0/1/2, `hp_stance`, `hp_subject`), словарь опыта у человека не считается;
- **(в)** мультимодельная оценка: ≥3 модели через OpenRouter × 2 повтора, temperature 0, seed; Cohen κ, Krippendorff α, Jaccard, матрицы согласия, consensus (≥2 из 3), legacy-прогоны для документирования;
- **(г)** слепая таблица для ручной разметки двумя кодировщиками и расчёт κ человек/человек, человек/модель, человек/consensus;
- **(д)** статистика с базовой линией, раздельно по источнику и типу автора: χ² с остатками и Monte-Carlo p, со-встречаемость (ожидаемое, PMI, permutation), Fisher exact для theory × hp, логистическая регрессия «вторичная метка ~ длина» до и после нормировки.

## Структура

```
config.yaml                 # модели, пороги, пути, seed — единственный конфиг
prompts/
  codebook.md               # определения и критерии исключения; единственный источник текста
  theory_v3.md              # промпт рамок: {{codebook:...}} подставляется из codebook.md
  hardproblem_v1.md         # промпт слоя трудной проблемы
pipeline/
  common.py                 # конфиг, корпус, промпты (hash), LLM-клиент с кешем, meta прогона
  verify_quotes.py          # верификация цитат (как в v2) + отбор независимых фрагментов
  aggregate.py              # валидация evidence, n_spans/density/primary/secondary, отчёты
  classify_theory.py        # прогон рамок          → runs/<slug>_<date>/theory_r<k>.json
  classify_hardproblem.py   # прогон трудной проблемы → runs/<slug>_<date>/hardproblem_r<k>.json
  agreement.py              # κ, α, Jaccard, consensus.csv, legacy
  stats.py                  # χ², остатки, PMI, permutation, Fisher, логит; csv/md/png
  export_annotation.py      # annotation_sheet.xlsx (слепая, с кодбуком и author_type)
  import_annotation.py      # κ людей, human_labels.csv, disagreements.csv
  mock_llm.py               # заглушка для тестов (llm.provider: mock), не классификатор
runs/                       # результаты по прогонам (+ legacy_deepseek-v3.2_20251204 — декабрьский прогон v1)
reports/                    # сводные таблицы, графики, consensus.csv, annotation_sheet.xlsx
legacy/                     # код v1/v2 (enhanced_classifier.py, metaphor_analyzer.py …), только для истории
test_classifier.py          # регрессионные тесты (оффлайн + live-контрпримеры)
```

Каждый результат хранит `model`, `prompt_hash` (первые 16 символов sha256 отрендеренного промпта, включая кодбук), `codebook_hash`,
`timestamp`, `code_version` (git-коммит), а `runs/<run>/meta.json` — снимок конфига и параметры генерации.

## Установка

```bash
pip install -r requirements.txt
cp .env.example .env            # вписать OPENROUTER_API_KEY
```

Python ≥ 3.10. Модели задаются в `config.yaml` (`models:`), по умолчанию `anthropic/claude-sonnet-4.6`,
`openai/gpt-4.1`, `deepseek/deepseek-v3.2`; используйте версионированные ID, где они есть.

## Запуск

```bash
# 1. Прогоны (все модели × 2 повтора; прерванный прогон продолжается с места остановки благодаря кешу)
python -m pipeline.classify_theory      --config config.yaml
python -m pipeline.classify_hardproblem --config config.yaml
#    варианты: --model claude-sonnet-4.6  --repeats 1  --limit 10  --ids 52,0  --run-name my_run  --force

# 2. Длинные таблицы + отчёт о фильтрации свидетельств + «длина × число меток»
python -m pipeline.aggregate  --config config.yaml

# 3. Согласие и consensus (включая legacy-прогоны из config.yaml)
python -m pipeline.agreement  --config config.yaml

# 4. Слепая таблица для ручной разметки (нужны прогоны из п. 1; --all — включить все тексты)
python -m pipeline.export_annotation --config config.yaml

# 5. После заполнения таблицы кодировщиками A и B
python -m pipeline.import_annotation --config config.yaml --input reports/annotation_sheet_filled.xlsx

# 6. Статистика на consensus-разметке и на человеческой
python -m pipeline.stats --config config.yaml --labels consensus
python -m pipeline.stats --config config.yaml --labels human

# Тесты
pytest -q                                  # оффлайн
OPENROUTER_API_KEY=... pytest -m live -q   # контрпримеры на реальной модели (TEST_MODEL=... чтобы сменить модель)
```

Проверить пайплайн без API: поставьте `llm.provider: mock` в копии конфига (заглушка отвечает по регулярным
выражениям и годится только для проверки механики, не для разметки).

## Как считается разметка

1. Модель возвращает только список `evidence` (`class`, `span`, `mapping{source,target}`, `level`,
   `exclusion_checked`) и `notes`. Оценок, confidence и primary у модели не запрашивается.
2. Валидация (`aggregate.validate_theory_response`): отбрасываются фрагменты без `mapping`, без
   `exclusion_checked: true`, короче `min_span_words`, не найденные в тексте (`verify_quotes`: точное
   вхождение или доля токенов ≥ 0.7 с префиксным матчингом, как в v2) и перекрывающиеся с уже отобранными
   того же класса. Доля отброшенных по каждой причине — в `reports/filter_report.md`.
3. Агрегация: `n_spans[c]`, `density[c] = n_spans / words × 100`; **primary** = argmax `n_spans`
   (при равенстве — по покрытию текста фрагментами, затем по порядку классов; флаг `primary_tie`);
   **secondary** = все классы с `n_spans ≥ 2` и `density ≥ 1.0` (полный вектор в `labels`, `is_hybrid`);
   `secondary_raw`/`is_hybrid_raw` — без порога плотности, для отчёта о влиянии длины;
   **confidence** — из числа фрагментов primary (1 / 2 / ≥3 → low / medium / high); **UND**, если фрагментов нет.
4. Слой трудной проблемы: `hp_level` модели принимается только при наличии верифицированных фрагментов
   не о человеке; иначе 0 (флаг `level_reset_no_verified_spans`). `hp_stance` — только при level ≥ 1.
5. Consensus: метка модели = большинство её повторов; класс присваивается, если совпал у ≥2 из 3 моделей,
   иначе UND с флагом `disputed`. Multi-label consensus — классы, присутствующие у ≥2 моделей.

## Выходные файлы

| Файл | Содержимое |
|---|---|
| `runs/<slug>_<date>/theory_r<k>.json` | по тексту: evidence (kept/drop_reason, verification), n_spans, density, primary, secondary, labels, confidence, n_words |
| `runs/<slug>_<date>/hardproblem_r<k>.json` | hp_level, hp_stance, hp_subject, hp_spans, флаги |
| `runs/<slug>_<date>/meta.json` | модель, хеши промптов/кодбука, дата, версия кода, снимок конфига |
| `reports/results_theory_long.csv`, `results_hardproblem_long.csv` | одна строка на текст × прогон |
| `reports/filter_report.md` | доля отфильтрованных свидетельств по причинам |
| `reports/length_report.md` | Spearman ρ длины и числа меток до/после нормировки, медианы длины |
| `reports/agreement_summary.md`, `agreement_pairwise_kappa_*.csv`, `confusion_*` | κ, α (с bootstrap CI), test–retest, Jaccard, legacy |
| `reports/consensus.csv` | consensus_primary, disputed, consensus_labels, per-model метки, hp_level, hp_stance, legacy primary |
| `reports/annotation_sheet.xlsx` | слепая таблица (листы instructions, annotation, codebook, author_type); ключ выборки — `annotation_sample_key.csv` (кодировщикам не показывать) |
| `reports/human_agreement.md`, `human_labels.csv`, `disagreements.csv`, `author_type.csv` | результат импорта ручной разметки |
| `reports/stats_<labels>/` | distribution_*, cooccurrence_*, theory_x_hp_*, chi2_*, length_logit, png |

## Что изменилось относительно v2 (`legacy/`)

- Удалены веса уровней/типов метафор, множитель ×1.2 за ontological, «КРИТИЧЕСКИ ВАЖНО», паттерн
  «техтермин + художественный глагол → метаметафора»; уровень остался только как аннотация.
- Текст берётся из `descr_clean` (v2 склеивал `descr + descr_en + descr_clean + descr_lemmas`, утраивая текст).
- Модель не выбирает класс и не даёт confidence; всё вычисляется из верифицированных фрагментов.
- Прежние результаты (`analysis_results_20251204_144611`, deepseek-v3.2, промпт v1) перенесены в
  `runs/legacy_deepseek-v3.2_20251204/` и участвуют в `agreement.py` как legacy. Майский прогон
  (claude-sonnet-4.6, промпт v2) в репозитории отсутствует: положите его
  `enhanced_classification_results.json` по пути из `config.yaml` (`legacy_runs`), и он подхватится.
- `llm_class-env/` убран из репозитория; ноутбук `covert.ipynb` удалён.

## Критерии приёмки (ТЗ п. 8) → где проверять

1. Воспроизведение с нуля: `python -m pipeline.classify_theory --config config.yaml` и `classify_hardproblem`, результат в `runs/`.
2. Каждый evidence с верифицированной цитатой и mapping; доля отфильтрованных — `reports/filter_report.md`; тест `test_validation_filters_by_reason`.
3. Контрпримеры (эмульсия в темноте ≠ ENACT, древесина хранит аромат ≠ PAN, переводили вместе с ИИ ≠ COMP, + 10) — `test_classifier.py::test_live_counterexamples` (live), формулировки исключений — `test_theory_prompt_contains_exclusions_from_spec` (оффлайн).
4. «Демон» → `hp_level = 2`, словарь опыта у человека → 0 — `test_live_hardproblem`, `test_live_demon_from_corpus`; логика сброса — `test_hp_human_only_span_resets_level`.
5. κ/α по трём моделям и legacy — `python -m pipeline.agreement`.
6. `annotation_sheet.xlsx` слепая, с кодбуком — `python -m pipeline.export_annotation`, тест `test_export_annotation_is_blind_and_deterministic`.
7. Таблица пар с ожидаемыми значениями и p — `reports/stats_consensus/cooccurrence_all.md`.
