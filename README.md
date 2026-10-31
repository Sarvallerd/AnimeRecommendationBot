# AnimeRecommendationBot

Telegram-бот для личных чатов на русском: поиск аниме → подтверждение MAL ID → до пяти рекомендаций, оценки аниме 1–10, полезности 0–5 и текстовые отзывы. Rust обслуживает Telegram и PostgreSQL; Python заранее готовит неизменяемый JSON-пакет из Neelagiri 2025, алиасов Nafiul 2026 и GloVe; рекомендуемый пакет строит соседей ItemKNN по оценкам пользователей. Персонализации в v1 нет.

## Код и документация

| Путь | Назначение |
| --- | --- |
| `bot/` | Rust-бот, диалог, PostgreSQL и миграции |
| `src/recsys/cli/`, `src/recsys/contracts/` | Python CLI, проверка артефактов и происхождения |
| `src/recsys/data/`, `src/recsys/content/`, `src/recsys/release/` | Источники, векторы и сборка выпуска |
| `src/recsys/research/`, `src/recsys/historical/` | Подготовка, модели, оценка, кампания и исторические методы |
| `src/recsys/resources/` | 14 закреплённых JSON/prompt ресурсов: реестры, протоколы, кандидаты, ожидания и lock |
| `recsys/tests/`, `contracts/`, `deploy/`, `dev/` | Тесты, межъязыковой контракт, эксплуатация и проверка окружения |

Справочники: [архитектура и разработка](https://app.notion.com/p/3f03908d6b1f81a1863bee31d0e66ec9), [данные и выпуски](https://app.notion.com/p/3fb3908d6b1f8136a1dfff3c4a12108e), [исследования](https://app.notion.com/p/3fb3908d6b1f81989401e704cbc4c97b), [бот и PostgreSQL](https://app.notion.com/p/3fb3908d6b1f81d582eadb84bca22973), [эксплуатация и приёмка](https://app.notion.com/p/3fb3908d6b1f8165a819fb91f377e4b6), [архив](https://app.notion.com/p/3fb3908d6b1f81788538fd13e11401a7). Правила работы — [AGENTS.md](AGENTS.md); версия layout и исторические хеши — [migration record](docs/migrations/arb054-recsys-layout-v2.json).

## Разработка и проверки

Нужны Python 3.12, uv 0.11.12, Rust 1.95.0, C-компилятор, `pkg-config` и OpenSSL headers. Команды выполняются из корня репозитория. Ruff 0.17.1 проверяет формат, импорты и аннотации аргументов; mypy 2.4.0 строго проверяет фактические типы.

```bash
uv sync --locked
uv run --locked recsys --help
uv run --locked check_bundle tests/fixtures/bundle
UV_PROJECT_ENVIRONMENT=.tools/lint-venv uv run --locked --only-group lint ruff check .
UV_PROJECT_ENVIRONMENT=.tools/lint-venv uv run --locked --only-group lint ruff format --check .
UV_PROJECT_ENVIRONMENT=.tools/typing-venv uv sync --locked --no-install-project --no-default-groups --group typing --extra quality --extra torch-cpu
UV_PROJECT_ENVIRONMENT=.tools/typing-venv uv run --locked --no-sync python -m mypy
ARB_TEST_TORCH=1 uv run --locked --no-editable --extra quality --extra torch-cpu python -m unittest discover -s recsys/tests -p 'test_*.py'
cargo test --locked
python3 scripts/check_repo_hygiene.py
python3 deploy/tests/level1-acceptance.py
```

Последняя команда требует Docker: FakeTelegram, временная PostgreSQL и малые фикстуры A→B→A. Это не полная или живая приёмка. Полный запуск ML-02 выполняется отдельно после проверки инженерной кампании; выбранной модели/N пока нет. Научные входы, результаты, секреты и локальные инструменты остаются вне Git.

Пакет 0.6.0 записывает валидационные прогнозы по умолчанию в lossless `scores.bin` (`arb-score-blocks-v1`, prediction schema 2). Один popularity-вектор применяется ко всем нулевым seed-запросам и popularity arm, а signed int64 и bitmap сохраняют точные баллы и отличие null от нуля. `research predict-validation --score-format jsonl-v1` выбирает прежний JSONL; исходный `ml02-cpu-v1` остаётся семивариантным JSONL-профилем. Новый восьмивариантный `ml02-native-v3` использует бинарный формат; исторический `ml02-native-v2` требует checkout `a06e369b21f96c5d8ee37765e3116448e1d12b6e` и не возобновляется в новом producer. Новый план, отчёт и предсказание имеют версии 5, 4 и 2. Fresh admission вычисляется из проверенных C/Q/Q0 и верхней границы байтов каждого варианта с запасом 20 GiB, margin 4 GiB и условной оценкой 8 GiB прочих артефактов. Проверяются оба тома и stop 24 GiB; это не гарантия завершения. Формат, формула, границы и синтетический benchmark описаны в [score blocks v1](docs/score-blocks-v1.md), происхождение — в [миграции ARB-065](docs/migrations/arb065-binary-predictions-v1.json). Full ML-02 quality и выбор N пока не завершены.

Для нового native-запуска подготовьте данные новым producer, установите ровно один PyTorch extra (ниже показан CUDA) и задайте физический том. Admission вычисляется для конкретных размеров подготовленных данных и оставшихся вариантов; прежняя фиксированная оценка 192 GiB к этому профилю не относится.

```bash
uv sync --locked --no-editable --extra quality --extra torch-cuda
uv run --locked --no-editable --extra quality --extra torch-cuda recsys research campaign-run \
  --campaign ml02-native-v3 --native-train-device cuda:0 \
  --physical-storage-root /mnt/c --prepared-dir /absolute/prepared \
  --expected-prepared-identity sha256:... --code-revision FULL_40_HEX_SHA \
  --glove /absolute/glove.6B.300d.txt --output-dir /absolute/new-campaign
```

Команда не открывает test labels и не выбирает N. Протокол, лимиты и статус выполнения ведутся в [исследовательском справочнике](https://app.notion.com/p/3fb3908d6b1f81989401e704cbc4c97b) и [задаче ARB-065](https://app.notion.com/p/4063908d6b1f811581a7c93b9469ba3a).

Пакет 0.5.2 ускоряет validation evaluator на полном каталоге: проверка принадлежности MAL ID использует множество, а ранжирование выбирает десять лучших кандидатов после полной проверки вектора. На синтетических данных медиана полной оценки при C=28 955 и Q=41 снизилась с 1,882625 до 1,438217 с (в 1,31 раза быстрее); при C=21 и Q=1740 выросла с 0,559213 до 0,574044 с (на 2,65% медленнее). Это два чередующихся повтора для каждого сценария; они не доказывают завершение полной кампании ML-02 в прежнем часовом лимите. Маски, метрики, порядок ранжирования, форматы артефактов и лимиты кампании сохранены. Новый producer SHA требует свежей подготовки и нового запуска; результаты прежнего producer нельзя возобновлять под новой версией. Полное сравнение качества и выбор N пока не завершены. Связь с предыдущими байтами и версиями закреплена в [миграции ARB-064](docs/migrations/arb064-bounded-evaluation-v1.json).

Пакет 0.5.1 допустил до 8 MiB для публичной проекции и отчёта `ml02-native-v2`, включая полный проверяемый envelope. Научные форматы и лимиты вычислений не изменились; историческое изменение закреплено в [миграции ARB-063](docs/migrations/arb063-native-campaign-report-cap-v1.json).

Исторически пакет 0.5.0 добавил закрытый профиль `ml02-native-v2` полной validation-кампании: семь прежних CPU-моделей и предобъявленный PyTorch-кандидат. Его план/отчёт имели версии 4/3, а исходный ресурс и его SHA-256 закреплены в [миграции ARB-062](docs/migrations/arb062-native-validation-campaign-v2.json). Условная оценка 192 GiB относилась только к этому историческому профилю. Для восстановления используйте отдельный закреплённый checkout `a06e369b21f96c5d8ee37765e3116448e1d12b6e` и соответствующие ему подготовленные артефакты.

Пакет 0.4.0 добавляет предобъявленного PyTorch-кандидата `item-seed-mean-bpr-v1` для офлайн-обучения на train-only взаимодействиях. Он принимает до десяти MAL ID нового пользователя и возвращает один ограниченный запрос; веса остаются отдельным проверяемым `safetensors`-артефактом. Это инженерная синтетическая приёмка, а не выбор модели для выпуска. Обычная установка и CPU baseline не устанавливают и не импортируют PyTorch. Выберите ровно один optional extra: `torch-cpu` или `torch-cuda` (официальные индексы и версии зафиксированы в `pyproject.toml` и `uv.lock`). Перед установкой проверьте свободное место на томе с запасом не менее 20 GiB сверх ожидаемого объёма.

```bash
uv sync --locked --no-editable --extra quality --extra torch-cpu
uv run --locked --no-editable --extra quality --extra torch-cpu recsys research training-fit \
  --candidate item-seed-mean-bpr-v1 --prepared-dir /absolute/prepared \
  --expected-prepared-identity sha256:... --output-dir /absolute/new-fit-run \
  --code-revision FULL_40_HEX_SHA --train-device cpu
uv run --locked --no-editable --extra quality --extra torch-cpu recsys research training-predict \
  --prepared-dir /absolute/prepared --expected-prepared-identity sha256:... \
  --model-dir /absolute/new-fit-run/model --expected-model-identity sha256:... \
  --output-dir /absolute/new-prediction-run --code-revision FULL_40_HEX_SHA \
  --seed-mal-id 1 --seed-mal-id 2 --limit 5
```

Команды возвращают путь к опубликованному `run.json`; model identity — SHA-256 байтов `model/training-manifest.json` с префиксом `sha256:`. Для CUDA training укажите `--train-device cuda:0` в отдельном окружении `--extra torch-cuda`; CPU inference доступен через `--inference-device cpu`. Для CPU-операции отдельный worker запускается с пустым `CUDA_VISIBLE_DEVICES`, не меняя окружение родительского процесса; даже с CUDA-сборкой PyTorch его внутренний оптимизатор может проверять доступность CUDA, но приложение не запрашивает CUDA и не инициализирует её. Этот режим не гарантирует отсутствия обращения к драйверу; прямой вызов native API внутри уже запущенного процесса не изолирует его окружение. Один worker имеет фиксированные пределы 300 секунд, 4 GiB наблюдаемого RSS и 64 MiB артефакта, без автоматического перехода на CPU. CUDA admission требует 2 GiB свободной VRAM. Полный validation-запуск ML-02 уже разрешён и следует после принятия исправления 0.5.1; выбор N, публикация бота и rollout остаются отдельными задачами. Новые identities и сохранённые исторические hashes описаны в [миграции ARB-060](docs/migrations/arb060-native-pytorch-training-v1.json).

Пакет 0.3.0 сохраняет семь ML-02 моделей как CPU baselines. Параметр `--backend cpu` (по умолчанию) поддерживается для `research model-fit`, `predict-validation` и `campaign-run`; прежний `--backend gpu` возвращает `research_gpu_scoring_retired` без чтения входов и подготовки выходов. Команда `research gpu-preflight` также возвращает этот код и принимает прежние флаги только для диагностики. Для восстановления прежнего CuPy scoring используйте отдельный закреплённый checkout `fa983eaeeeeb41b41ac893b7de9827a40c25d27d`. Форматы новых CPU campaign plan/report — 3/2; старые артефакты не преобразуются. Точные связи версий и исходных файлов записаны в [миграции ARB-059](docs/migrations/arb059-cupy-scoring-retirement.json).

## Восстановление данных

В новой копии восстановите шесть указателей из `data/snapshots/` через существующий приватный DVC remote. Создайте отдельный пустой cache вне Git с правами 0700; ниже замените оба абсолютных пути. Настройки сохраняются только в игнорируемой `.dvc/config.local`.

```bash
data_tool() { UV_PROJECT_ENVIRONMENT=.tools/data-venv uv run --locked --only-group data "$@"; }
data_tool dvc config --local cache.dir /absolute/private/empty-cache
data_tool dvc config --local cache.type copy
data_tool dvc remote add --local --default local /absolute/private/existing-remote
data_tool dvc remote modify --local local jobs 2
data_tool dvc remote modify --local local verify true
chmod 600 .dvc/config.local
data_tool dvc pull -r local
```

Сверьте восстановленные байты с реестрами и release lock по справочнику данных. DVC checkout содержит входы; работающий бот использует отдельно опубликованный неизменяемый bundle. Локальный remote на том же диске не заменяет резервную копию.

## Принятый выпуск v1

Пакет 0.2 проверяет готовый v1, а свежая реконструкция требует исходного производителя. До запуска разместите проверенные base, aliases, Nafiul cache и GloVe в приватном `DATA` по справочнику данных; этот же каталог укажите как `ARB_DATA_DIR`. Замените абсолютные пути; новый reference checkout должен быть чистым.

```bash
DATA=/absolute/private/animebot/data
REFERENCE=/absolute/private/arb-reference-v1
git worktree add --detach "$REFERENCE" 1441573fbbbffc4693c7e1100aaae4458d9e88d4
(
  cd "$REFERENCE" &&
  test "$(git rev-parse HEAD)" = 1441573fbbbffc4693c7e1100aaae4458d9e88d4 &&
  test -z "$(git status --porcelain)" &&
  uv sync --python 3.12.14 --locked --no-editable --extra quality &&
  test "$(uv run --python 3.12.14 --locked --no-editable --extra quality recsys --version)" = "recsys 0.1.0" &&
  uv run --python 3.12.14 --locked --no-editable --extra quality recsys release prepare --release current --data-dir "$DATA"
)
```

Если Python отсутствует, установите `uv python install 3.12.14`. Сохраните исходные manifest/producer hashes. Ошибка `release_reference_producer_required` в новом дереве направляет к этой процедуре; восстановление не подтверждает новую полную приёмку. Старые ML-артефакты также остаются с исходным checkout.

## Запуск, обновление и откат

Нужны Docker Engine и Compose 2.30.0+. Храните настройки и данные вне временных worktree. Создайте `compose.env` по [образцу](deploy/compose.env.example), отдельные `secrets/bot.env` и `secrets/postgres.env` по [bot](deploy/bot.env.example) и [postgres](deploy/postgres.env.example). Каталог секретов — 0700, файлы — 0600; существующий `.env` сохраняйте.
В `compose.env` задайте абсолютные `ARB_DATA_DIR`, `ARB_BUNDLE_DIR`, `ARB_BOT_ENV`, `ARB_POSTGRES_ENV`, свои `ARB_UID/ARB_GID` (`id -u` и `id -g`) и `ARB_LOG=info`. Bundle доступен UID 10001: каталоги 0755, файлы 0644. В secret-файлах нужны `TELOXIDE_TOKEN`, `DATABASE_URL` с URL-кодированным паролем и `sslmode=disable`, `POSTGRES_PASSWORD`; raw значения пишутся без shell-кавычек.

```bash
PROJECT=animebot-local
SETTINGS=/absolute/private/animebot/compose.env
COMPOSE_FILE="$PWD/compose.yaml"
compose() { docker compose --env-file "$SETTINGS" -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }
compose --profile tools build builder bot
compose --profile tools run --rm --no-deps --entrypoint /usr/local/bin/prepare-artifacts builder
compose --profile tools run --rm --no-deps builder release itemknn --data-dir /data
# /data смонтирован из ARB_DATA_DIR: задайте ARB_BUNDLE_DIR=<ARB_DATA_DIR>/bundles/sha256-... по пути, напечатанному последней командой.
compose run --rm --no-deps --entrypoint /usr/local/bin/check_bundle prepare /artifacts
compose up -d bot
compose ps
```

`release itemknn` берёт проверенный GloVe-пакет текущего выпуска, закреплённые оценки датасета и снимок связей AniList, строит для каждого тайтла пять соседей ItemKNN (все пользователи, завершённые тайтлы с оценкой ≥ 8, сжатие 10) и подставляет соседей GloVe там, где ItemKNN их не находит. Каждая франшиза занимает одну карточку и показывается своим первым сезоном — самым смотримым TV-сезоном без более раннего TV-приквела (считаются все записи в списках, кроме «запланировано»), а без сериала — самым ранним фильмом, ONA или OVA; франшиза самого тайтла не показывается. Кроссоверы франшизы не склеивают. Кроме датасета в `ARB_DATA_DIR` нужен `raw/anilist-relations-2026-v1` из DVC; снимок получен `scripts/fetch_anilist_relations.py`. `catalog.json` совпадает с GloVe-пакетом побайтно; повторная сборка обязана дать ту же identity. Выбор алгоритма обоснован [отчётом ML-02](recsys/reports/ml02-validation-v1/README.md).

Один токен обслуживает один poller. `prepare` проверяет выбранный bundle и применяет миграции до запуска бота; PostgreSQL хранит историю в постоянном томе. Обычный перезапуск — `compose restart bot`. Для обновления сначала сохраните прежний bundle и выполните:

```bash
compose stop bot
# Атомарно замените ARB_BUNDLE_DIR в SETTINGS, сохранив владельца и права.
compose up -d --no-deps --force-recreate prepare
compose ps -a prepare
compose logs --since 5m prepare
```

Дождитесь завершения `prepare` с кодом 0 и проверьте ожидаемую `sha256:` identity в его логах. Только затем:

```bash
compose up -d --no-deps --force-recreate bot
compose ps bot
compose logs --since 5m bot
```

Проверьте ту же identity в новых логах бота. Откат повторяет последовательность с прежним `ARB_BUNDLE_DIR`. Сохраняйте старый bundle и PostgreSQL volume; `down -v` удаляет историю.

Миграции только добавляют таблицы и ограничения, но бот запускается лишь на схеме, версии которой знает. Миграция 2 (ARB-069) добавляет список тайтлов запроса, план выдачи и отметки «Уже смотрел»; выдача до неё становится своим планом. Бот прежней версии на схеме 2 не стартует. Поэтому перед обновлением образа снимите `pg_dump`; откат образа — это восстановление этой копии вместе с прежним образом. Подробная процедура и отдельная живая приёмка находятся в справочнике эксплуатации.
