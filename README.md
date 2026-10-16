# AnimeRecommendationBot

Telegram-бот для личных чатов: поиск аниме по названию и алиасам → подтверждение записи с MAL ID → до пяти похожих произведений → оценки и отзывы. Интерфейс русский. Telegram обрабатывает Rust; Python заранее строит рекомендации из закреплённого каталога Neelagiri 2025, алиасов Nafiul 2026 и GloVe 6B 300d. Исторический MyAnimeList 2020 остаётся выбираемым отдельно. Во время диалога бот читает готовый неизменяемый пакет JSON, а PostgreSQL хранит запросы, фактические выдачи, оценки и отзывы. Персонализации в v1 нет.

Названия и алиасы берутся из каталога, главным образом на английском и японском. Русская транслитерация автоматически не добавляется. Результаты ранжируются по сходству текста описаний, названий и жанров; исходный MAL ID и повторы исключены. В карточке неизвестные метаданные показываются как отсутствующие.

## Куда идти дальше

- **Запустить бота:** [первый запуск через Docker Compose](#первый-запуск-через-docker-compose), затем [руководство по развёртыванию](deploy/README.md). Обновление и откат опубликованного пакета описаны [там же](deploy/README.md#immutable-bundle-update-and-rollback).
- **Подготовить текущий выпуск:** [закреплённый выпуск и выбор `current`](recsys/RELEASE.md), [источники и сборка Python-пакета](recsys/README.md). Исторический MAL 2020 выбирается отдельно.
- **Разрабатывать и проверять:** [руководство разработчика](docs/development.md), [единое окружение](dev/environment.md), [архитектурная карта](docs/architecture.md) и [границы Python-модулей](docs/python-boundaries.md).
- **Версионировать входные данные:** [локальный DVC remote и восстановление снимков](docs/data-versioning.md). Пять выбранных входов зарегистрированы; байты восстанавливаются отдельно в каждой рабочей копии.
- **Проводить исследование:** [протокол, модели и кампания](recsys/RESEARCH.md). Для обычной проверки используйте маленькие фикстуры; полные входные файлы и результаты остаются приватными.
- **Проверить выпуск и вести изменение:** [стандартная приёмка первого уровня](deploy/LEVEL1_ACCEPTANCE.md), [частная процедура полного и живого испытания](deploy/ACCEPTANCE.md), [правила задач и агентов](AGENTS.md), [процесс pull request](.github/README.md). [Основной Python-код](src/README.md) находится в `src/recsys/`; работающий бот находится в `bot/`. История удалённого Python-бота доступна через [миграцию layout](docs/migrations/arb053-python-src-v1.md).

## Требования

- Docker Engine и Docker Compose **2.30.0+**. Образы закреплены: Rust 1.95.0, Python 3.12.14, PostgreSQL 16.15, uv 0.11.12 и Debian Bookworm. Первая сборка скачает инструменты и соберёт Rust.
- Для получения источников нужен доступ к MyAnimeList-репозиторию и Stanford GloVe. Архив GloVe около 862 МБ, распакованный файл около 1,04 ГБ; выделите несколько гигабайт для исходников, рабочих матриц, двух пакетов и образов.
- Отдельный токен Telegram-бота от BotFather. Токен хранится только в локальном игнорируемом файле; приложение не загружает корневой `.env` автоматически.
- Один активный polling-процесс на один токен. Используйте личный чат с ботом.

## Единое окружение для локальной разработки

Корневые `pyproject.toml`, `uv.lock`, `Cargo.toml` и `Cargo.lock` объединяют Python-подготовку и две Rust-команды. Нужны Python 3.12, Rust 1.95, C-компилятор, `pkg-config`, заголовки OpenSSL и uv 0.11.12. Из корня репозитория:

```bash
uv sync --locked
uv run --locked recsys --help
uv run --locked check_bundle tests/fixtures/bundle
cargo test --locked
```

Форматирование и базовая проверка Python выполняются закреплённым Ruff 0.17.1 в отдельном окружении. Из корня репозитория:

```bash
UV_PROJECT_ENVIRONMENT=.tools/lint-venv uv run --locked --only-group lint ruff check .
UV_PROJECT_ENVIRONMENT=.tools/lint-venv uv run --locked --only-group lint ruff format --check .
```

Правила: Python 3.12, 100 колонок, четыре пробела, двойные кавычки, LF, базовые E4/E7/E9/F, порядок импортов I, аннотации аргументов ANN001–ANN003 и строковые типы в `typing.cast` (TC006). Проверка охватывает Python-код и локальные `.pyi`; [запись о миграции layout](docs/migrations/arb054-recsys-layout-v2.json) фиксирует физические пути и исторические producer-хеши. Для проверки фактических типов вызовов установите отдельную CPU-среду без приложения и запустите строгий mypy:

```bash
UV_PROJECT_ENVIRONMENT=.tools/typing-venv uv sync --locked --no-install-project --no-default-groups --group typing --extra quality
UV_PROJECT_ENVIRONMENT=.tools/typing-venv uv run --locked --no-sync python -m mypy
```

Группа `typing` содержит только проверяющий mypy и его научные стабы; обычная установка и GPU-среда её не включают. Результаты проверок и `.tools/typing-venv` остаются локальными.

Для сравнения качества добавьте `uv sync --locked --extra quality` и передавайте `--extra quality` в последующих `uv run`. Бот читает переменные окружения напрямую; для локального запуска подготовьте `TELOXIDE_TOKEN`, `DATABASE_URL` и `ARTIFACTS_DIR` в своём `.env` и явно передайте его: `uv run --locked --env-file .env bot --check-config`, затем `uv run --locked --env-file .env bot`. Проверка конфигурации не обращается к базе или Telegram. Подробности и поведение редактируемой установки — в [руководстве по единому окружению](dev/environment.md). Для полной сборки пакета и запуска без локального Rust/Python окружения используйте Docker ниже.

Исходные лицензии и версии записаны в [реестре источников](src/recsys/resources/registries/source_registry.json) и [руководстве Python-пакета](recsys/README.md): MAL 2020 закреплён коммитом `9a1d7f5`, GloVe — SHA-256 архива и текста. Контрольные суммы обнаруживают повреждение и подмену входных байтов, но сами по себе не подтверждают происхождение источника.

## Первый запуск через Docker Compose

Клонируйте репозиторий, перейдите в его корень и задайте устойчивый локальный каталог. Ниже пример; пути должны быть абсолютными. Файлы внутри `.arb/` игнорируются Git и остаются при обновлении исходников.

```bash
git clone https://github.com/Sarvallerd/AnimeRecommendationBot.git
cd AnimeRecommendationBot
ROOT="$PWD"
PROJECT=animebot-local
LOCAL="$ROOT/.arb/local"
SETTINGS="$LOCAL/compose.env"
COMPOSE_FILE="$ROOT/compose.yaml"
mkdir -p "$LOCAL/data" "$LOCAL/secrets"
chmod 700 "$LOCAL/secrets"
id -u
id -g
```

Создайте `secrets/postgres.env` с буквальным `POSTGRES_PASSWORD=...` и `secrets/bot.env` с `TELOXIDE_TOKEN=...` и `DATABASE_URL=postgresql://anime_bot:<URL-encoded-password>@postgres:5432/anime_bot?sslmode=disable`. Образцы есть в [`deploy/`](deploy/README.md). Зарезервированные символы пароля в URL кодируются: `$` как `%24`, `@` как `%40`, `/` как `%2F`. В raw env-файлах Compose не нужны shell-кавычки. Поставьте обоим файлам режим `0600`. Существующий `.env` не перезаписывайте.

Создайте `compose.env` по [`deploy/compose.env.example`](deploy/compose.env.example) и укажите:

```text
ARB_DATA_DIR=/absolute/path/to/.arb/local/data
ARB_BUNDLE_DIR=/absolute/path/to/.arb/local/data/bundles/sha256-...
ARB_UID=<вывод id -u>
ARB_GID=<вывод id -g>
ARB_LOG=info
ARB_BOT_ENV=/absolute/path/to/.arb/local/secrets/bot.env
ARB_POSTGRES_ENV=/absolute/path/to/.arb/local/secrets/postgres.env
```

До первого экспорта `ARB_BUNDLE_DIR` может указывать на будущий путь; бот запускается только после выбора созданного пакета.

**Перед первой подготовкой нового `current` восстановите принятый пакет v1 из исходного производителя.** Версия 0.2 проверяет и повторно использует его артефакты, но не создаёт выпуск с прежними хешами версии 0.1. Сначала разместите проверенные Neelagiri base и Nafiul aliases, исходный кэш Nafiul и текст GloVe в одном приватном каталоге. В командах замените `/private/DATA` на абсолютный `ARB_DATA_DIR` из `compose.env` (`$LOCAL/data` в примере выше) согласно [контракту данных и выпуска в Notion](https://app.notion.com/p/3fb3908d6b1f8136a1dfff3c4a12108e). Из корня чистой рабочей копии создайте отдельный checkout исходного коммита и проверьте его SHA:

```bash
git worktree add --detach /private/arb-reference-v1 1441573fbbbffc4693c7e1100aaae4458d9e88d4
cd /private/arb-reference-v1
test "$(git rev-parse HEAD)" = 1441573fbbbffc4693c7e1100aaae4458d9e88d4
test -z "$(git status --porcelain)"
uv sync --python 3.12.14 --locked --no-editable --extra quality
test "$(uv run --python 3.12.14 --locked --no-editable --extra quality recsys --version)" = "recsys 0.1.0"
uv run --python 3.12.14 --locked --no-editable --extra quality recsys release prepare --release current --data-dir /private/DATA
cd "$ROOT"
```

Если Python 3.12.14 ещё не установлен для `uv`, сначала выполните `uv python install 3.12.14`. Используйте именно проверенные входные байты и исходный lock; не переписывайте manifest и producer-хеши. После подготовки вернитесь в checkout версии 0.2, укажите тот же каталог данных и проверьте сохранённые `catalog release` и `bundle` перед выбором пакета для бота. При отсутствии старого производителя текущая команда выдаёт `release_reference_producer_required` до записи. Это восстановление не является новой приёмкой полного каталога; полную приёмку v1 запускают из закреплённого checkout, а проверка старых свидетельств в текущем дереве остаётся доступной.

Создайте `data/raw` от имени `ARB_UID` и выполните:

```bash
compose() { docker compose --env-file "$SETTINGS" -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }
compose --profile tools build builder bot
compose --profile tools run --rm --no-deps builder obtain --snapshot nafiul-2026-v1 --data-dir /data/raw/nafiul-2026-v1
compose --profile tools run --rm --no-deps --entrypoint /usr/local/bin/prepare-artifacts builder
```

Последняя команда запускает `recsys release prepare --release current --data-dir /data`. До запуска разместите проверенные immutable base и aliases в `/data/datasets/sha256-...` и `/data/aliases/sha256-...`, кэш Nafiul в `/data/raw/nafiul-2026-v1`, текст GloVe в `/data/raw/glove-6b-300d-v1`. Команда работает offline, проверяет lock и печатает путь проверенного пакета и подсказку `ARB_BUNDLE_DIR`. Выбор работающего пакета в `compose.env` остаётся отдельным шагом.

```bash
compose --profile tools run --rm --no-deps builder validate /data/bundles/sha256-<digest>
compose run --rm --no-deps --entrypoint /usr/local/bin/check_bundle prepare /artifacts
compose up -d bot
compose ps
```

`prepare` ждёт готовности PostgreSQL, проверяет весь пакет и применяет миграцию. Только после успешного `prepare` стартует polling-бот. `bot --prepare` не обращается к Telegram; `bot --check-config` проверяет лишь конфигурацию и существование каталога. При прямом запуске из исходников нужен явный `cargo run --locked --bin bot -- ...`, Rust 1.95, C-компилятор, `pkg-config`, OpenSSL headers и переменные `TELOXIDE_TOKEN`, `DATABASE_URL`, `ARTIFACTS_DIR`. База в Compose использует собственный том; внешний PostgreSQL для обычного запуска не нужен. Подробности: [Rust](bot/docs/build.md), [Compose](deploy/README.md).

## Диалог

Команды `/start`, `/help`, `/cancel`, `/recommend`, `/rate`, `/feedback` работают во всех состояниях. `/recommend` и `/rate` сначала спрашивают название и показывают варианты с MAL ID; даже единственное совпадение нужно подтвердить кнопкой. Поиск учитывает регистр, алиасы и ограниченные опечатки. `/recommend` отправляет до пяти карточек с кратким описанием (до трёх предложений): полный текст открывается кнопкой и листается в той же карточке. При доступной обложке карточка приходит одним фото с подписью; без неё — одним текстовым сообщением. Обложку бот ищет сначала через Jikan, затем через AniList по тому же MAL ID. Это управляется `COVERS_ENABLED=true|false` (по умолчанию `true`). Затем можно оценить полезность конкретной рекомендации от **0 до 5**. `/rate` записывает оценку аниме от **1 до 10**; новый запрос может обновить текущую оценку, сохранив прежние события. `/feedback` сохраняет один текст целиком, включая Unicode и переносы строк.

После рестарта кнопки незавершённого диалога устаревают, а уже записанные строки остаются. При отказе базы бот предлагает явный повтор исходного действия; не считайте сообщение Telegram и запись PostgreSQL одной атомарной операцией. При сбое между ними карточка иногда может появиться повторно. [Сценарии](bot/docs/dialogue.md), [карточки](bot/docs/recommendations.md), [оценки](bot/docs/ratings.md), [отзывы](bot/docs/feedback.md).

## Обновление и проверки

Перезапуск: `compose restart bot`. Для обновления пакета сначала остановите бота, выберите новый **неизменяемый** каталог в `ARB_BUNDLE_DIR`, заново создайте `prepare` и потребуйте выход `0`, затем заново создайте бота. Для отката верните старый каталог тем же порядком. Не используйте `down -v` с рабочей базой: команда удаляет том с историей. Точные команды и проверка загруженной версии — в [руководстве Compose](deploy/README.md).

Быстрые проверки без реального токена и полного датасета:

```bash
python3 scripts/check_agent_setup.py
python3 contracts/check_bundle.py tests/fixtures/bundle
python3 -m unittest discover -s contracts -p 'test_*.py'
python3 -m unittest discover -s deploy/tests -p 'test_acceptance_evidence.py'
python3 deploy/tests/level1-acceptance.py
```

Последняя команда — стандартная [автоматическая приёмка первого уровня](deploy/LEVEL1_ACCEPTANCE.md): отдельная тестовая PostgreSQL, имитация Telegram и синтетическое переключение пакетов A→B→A.

Офлайн-подготовка `current` на полных закреплённых данных прошла: она воспроизвела [принятый пакет ARB-031](recsys/RELEASE.md) из 28 955 записей, а повторный запуск вернул те же байты. Синтетическая проверка Compose также прошла на [проверенном коммите](https://github.com/Sarvallerd/AnimeRecommendationBot/actions/runs/38158265951).

Отдельная добровольная приёмка из [runbook](deploy/ACCEPTANCE.md) собирает настоящий каталог в двух режимах и проверяет живой личный диалог, отказ базы, рестарт, обновление и откат. До наблюдения всех шагов её итог остаётся `PENDING`.

[Отчёт о качестве ARB-018](recsys/reports/arb018-v1/README.md) сравнивает выбранный GloVe-алгоритм с жанровым и исправленным исходным вариантом на фиксированных 20 запросах. Оценку 268 пар сделал один LLM-когорт; часть оценок осталась неизвестной, исторические методы использовали разные каталоги, а время измерялось для подготовки и ранжирования 20 запросов, не для полной сборки всех соседей. Эти результаты не доказывают общее превосходство метода.

Исходники: [`bot/`](bot/docs/build.md) — активный Rust-бот; [`src/recsys/`](src/README.md) — исходники Python-подготовки; [`recsys/`](recsys/README.md) — её руководства, тесты и исторические отчёты; [`contracts/`](contracts/README.md) — формат пакета; [`deploy/`](deploy/README.md) — запуск; [`dev/notebooks/`](dev/notebooks/README.md) — исторические исследовательские ноутбуки.
