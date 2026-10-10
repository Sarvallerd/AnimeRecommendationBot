# AnimeRecommendationBot

Telegram-бот для личных чатов: поиск аниме по названию и алиасам → подтверждение записи с MAL ID → до пяти похожих произведений → оценки и отзывы. Интерфейс русский. Telegram обрабатывает Rust; Python заранее строит рекомендации из MyAnimeList 2020 и GloVe 6B 300d. Во время диалога бот читает готовый неизменяемый пакет JSON, а PostgreSQL хранит запросы, фактические выдачи, оценки и отзывы. Персонализации в v1 нет.

Названия и алиасы берутся из каталога, главным образом на английском и японском. Русская транслитерация автоматически не добавляется. Результаты ранжируются по сходству текста описаний, названий и жанров; исходный MAL ID и повторы исключены. В карточке неизвестные метаданные показываются как отсутствующие.

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

Для сравнения качества добавьте `uv sync --locked --extra quality` и передавайте `--extra quality` в последующих `uv run`. Бот читает переменные окружения напрямую; для локального запуска подготовьте `TELOXIDE_TOKEN`, `DATABASE_URL` и `ARTIFACTS_DIR` в своём `.env` и явно передайте его: `uv run --locked --env-file .env bot --check-config`, затем `uv run --locked --env-file .env bot`. Проверка конфигурации не обращается к базе или Telegram. Подробности и поведение редактируемой установки — в [руководстве по единому окружению](dev/environment.md). Для полной сборки пакета и запуска без локального Rust/Python окружения используйте Docker ниже.

Исходные лицензии и версии записаны в [реестре источников](recsys/src/recsys/source_registry.json) и [руководстве Python-пакета](recsys/README.md): MAL 2020 закреплён коммитом `9a1d7f5`, GloVe — SHA-256 архива и текста. Контрольные суммы обнаруживают повреждение и подмену входных байтов, но сами по себе не подтверждают происхождение источника.

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

До первого экспорта `ARB_BUNDLE_DIR` может указывать на будущий путь; бот запускается только после выбора созданного пакета. Создайте `data/raw` от имени `ARB_UID` и выполните:

```bash
compose() { docker compose --env-file "$SETTINGS" -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }
compose --profile tools build builder bot
compose --profile tools run --rm --no-deps builder obtain --data-dir /data/raw
compose --profile tools run --rm --no-deps --entrypoint /usr/local/bin/prepare-artifacts builder
```

Последняя команда повторно проверяет источники без сети, нормализует каталог, строит top-5, экспортирует пакет и печатает его каталог `sha256-...`. Обновите только `ARB_BUNDLE_DIR` в `compose.env` на абсолютный путь к этому каталогу. Пакет содержит `catalog.json`, `neighbors.json`, `manifest.json`; рабочие отчёты остаются в `data/work`.

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
bash deploy/tests/compose-acceptance.sh
```

Полная приёмка из [runbook](deploy/ACCEPTANCE.md) отдельно собирает настоящий каталог в двух режимах, проверяет PostgreSQL и настоящий личный диалог, отказ базы, рестарт, обновление и откат. До наблюдения всех шагов её итог остаётся `PENDING`.

[Отчёт о качестве ARB-018](recsys/reports/arb018-v1/README.md) сравнивает выбранный GloVe-алгоритм с жанровым и исправленным исходным вариантом на фиксированных 20 запросах. Оценку 268 пар сделал один LLM-когорт; часть оценок осталась неизвестной, исторические методы использовали разные каталоги, а время измерялось для подготовки и ранжирования 20 запросов, не для полной сборки всех соседей. Эти результаты не доказывают общее превосходство метода.

Исходники: [`bot/`](bot/docs/build.md) — активный Rust-бот; [`recsys/`](recsys/README.md) — Python-подготовка; [`contracts/`](contracts/README.md) — формат пакета; [`deploy/`](deploy/README.md) — запуск; [`src/`](src/README.md) и [`dev/notebooks/`](dev/notebooks/README.md) — сохранённый исходный Python-код и исследовательские ноутбуки.
