# Разработка и проверки

Рабочая карта: [README](../README.md) ведёт к запуску; [архитектура](architecture.md) показывает границы; [единое окружение](../dev/environment.md) содержит команды установки. Локальный проект приложения `AnimeRecSys` уже связан с этим репозиторием. Отдельные задачи в проекте не получают автоматически общий текст переписки: устойчивые решения записывайте в Notion и в документах репозитория.

## Перед изменением

Возьмите конкретную задачу, её зависимости и критерии из Notion. Следуйте [правилам агентов](../AGENTS.md) и [процессу pull request](../.github/README.md): одна задача, ветка, worktree и PR; проверьте исходный SHA, чистоту рабочей копии и владельцев файлов. Раздельным задачам нужны отдельные базы, Compose-проекты и выходные каталоги. Не запускайте два polling-процесса на одном Telegram-токене. Результаты и приватные свидетельства храните вне Git.

Обычное окружение собирается в каждом worktree отдельно по [руководству](../dev/environment.md). Для быстрой проверки установки без реальных данных используйте `uv run --locked recsys --help` и `uv run --locked check_bundle tests/fixtures/bundle` из корня после `uv sync --locked`. Для исследования с зафиксированными версиями используйте установленный `--no-editable --extra quality` пакет, как описано в [протоколе](../recsys/RESEARCH.md); опциональная GPU-ветка требует отдельного `--extra gpu`, поддерживаемого варианта и явного `--backend gpu`. [Группа DVC `data`](data-versioning.md) запускается в отдельном окружении и не подменяет Python/Rust окружение приложения.

Для воспроизводимого эксперимента фиксируйте Git SHA, принятые идентичности входов и DVC-указателей, параметры и backend, зафиксированное окружение, отдельный выходной каталог и пределы ресурсов. Сохраняйте исходные пакеты и полные приватные выходы; публикуйте только разрешённые агрегаты. Выбор нового runtime-пакета требует явной процедуры выпуска и развёртывания, даже если исследовательский результат получен успешно.

## Как выбрать проверку

| Что меняется | Где смотреть и что запускать |
| --- | --- |
| Конфигурация ролей и правила задач | `python3 scripts/check_agent_setup.py`; [workflow](../.github/README.md) |
| Python-подготовка, выпуск и исследования | Установленный пакет и профильные `recsys/tests/` по [окружению](../dev/environment.md); входы и команды выпуска в [release](../recsys/RELEASE.md), исследовательский контракт в [research](../recsys/RESEARCH.md) |
| Локальный DVC и метаданные снимков | Фикстурная проверка `scripts/check_dvc.py` в отдельном окружении группы `data`: [команда и проверка восстановления](data-versioning.md#настройка) |
| Rust runtime и база | `cargo test --locked`, миграционные и профильные тесты; [сборка](../bot/docs/build.md), [база](../bot/docs/database.md) |
| Общий JSON-контракт | `python3 contracts/check_bundle.py tests/fixtures/bundle`, `python3 -m unittest discover -s contracts -p 'test_*.py'`; [спецификация](../contracts/README.md) |
| Общее окружение и контейнеры | `python3 dev/tests/check_environment.py`; [окружение](../dev/environment.md), [Compose](../deploy/README.md) |
| Стандартная приёмка | `python3 deploy/tests/level1-acceptance.py` с FakeTelegram, временной PostgreSQL и малым A→B→A; [ограничения и процедура](../deploy/LEVEL1_ACCEPTANCE.md) |
| Полный каталог и живой диалог | Отдельная частная [процедура приёмки](../deploy/ACCEPTANCE.md); отсутствие наблюдаемого шага остаётся `PENDING` |

Перед слиянием по процессу проекта должны пройти пять CI jobs: `agent-setup`, `python`, `rust` ([основной workflow](../.github/workflows/bootstrap.yml)), `compose` ([Compose workflow](../.github/workflows/compose.yml)) и `dev-environment` ([проверка окружения](../.github/workflows/dev-environment.yml)). DVC-фикстура выполняется внутри `python`.

## Канонические маршруты

- Текущая подготовка: [набор данных](../recsys/DATASET.md) → [алиасы](../recsys/ALIASES.md) → [именованный выпуск](../recsys/RELEASE.md) → [экспорт](../recsys/EXPORT.md).
- Исторический MAL 2020: явный selector в [release](../recsys/RELEASE.md), исходный контекст в [Python-пакете](../recsys/README.md); [архив](../src/README.md) не является рабочим runtime.
- Исследования и эксперименты: [протокол и кампания](../recsys/RESEARCH.md), [локальный DVC remote](data-versioning.md). Текущий статус отдельных научных запусков хранится в задачах Notion.
- Развёртывание: [первый запуск](../README.md#первый-запуск-через-docker-compose), [обновление и откат](../deploy/README.md#immutable-bundle-update-and-rollback), [приёмка](../deploy/LEVEL1_ACCEPTANCE.md).
