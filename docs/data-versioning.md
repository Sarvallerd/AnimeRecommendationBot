# Версионирование данных

Репозиторий хранит код, спецификации и небольшие DVC-указатели. Большие входные
файлы лежат в приватном локальном DVC remote. Настройка адресов находится только
в `.dvc/config.local`; Git её игнорирует. По умолчанию cache и remote создаются
как отдельные каталоги в
`${XDG_DATA_HOME:-$HOME/.local/share}/anime-recommendation-bot/dvc/`. Это
хранилище переживает удаление рабочего Git worktree. Локальный remote на том же
диске **не защищает от потери диска**.

## Настройка

Из корня репозитория, с установленным `uv` версии 0.11.12:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data python scripts/dvc_local.py configure
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data python scripts/dvc_local.py check
```

Для другого расположения передайте `--storage-root /absolute/private/path` обеим
командам. Каталог должен находиться вне всех Git worktree и общего каталога Git,
не содержать символьных ссылок и принадлежать текущему пользователю с правами
`0700`. Скрипт создаёт приватные cache и remote, а также `.dvc/config.local` с
правами `0600`. Повторная настройка с теми же значениями безопасна; при
расхождении конфигурации она остановится и попросит исправить конфликт вручную.
`check` только проверяет существующую настройку. Настройки global DVC и Git не
меняются.

Отдельную проверку полного цикла на фиктивных файлах запускайте так:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data python scripts/check_dvc.py
```

Проверка создаёт временный репозиторий, отдельный источник и локальное хранилище,
проходит A→B→A, перенос в свежий clone с пустым cache и обнаружение повреждения.
Она не читает реальные данные. Группа `data` не входит в обычную установку
бота, исследовательского CPU или GPU окружения.

## Снимки и восстановление

Зарегистрированы ровно пять входных наборов неизменяемого
[`release_neelagiri_2025_nafiul_2026_v1.json`](../src/recsys/release_neelagiri_2025_nafiul_2026_v1.json).
Git хранит следующие указатели, а сами файлы передаются через локальный DVC
remote. Имена SHA-256 каталогов совпадают с идентификаторами исходных
манифестов в release lock.

| Набор | DVC-указатель | Доверенный источник | Файлов |
| --- | --- | --- | ---: |
| Подготовленная база Neelagiri 2025 | [base](../data/snapshots/datasets/sha256-78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6.dvc) | `inputs.base.identity` в release lock | 5 |
| Алиасы Nafiul 2026 | [aliases](../data/snapshots/aliases/sha256-ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403.dvc) | `inputs.aliases.identity` в release lock | 3 |
| Каталог release | [release](../data/snapshots/releases/sha256-8da879636b476ffc9d72bf3a08447b778ef047232fc02cd69b1ed0ca02168b5c.dvc) | `inputs.catalog_release.identity` в release lock | 2 |
| Исходный Nafiul 2026 | [nafiul](../data/snapshots/raw/nafiul-2026-v1.dvc) | [реестр Nafiul](../src/recsys/source_registry_nafiul_2026_v1.json) и включённый `source-manifest.json` | 2 |
| Текст GloVe 6B 300d | [glove](../data/snapshots/raw/glove-6b-300d-v1/glove.6B.300d.txt.dvc) | `inputs.embeddings` в release lock и [реестр GloVe](../src/recsys/source_registry_glove_6b_300d_v1.json) | 1 |

Подготовленная база содержит псевдонимизированные взаимодействия; доступ к
remote ограничивают как к приватным данным. GloVe зарегистрирован как один
извлечённый текстовый файл, без исходного ZIP-архива. Исходные пакеты сохраняются
на месте и проверяются отдельно по SHA-256. Для восстановления нужна рабочая
копия Git с этими указателями и доступ к тому же локальному remote.

В новой рабочей копии создайте **пустой отдельный cache** вне Git. Укажите его
абсолютный путь и абсолютный путь к существующему приватному remote из первичной
настройки. Эти адреса сохраняются только в `.dvc/config.local` новой копии:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data dvc config --local cache.dir /absolute/private/empty-cache
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data dvc config --local cache.type copy
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data dvc remote add --local --default local /absolute/private/existing-remote
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data dvc remote modify --local local jobs 2
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data dvc remote modify --local local verify true
```

Убедитесь, что cache пуст, затем восстановите нужные указатели. Для всех пяти:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data dvc pull -r local \
  data/snapshots/datasets/sha256-78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6.dvc \
  data/snapshots/aliases/sha256-ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403.dvc \
  data/snapshots/releases/sha256-8da879636b476ffc9d72bf3a08447b778ef047232fc02cd69b1ed0ca02168b5c.dvc \
  data/snapshots/raw/nafiul-2026-v1.dvc \
  data/snapshots/raw/glove-6b-300d-v1/glove.6B.300d.txt.dvc
```

Для публикации из копии с уже проверенными файлами используйте:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/arb-data-venv uv run --locked --only-group data dvc push -r local \
  data/snapshots/datasets/sha256-78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6.dvc \
  data/snapshots/aliases/sha256-ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403.dvc \
  data/snapshots/releases/sha256-8da879636b476ffc9d72bf3a08447b778ef047232fc02cd69b1ed0ca02168b5c.dvc \
  data/snapshots/raw/nafiul-2026-v1.dvc \
  data/snapshots/raw/glove-6b-300d-v1/glove.6B.300d.txt.dvc
```

После `pull` отдельно сверяйте SHA-256 и
размер каждого файла с доверенными манифестами и реестрами: DVC хеши отвечают
за перенос, а принятые идентификаторы и release lock — за научную
воспроизводимость. Изменяемый DVC checkout нельзя использовать как
`ARB_BUNDLE_DIR`: публикация runtime bundle остаётся отдельной неизменяемой
операцией. `dvc repro` не используется для подготовки этого bundle.

Личные acceptance-логи, скрытые метки, истории пользователей, профили, токены,
базы PostgreSQL, предсказания и оценки не включаются в DVC snapshots. Содержимое
локального remote также приватно; его нельзя публиковать как репозиторий.
