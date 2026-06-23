# CLAUDE.md

Указания для Claude Code (claude.ai/code) при работе с этим репозиторием.

**Язык:** общайся и думай по-русски (ответы, рассуждения, комментарии, докстринги).

## Что это

Бот на имитационном обучении (SFT / behavioral cloning) для Kaggle-соревнования **Orbit Wars**.
Set-transformer policy/value-сеть обучается на публичных реплеях лидерборда, имитируя сильных игроков,
и затем сабмитится как агент `act(obs) -> moves`. Механики игры, которые кодируют фичи и декодер, см.
в `orbit_wars_rules.md` и `insights/*.md`.

Комментарии и докстринги в коде — на **русском** и **краткие**; держись этого при правках.

## Граница сабмишна (важно)

Две половины репо, которые нельзя смешивать:

- **Runtime / сабмишн**: `model.py` + `core/` (только numpy + torch). Именно это крутится внутри
  Kaggle-окружения на инференсе. `core/__init__.py` — публичная поверхность; `model.PolicyValueNet`
  лежит на верхнем уровне и импортирует `core`. Там же, на верхнем уровне `model.py`, живёт
  `bucket_to_ships` — **единственный источник правды** для floor-декода доли гарнизона в число
  кораблей (его же импортируют разметка датасета и ETL, чтобы метка и декод не разъехались).
  `core/geo_lite.py` — тонкий фасад над пакетом `orbit_lite` (`producer-orbit-wars-utils/`,
  torch-only) и **единственный** его импортёр *внутри сабмишна* — поэтому `orbit_lite/` надо
  бандлить в сабмишн рядом с `model.py` (фасад сам добавляет родительскую директорию в `sys.path`).
  (Офлайновые `agents/` тоже импортируют `orbit_lite` напрямую, но в сабмишн не входят.) Фасад даёт
  геометрию запуска (lead-угол / угол→планета / валидация запуска); легаси-инструменты на numpy
  (`core/intercept.py`, `core/utils.validate_launch`) остаются, потому что на них всё ещё опираются
  `core/features.py` и `dataprep`.
- **Только офлайн**: `sft/`, `dataprep/`, `configs/`, `eval/`, `agents/`, `hydra_utils.py`,
  `notebooks/`. Тянут `hydra`, `torch.distributed`, `kaggle`, `kaggle_environments`, `trueskill`,
  датасеты, matplotlib — нужны для обучения/ETL/оценки, в сабмишн не попадают. `agents/` —
  отдельные эвристические боты (`orbit_lite` flow-diff планировщики, перенесённые из Kaggle-ноутбуков),
  используются как оппоненты на эвале.

## Окружение и команды

Зависимости перечислены в `requirements.txt` (версии сняты с рабочего `.venv`, Python 3.12: torch,
hydra-core, omegaconf, numpy, tensorboard, kaggle, tqdm, kaggle-environments, trueskill, matplotlib).
Сам `.venv` в git **не коммитится** (в `.gitignore`, как и `data/`, `replays/`, `*.jsonl`, `*.pt`,
`outputs/`, `tb/`). Репозиторий **является** git-репо. Всегда запускай интерпретатором из venv:
`.venv/bin/python`.

```bash
# Smoke-тесты архитектуры (kaggle_environments не нужен): формы/маски forward, set-инвариантность,
# корректность intercept против симулированного оракула, валидность act(), validate_launch vs brute-force
.venv/bin/python smoke_test.py

# Корректность SFT-пайплайна (round-trip меток цели и бакета доли, инвариант маски, обучаемость на одном батче)
.venv/bin/python -m sft.check --path data/sft.full_send.jsonl     # либо маленький /tmp/sft.smoke.jsonl

# Обучение (Hydra). Девайс (CUDA / MPS / CPU) выбирается автоматически.
.venv/bin/python -m sft.train                                     # один процесс
.venv/bin/torchrun --standalone --nproc_per_node=N sft/train.py   # мульти-GPU DDP
.venv/bin/python -m sft.train train.batch_size=512 data.w_hold=0.05   # CLI-оверрайды (Hydra dot-path)

# Локальный турнир (Hydra; см. раздел «Локальная оценка»)
.venv/bin/python -m eval.run pool=baselines mode=1v1 episodes=25

tensorboard --logdir outputs                                      # запуски пишутся под outputs/
```

Запуски ложатся в `outputs/<timestamp>/` (cwd Hydra) с `checkpoints/` (best/last/epochNN `.pt`) и
`tb/` внутри. Чекпойнты несут в себе `model_cfg` + `feature_cfg`, поэтому веса грузятся в
`PolicyValueNet.act` без угадывания форм. Все Hydra-входы (`sft.train`, `eval.run`) печатают полный
конфиг в начале запуска через `hydra_utils.print_cfg`.

Pytest-набора нет — тестовая обвязка это `smoke_test.py` и `sft.check`; гоняй их после правок
`core/`, `model.py` или кода датасета/лоссов.

## Архитектура модели (`model.py` + `core/features.py`)

**Set-transformer над разнородными токенами-сущностями**, декодируемый как задача «ребро источник→цель»:

1. `core.features.encode(obs)` превращает сырой obs-dict в паддированные фич-тензоры по типам (планеты,
   кометы, флоты) плюс единичный токен-солнце и глобальный вектор «доп-фич side». Возвращает также
   `places` (метаданные декода, включая готовый `intercept.Target`) и `owned_idx`.
2. Per-type MLP-энкодеры проецируют каждую сущность в `d_model` и добавляют обучаемый type-эмбеддинг
   **плюс относительный owner-эмбеддинг** (`model.owner_emb`, таблица `[5, d_model]`): к каждому
   токену-планете/комете/флоту прибавляется эмбеддинг по слоту владельца **относительно нас** —
   `0=мы, 1=CCW-сосед, 2=напротив, 3=CW-сосед, 4=нейтрал` (солнце/глобал — без owner). Слот считается
   в `features.encode` чисто по `owner id` через перестановку `_OWNER_POS={0:0,1:1,2:3,3:2}` (выведена
   из движка: спавн ротационно-симметричен, `id0↔id3`/`id1↔id2` напротив; в 1v1 враг всегда слот 2).
   Это даёт направленную идентификацию оппонентов в 4p; в 1v1 эквивалентно старому `is_enemy`.
   Всё конкатится в единый набор токенов, прогоняется через `TransformerEncoder` (перестановочно
   инвариантен; паддинг маскируется через `src_key_padding_mask`).
3. Хиддены планет+комет — это «места». `mlp_from`/`mlp_to` дают матрицу скоров `S[from, to]` (scaled
   dot-product), к ней добавляется колонка `hold` → логиты `[B, M, M+1]`, softmax по оси `to`.
   Self-target по диагонали и паддинг-колонки маскируются в `-inf`. Value-голова читает глобальный токен.
4. Декод `act()`: каждое своё место argmax-ом выбирает одну цель (или hold). **Число кораблей**
   выбирает голова `mlp_frac` — **4-классовый бакет-классификатор доли гарнизона {25, 50, 75, 100}%**,
   обусловленный **парой (источник, цель)**: на вход идёт конкат хидденов `h_src ⊕ h_tgt` (то есть
   факторизация `p(куда) · p(сколько | куда)`). Бакет переводится в целое число кораблей функцией
   `bucket_to_ships` (floor-округление доли: не пере-засылаем, ошибка < 1 корабля; 100% → весь
   гарнизон). **Угол** запуска берётся из `core.geo_lite.GeoEngine.intercept` (lead-угол orbit_lite),
   чтобы движущиеся цели брались с упреждением.

**Голова числа кораблей и DDP**: при teacher forcing (обучение) `forward(frac_pairs=(b_idx, s_idx,
t_idx))` считает `mlp_frac` **внутри** обёрнутого forward и кладёт результат в `frac_logits` [N, 4].
Это принципиально для DDP — иначе градиенты `mlp_frac` не синхронизируются между рангами. `t_idx` —
**экспертная** цель из меток (см. `sft.loss.frac_pairs_from`).

**Контракт размерностей фич**: `PLANET_FEAT_DIM=20`, `COMET_FEAT_DIM=25`, `FLEET_FEAT_DIM=14`,
`GLOBAL_FEAT_DIM=11` в `core/features.py` зашиты в размеры входов энкодера в `model.py` (а в
`sft/dataset.collate` динамический паддинг тянет те же константы, не хардкод). `PLANET_FEAT_DIM`
предполагает `len(FeatureConfig.horizons)==3` — менять horizons ломает контракт; движок намеренно не
выставляет `horizons` обучаемым параметром (`sft/engine.py:_feature_cfg`).

**Фичи назначения флота** (последние 4 канала `FLEET_FEAT_DIM`): `dx, dy` — центр планеты-назначения
относительно флота (нормировка /50), `eta` (нормировка на `FLEET_ETA_NORM=150`), `has_target` (флаг).
Назначение в obs **не лежит** (`fleets`-строка — `[id, owner, x, y, angle, from_planet_id, ships]`,
куда летит не хранится), поэтому планета-цель и ETA считаются `core.geo_lite.GeoEngine.fleet_targets`
(first-contact orbit_lite из текущей позиции флота вдоль heading, верный движку). Флот ни в кого не
попадающий (край/солнце/мимо) → нули + `has_target=0` (не врём «цель в (0,0), eta=0»). `encode`
принимает опциональный `geo`: `model.act` строит `GeoEngine` один раз и переиспользует его и для фич,
и для угла декода (иначе `PlanetMovement` строился бы дважды за ход); в воркерах обучения `geo=None` →
строится лениво внутри `encode` (заметная цена: torch-`PlanetMovement` на каждый ход).

`core/intercept.py` — самодостаточный numpy-инструмент lead-угла (статичные / орбитальные / кометные
цели; логарифмический `fleet_speed` 1→6). `core/utils.py` — `build_mlp` и `validate_launch`
(векторная проверка прямолинейного полёта на столкновение с солнцем / другими планетами / выход за
поле). Они всё ещё подпирают фич-математику в `core/features.py`, поэтому остаются — но **боевая
геометрия запуска** (угол декода, снайпер эвала, угол→планета в ETL) теперь идёт через
`core/geo_lite.py` (`orbit_lite`).

**Шим `initial_planets` в `core/geo_lite.py` (важно)**: `orbit_lite` реконструирует фазу каждой
орбиты из `initial_planets`, считая их позициями *на шаге 0 игры* (`angle = a0 + angvel·(step-1)`,
вращение вокруг `(50,50)`). В реплеях этого репо `initial_planets == текущие planets`, из-за чего
прогноз orbit_lite перелетал бы каждую орбитальную цель на `angvel·(step-1)` (~40 ед. поля на больших
шагах → неверный intercept/резолв). Поэтому фасад восстанавливает истинный `a0`, **откручивая**
текущие планеты назад на `angvel·(step-1)` перед передачей obs в orbit_lite. Операция идемпотентна
(настоящий game-initial obs даёт тот же `a0`) и проверяется `smoke_test.test_geo_lite` (ошибка
1-шагового прогноза орбиты ≈ 0).

## Поток обучающих данных

`sft/dataset.py` читает `data/sft.full_send.jsonl` (дефолт; есть и вариант `data/sft.all_send.jsonl` —
см. ниже). Каждый ход кодируется **на лету** в воркерах DataLoader; таргеты строятся
layout-независимо (на источник: планета-назначение / `HOLD` / `IGNORE`) и переводятся в индексы мест
уже в `collate`, который паддит **динамически до максимума батча** по каждому типу сущности (модель
shape-driven, фиксированного паддинга 40/16/256 на обучении нет). Сплит train/val — **по эпизодам**
(`meta.episode_id`): соседние ходы коррелируют, поэлементный сплит протёк бы.

**Две головы — две метки на источник:**
- Метка цели (`labels`): индекс места-назначения, `HOLD` или ignore.
- Метка доли (`frac_labels`): бакет {0:25, 1:50, 2:75, 3:100}%, ставится **только** для разрешённых
  вылетов (нужен валидный индекс цели `t_idx` для гейзера `h_tgt`) **и только при ТОЧНОМ совпадении**:
  `ship_bucket` даёт бакет лишь когда `ships == bucket_to_ships(b, garrison)`, иначе `IGNORE` (вылет не
  даёт сигнала голове доли). Так лейбл всегда воспроизводим на инференсе floor-декодом — модель не учит
  бакет, который декодит в другое число (эксперт послал 4 из 7 → floor-декод 50% даёт 3 → такой вылет
  в IGNORE). Тай-брейк при схлопывании (малый гарнизон) — наибольший бакет. Опирается на ту же
  `bucket_to_ships`, что и декод, поэтому метка и декод не разъезжаются. **Цена:** ~29% частичных
  вылетов (и ~34% бакета 50%, т.к. эксперты округляют половину вверх, а декод — вниз) уходят в IGNORE;
  `full_send` (всё = 100%) не затронут. `frac_weights`-ориентир печатает `dataprep/preprocess.py`.

Чтобы у головы доли был сигнал, `sends`-строки несут **число кораблей**: `[from_id, dest_id, ships]`
(старые 2-колоночные файлы грузятся без метки доли). На `full_send` все вылеты — весь гарнизон, так что
обучающего сигнала по доле почти нет; для головы числа кораблей предназначен вариант **`all_send`**
(фильтр `partial_send`, оставляет частичные вылеты — `configs/data/all_send.yaml`).

`sft/loss.py`:
- `policy_loss` — взвешенный cross-entropy по источникам. Число классов **динамическое** (`M_b+1` на
  батч), стандартный per-class `weight=` бессмыслен — поэтому down-weight-ится **только** колонка
  `hold` (всегда последняя, `hold_idx = M_b`) на `w_hold` (≈0.074 для full_send — send/hold-баланс;
  ≈0.134 для all_send).
- `fraction_loss` — CE головы доли по **фиксированным** 4 классам, поэтому обычный `weight=`-вектор
  работает: `train.frac_weights` балансирует перекос к 100% (~67% вылетов — full). Ориентир (обратная
  частота) печатает `dataprep/preprocess.py`.
- Общий лосс: `policy_loss + w_frac · fraction_loss` (`sft/engine.py`). Value-голова не обучается
  (`value_weight=0`, данные winner-only); DDP идёт с `find_unused_parameters=True`, чтобы стерпеть её
  мёртвые градиенты. Лучший чекпойнт (`best.pt`) выбирается по **`send_acc`** на валидации.

## Офлайн-ETL (`dataprep/`)

Пять последовательных стадий, все офлайн (в сабмишн не входят). Легаси-парсер одним файлом — в
`dataprep/legacy/` (только для справки). Все стадии разом гоняет **оркестратор** `dataprep/build.py`
с прогресс-баром на каждом шаге (он же — путь для пересборки одной командой):

```bash
# докачать реплеи топ-80 команд и пересобрать весь датасет с балансом по командам:
.venv/bin/python -m dataprep.build --top 80 --cap 30000 --seed 0
# только пересборка из уже скачанных replays/ (без обращения к Kaggle):
.venv/bin/python -m dataprep.build --skip-download --cap 30000
```

`--overwrite` форсит convert собрать датасет с нуля; иначе convert докачивает только новые реплеи через
манифест, а производные файлы (filter/balance/preprocess) пересобираются всегда. Ниже — те же стадии по
отдельности:

```bash
# 1. Скачать публичные реплеи с лидерборда Kaggle (вежливо ~1 req/sec, резюмируемо, дедуп по id)
.venv/bin/python -m dataprep.download --top 50 --out replays/

# 2. Распарсить реплеи -> сэмплы {state, action, meta}
.venv/bin/python -m dataprep.convert --in "replays/*.json" --out data/samples.jsonl --who winner

# 3. Отфильтровать по классу действия. Доступные фильтры (--keep, по умолчанию full_send):
#    full_send       — hold ЛИБО вылет всего гарнизона (метки согласованы с декодом-«всё»)
#    partial_send    — hold ЛИБО любой вылет (включая ЧАСТИЧНЫЙ) — сигнал для головы числа кораблей
#    distinct_sources, single_launch — дополнительные срезы
.venv/bin/python -m dataprep.filter --in data/samples.jsonl --out data/samples.full_send.jsonl --keep full_send
.venv/bin/python -m dataprep.filter --in data/samples.jsonl --out data/samples.all_send.jsonl  --keep partial_send

# 4. Сбалансировать число сэмплов по командам (meta.team) — убрать перекос к частым победителям.
#    РЕКОМЕНДУЕТСЯ cap-only: оставить ВСЕ команды, обрезать только верх до N (анти-доминирование
#    без потери хвоста; распределение тяжелохвостое, строгое «поровну» выкидывает почти всё).
#    Отбор N из C внутри команды — алгоритм S (равномерно), сид --seed.
.venv/bin/python -m dataprep.balance --in data/samples.full_send.jsonl \
    --out data/samples.full_send.balanced.jsonl --cap 30000

# 5. Перевести каждый угол запуска -> планета-назначение (core.geo_lite.GeoEngine.planet_at_angle)
.venv/bin/python -m dataprep.preprocess --in data/samples.full_send.balanced.jsonl --out data/sft.full_send.jsonl
#    --horizon N  (горизонт прогноза для резолвера orbit_lite; дефолт 150, покрывает медленные долгие полёты)
```

Стадия **balance** (`dataprep/balance.py`) группирует по `meta.team` (имя команды из `info.TeamNames`,
для `--who winner` — победитель; численного id команды/сабмишна в реплеях нет). Распределение объёмов
команд **тяжелохвостое** (медиана ~сотни ходов, у топ-команд десятки-сотни тысяч), поэтому строгое
«у всех поровну» разрушительно (упирается в самую мелкую команду, выживает ~2-3%). Рекомендуемый
режим — **`--cap N`** (cap-only): оставить ВСЕ команды, обрезать только верх до `N` (нужное `N` из `C`
внутри команды берётся равномерно, алгоритм S). Есть и режим **коридора** `--tol`/`--center` (для
ровных распределений) и ручной `--target-n`. Балансируем именно **после фильтра** (filter режет у разных
команд разные доли). Поле `team` живёт в `samples.*.jsonl`, но дропается на шаге preprocess (тренер его
не читает — сплит по `meta.episode_id`), поэтому balance стоит до preprocess.
Сэмпловый баланс не ломает сплит train/val по эпизодам — эпизоды просто становятся меньше.

Выход шага 5 — обучающий файл: `{state, sends:[[from_id, dest_id, ships]], unresolved:[...], meta}`,
где `unresolved`-источники (угол не лёг ни на одну планету, ~0%) становятся `IGNORE` на обучении.
`ships` сохраняется специально для головы числа кораблей. `preprocess.py` печатает гистограмму бакетов
доли по резолвнутым вылетам и **ориентир для `train.frac_weights`** (обратная частота). Шаг 4 переведён
с `core.utils.planet_at_angle` (numpy) на `core.geo_lite.GeoEngine` (orbit_lite, через шим
`initial_planets` выше): метки ~99% совпадают со старым резолвером, остальное — случаи, которые
движок-верный orbit_lite резолвит, а старый дропал. **Поскольку метки могут сдвинуться — перегоняй
шаг 5 и переобучай** при переходе на него. Он строит torch-кэш движения на каждое состояние, поэтому
медленнее старого numpy-пути (нормально для одноразового офлайн-джоба).

**Подвох сдвига кадра** (`dataprep/convert.py`): kaggle_environments хранит `obs[t]` *после* применения
`action[t]`, то есть `action[t]` принят по `obs[t-1]`. Конвертер спаривает `(state_t, action_{t+1})`;
наивное спаривание в одном кадре даёт ~57% нарушений «послал больше кораблей, чем гарнизон» против 0%
при сдвиге.

Фильтр `full_send` существует, потому что декодер исторически слал весь гарнизон; обучение только на
ходах «hold-или-всё» держало метки согласованными с тем декодом. С появлением головы числа кораблей
(бакеты доли) частичные вылеты стали полезным сигналом — для этого есть `partial_send`/`all_send`
(`insights/` документируют эмпирическое округление дробного числа кораблей).

## Локальная оценка (`eval/`)

Офлайн-турнир, стравливающий чекпойнты и эвристики на **настоящем** движке Kaggle — `make('orbit_wars')`,
env **1.0.9**, та же версия, что у скачанных реплеев. Движок **не** в репо и **не** в `.venv`: ставится
из `requirements.txt` (`kaggle-environments==1.30.1`, `trueskill==0.4.5`). Установка
kaggle-environments тянет тяжёлые транзитивные зависимости (jax/transformers/litellm) — нормально для
офлайна, но раздувает `.venv`.

Вход — Hydra (зеркало `sft.train`):

```bash
# пул по умолчанию (бейзлайны-эвристики), 1v1
.venv/bin/python -m eval.run

# выбор агентов «флажками» — список ИМЁН из каталога configs/agent/
.venv/bin/python -m eval.run roster=[best, bestT, sniper] ckpt_dir=outputs/<ts>/checkpoints

# готовый пресет ростера из configs/pool/<name>.yaml (baselines / checkpoints / scripted)
.venv/bin/python -m eval.run pool=scripted mode=1v1 episodes=25

# ad-hoc чекпойнты из разных прогонов: имя в ростере -> путь через `ckpts`
.venv/bin/python -m eval.run roster=[runA,runB,sniper] \
    ckpts='{runA: outputs/A/checkpoints/best.pt, runB: outputs/B/checkpoints/best.pt}'

# разовый inline-список (переопределяет roster)
.venv/bin/python -m eval.run 'agents=[{label: best, ckpt: outputs/ts/checkpoints/best.pt}, {label: sn, heuristic: sniper}]'
```

Конфиг (`configs/eval.yaml`): каждый `agent/<name>.yaml` → узел `catalog.<name>`; `roster` выбирает
имена (имя не из каталога ищется в `ckpts` как ad-hoc путь к чекпойнту); группа `pool/*.yaml` — готовые
пресеты ростера. `mode` = число мест (`1v1` или `4p`).

Дизайн: каждый бот — `callable(obs, config) -> moves` (`eval/agents.py`). **Чекпойнт** грузится через
`PolicyValueNet.load(path)` (восстанавливает `ModelConfig`/`FeatureConfig` из `.pt`); его **режим
декода** — параметр интерфейса: `greedy` (argmax, как в сабмишне) или `sample` с температурой (сэмпл из
`softmax(logits/T)`; `act()` имеет knob `decode=`/`temperature=`, чтобы логика декода жила в одном
месте — оба выбора, и цели, и бакета доли, идут одним режимом). **Эвристики** (`HEURISTIC_KINDS`):
`sniper` (шлёт `target.ships+1` в слабейшую захватываемую достижимую цель через
`core.geo_lite.GeoEngine.validate_launch`), `full_send`, `random`, `hold`. **Скриптовые** боты (третий
вид): рукописные `orbit_lite`-планировщики в `agents/` (`agents.SCRIPTED_AGENTS` — `producer_hybrid`,
`apex_master`), задаются как `label=scripted:<name>` в инлайне или `{scripted: <name>}` в pool-yaml —
см. `configs/pool/scripted.yaml`.

`eval/runner.py` гоняет один эпизод и выводит полное **место** из финальных счётчиков кораблей (награда
движка отмечает только победителя, чего FFA-TrueSkill использовать не может). `eval/tournament.py` —
round-robin с циклической ротацией мест (карта 4-кратно симметрична → ротации гасят позиционный байас),
фиксированные сиды (одни и те же карты для каждого матча) и параллелизм по эпизодам
(`multiprocessing` fork; каждый чекпойнт грузится раз на воркер). `eval/rating.py` отдаёт TrueSkill
(μ/σ, Kaggle-шкала μ₀=600) плюс матрицу «A финишировал выше B». Таймауты в раннере намеренно подняты —
сравниваем качество политики, а не скорость инференса.

## Прочее

- `hydra_utils.py` — общий хелпер `print_cfg` для всех Hydra-входов (печатает полный конфиг на старте;
  лёгкий, без тяжёлых импортов, чтобы и `sft`, и `eval` могли его звать).
- `notebooks/` — Kaggle-ноутбуки: `kaggle_01_publish_data.ipynb` (публикация датасета),
  `kaggle_02_train_eval.ipynb` (обучение/эвал в Kaggle), `analyze_sft.ipynb` (анализ датасета).
- `insights/*.md` — заметки по механикам и датасету (формат, угол→планета, направление округления доли,
  гистограммы). Полные правила игры — в `orbit_wars_rules.md`.
