# Формат датасета Orbit Wars (replays → samples.jsonl)

**Дата:** 2026-06-13 · **Источники:** `replays/*.json` (1183 реплея), `data/samples.jsonl` (308 264 записи) · **Скрипты:** `dataprep/download.py`, `dataprep/convert.py`

Две стадии: **сырые реплеи Kaggle** → офлайн-конвертер → **SFT-семплы** для behavioral cloning.

---

## 1. Сырой реплей — `replays/episode-<EpisodeId>-replay.json`

Стандартный дамп `kaggle_environments`. Топ-левел ключи:

| ключ | что |
|---|---|
| `steps` | `list[500]`; `steps[t]` — список по агентам, `steps[t][p]` = `{action, observation, reward, status, info}` |
| `configuration` | `actTimeout=1, agentTimeout=2, runTimeout=1200, episodeSteps=500, cometSpeed=4.0, shipSpeed=6.0, seed` |
| `info` | `EpisodeId`, `TeamNames`, `Agents[].Name`, `seed`, `LiveVideoPath` |
| `rewards`, `statuses` | по 2 элемента (итоговые) |
| `id` | UUID; `name="orbit_wars"`, `module_version`, `schema_version`, `version` |

**Внутри `observation` (глобальное состояние лежит в obs игрока 0):**
- `planets` — строки `[id, owner, x, y, radius, ships, production]`. `owner = -1` — нейтральная.
- `fleets` — строки `[id, owner, x, y, angle, from_planet_id, ships]`.
- `comet_planet_ids`, `comets`, `angular_velocity`, `next_fleet_id`, `step`, `remainingOverageTime`.
- `initial_planets` — стартовая раскладка (есть только в obs, в фичах не используется).

**`action` агента:** `[[from_id, angle, ships], ...]` — список вылетов (может быть несколько с разных планет за ход).

### ⚠️ datetime в реплее НЕТ
Ни `create_time`/`end_time`, ни какого-либо timestamp реального времени. Есть только `EpisodeId` и игровой `step` (тик, не время). Время игры эпизода отдаёт **только Kaggle API** (`competition_list_episodes(sub_id)` → `ApiEpisode.create_time/end_time`), но `dataprep/download.py:100` берёт из ответа лишь `e.id`, поэтому локально время не сохранено — восстанавливается только повторным запросом к API по `submission_id`. См. [[orbit-wars-sft-replays]].

---

## 2. SFT-датасет — `data/samples.jsonl` (+ `.manifest`)

JSONL, одна запись = `{state, action, meta}`:

```json
{
  "state":  { ... obs победителя на шаге t ... },
  "action": [[14, -1.5938589971756159, 21]],   // ход(ы) [from_id, angle, ships]
  "meta":   { ... }
}
```

- **`state`** — то же obs, что выше, но `initial_planets` удалён. Ключи: `angular_velocity, comet_planet_ids, comets, fleets, next_fleet_id, planets, player, remainingOverageTime, step`. Подаётся в `features.encode`.
- **`action`** — экспертный таргет policy-головы. `[]` если хода не было (при `--who winner` пустые отбрасываются → 308 264 записи из ~250 эпизодов победителей).
- **`meta`** (13 полей): `episode` (UUID), `episode_id`, `seed`, `step`, `player`, `team`, `teams[]`, `is_winner`, `winner`, `n_players`, `final_score`, `rewards[]`, `source` (имя файла реплея). **datetime здесь тоже нет.**

`.manifest` — список уже обработанных реплеев; повторный запуск конвертера докачивает только новые (resume), `--overwrite` пересобирает с нуля.

### ⚠️ Сдвиг obs↔action на кадр (ключевая тонкость)
`kaggle_environments` пишет `steps[t].observation` уже ПОСЛЕ применения `steps[t].action`. Поэтому ход `action[t]` принят по `obs[t-1]`. Конвертер спаривает **(state шага t, action шага t+1)** → 0% «отправлено больше гарнизона» вместо ~57% при наивном спаривании. Последний шаг эпизода отбрасывается. Подробности и следствие для головы числа кораблей — [[sft-shipcount-analysis]].
