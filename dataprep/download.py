r"""Офлайн-загрузчик реплеев Orbit Wars с лидерборда — для обучения (SFT/BC).

ГРАНИЦЫ ИСПОЛЬЗОВАНИЯ (важно — чтобы не словить бан)
----------------------------------------------------
Инструмент ЭТАПА РАЗРАБОТКИ. Качаем реплеи -> учим модель ОФЛАЙН. Боевой сабмит
во время эпизода НИЧЕГО не качает и не шлёт наружу (§2.12 правил). Реплеи
официально публичны и качаемы (§2.11), как внешние данные допустимы (§2.6).

Реальный риск «блокировки» — не правила компа, а ФЛУД по API. Поэтому здесь:
  * пауза между скачиваниями реплеев (--sleep, по умолч. 1.2 c — ~1 req/sec);
  * докачка: уже скачанные episode-<id>-replay.json пропускаются;
  * дедуп episode_id (одни и те же матчи всплывают у разных команд);
  * жёсткие лимиты (--max-replays, --per-sub) с ЛОГОМ обрезаний;
  * мягкое предупреждение при подходе к ~3000 скачиваний за прогон
    (community-ориентир суточного потолка ~3600/день; официально не задокументирован).

ЦЕПОЧКА (всё на официальном Kaggle Public API, без Meta Kaggle и серых эндпоинтов)
--------------------------------------------------------------------------------
  leaderboard (team_id)            competition_leaderboard_view
    -> team-submissions (id)       competition_team_submissions
      -> episodes (episode_id)     competition_list_episodes
        -> replay (.json)          competition_episode_replay
Документированного teamId->submissionId иначе нет; team-submissions отдаёт
«public-safe» (лидербордные) сабмишены команды — этого хватает для сильной игры.

ТРЕБОВАНИЯ
----------
  pip install kaggle
  ~/.kaggle/kaggle.json  (Kaggle -> Settings -> Create New API Token), chmod 600

ЗАПУСК
------
  # реплеи топ-20 команд лидерборда (старт с маленького лимита!):
  python -m dataprep.download --top 20 --out replays/ --max-replays 50

  # снять лимит, когда убедился, что всё ок:
  python -m dataprep.download --top 50 --out replays/ --per-sub 100

  # конкретные команды / сабмишены:
  python -m dataprep.download --teams 15649057,15653847 --out replays/
  python -m dataprep.download --subs 53517858 --out replays/

  # затем конвертация в SFT-семплы:
  python -m dataprep.convert --in "replays/*.json" --out data/samples.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Set

SOFT_DAILY_WARN = 3000          # мягкий порог за прогон (ниже community-оценки ~3600/день)


# --- доступ к API ------------------------------------------------------------
def get_api():
    """Аутентифицированный Kaggle API. Падает с понятным текстом без кред."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:                        # noqa: BLE001
        raise SystemExit(f"Не импортируется kaggle: {e}\nУстанови: pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:                         # noqa: BLE001
        raise SystemExit(f"Аутентификация Kaggle не прошла: {e}\n"
                         f"Положи ~/.kaggle/kaggle.json (chmod 600).")
    return api


# --- перечисление team_id / submission_id / episode_id -----------------------
def top_team_ids(api, comp: str, n: int) -> List[int]:
    """Top-N team_id с лидерборда (с пагинацией по page_token)."""
    out: List[int] = []
    token: Optional[str] = None
    while len(out) < n:
        page = api.competition_leaderboard_view(
            comp, page_size=min(200, n - len(out)), page_token=token)
        if not page:
            break
        out.extend(int(r.team_id) for r in page)
        token = getattr(page, "next_page_token", None) or getattr(page[-1], "next_page_token", None)
        if not token:
            break
    return out[:n]


def submission_ids_for_team(api, team_id: int) -> List[int]:
    """submission_id'ы команды (public-safe / лидербордные)."""
    subs = api.competition_team_submissions(team_id) or []
    return [int(s.id) for s in subs]


def episode_ids_for_sub(api, sub_id: int) -> List[int]:
    """episode_id'ы одного сабмишена."""
    eps = api.competition_list_episodes(sub_id) or []
    return [int(e.id) for e in eps]


def download_replay(api, episode_id: int, out_dir: str) -> bool:
    """Скачивает реплей эпизода. True — скачали, False — уже на диске."""
    dst = os.path.join(out_dir, f"episode-{episode_id}-replay.json")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return False
    api.competition_episode_replay(episode_id, path=out_dir, quiet=True)
    return True


# --- основной цикл -----------------------------------------------------------
def harvest(api, sub_ids: List[int], out_dir: str, *, sleep: float,
            max_replays: Optional[int], per_sub: Optional[int]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    seen_eps: Set[int] = set()
    downloaded = skipped = errors = 0
    capped = False

    for i, sub in enumerate(sub_ids):
        try:
            eps = episode_ids_for_sub(api, sub)
        except Exception as e:                     # noqa: BLE001
            print(f"[{i+1}/{len(sub_ids)}] sub {sub}: список эпизодов не получен: {e}")
            continue
        if per_sub is not None and len(eps) > per_sub:
            print(f"[{i+1}/{len(sub_ids)}] sub {sub}: {len(eps)} эпизодов -> беру первые {per_sub}")
            eps = eps[:per_sub]
        else:
            print(f"[{i+1}/{len(sub_ids)}] sub {sub}: {len(eps)} эпизодов")

        for ep in eps:
            if ep in seen_eps:
                continue
            seen_eps.add(ep)
            if max_replays is not None and downloaded >= max_replays:
                capped = True
                break
            try:
                got = download_replay(api, ep, out_dir)
            except Exception as e:                 # noqa: BLE001
                errors += 1
                print(f"  [err ep {ep}] {str(e)[:100]}")
                time.sleep(sleep)
                continue
            if got:
                downloaded += 1
                if downloaded % 25 == 0:
                    print(f"  …скачано {downloaded}")
                if downloaded == SOFT_DAILY_WARN:
                    print(f"  !! приближаешься к ~{SOFT_DAILY_WARN}+ за прогон — "
                          f"возможен суточный rate-limit аккаунта, лучше сделать паузу")
                time.sleep(sleep)                  # пауза только после реального запроса
            else:
                skipped += 1
        if capped:
            print(f"!! достигнут --max-replays={max_replays}, останавливаюсь")
            break

    print(f"\nИтог: скачано {downloaded}, пропущено (уже было) {skipped}, "
          f"ошибок {errors}, уникальных эпизодов {len(seen_eps)}"
          f"{' [ОБРЕЗАНО ЛИМИТОМ]' if capped else ''}")
    print(f"Дальше: python -m dataprep.convert --in \"{out_dir}/*.json\" --out data/samples.jsonl")


def collect_sub_ids(api, args) -> List[int]:
    """Определяет список submission_id по выбранному режиму CLI."""
    if args.subs:
        return [int(s) for s in args.subs.split(",") if s.strip()]

    if args.teams:
        team_ids = [int(t) for t in args.teams.split(",") if t.strip()]
    elif args.top:
        print(f"Беру top-{args.top} команд лидерборда…")
        team_ids = top_team_ids(api, args.comp, args.top)
        print(f"  team_id: {team_ids}")
    else:
        raise SystemExit("Укажи источник: --top N | --teams ids | --subs ids")

    sub_ids: List[int] = []
    for tid in team_ids:
        try:
            sids = submission_ids_for_team(api, tid)
        except Exception as e:                     # noqa: BLE001
            print(f"  team {tid}: сабмишены не получены: {e}")
            continue
        sub_ids.extend(sids)
        time.sleep(min(args.sleep, 0.5))           # лёгкая пауза на этапе перечисления
    # дедуп с сохранением порядка
    seen: Set[int] = set()
    uniq = [s for s in sub_ids if not (s in seen or seen.add(s))]
    print(f"Найдено сабмишенов: {len(uniq)}")
    return uniq


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Вежливый офлайн-загрузчик реплеев Orbit Wars с лидерборда")
    ap.add_argument("--comp", default="orbit-wars", help="competition slug")
    src = ap.add_argument_group("источник (выбери один)")
    src.add_argument("--top", type=int, help="взять top-N команд лидерборда")
    src.add_argument("--teams", help="team_id через запятую")
    src.add_argument("--subs", help="submission_id через запятую (минуя лидерборд)")
    ap.add_argument("--out", default="replays", help="папка для реплеев")
    ap.add_argument("--sleep", type=float, default=1.2, help="пауза между скачиваниями, c (~1 req/sec)")
    ap.add_argument("--max-replays", type=int, help="жёсткий потолок числа скачиваний за прогон")
    ap.add_argument("--per-sub", type=int, help="макс. эпизодов на один сабмишен")
    args = ap.parse_args(argv)

    api = get_api()
    sub_ids = collect_sub_ids(api, args)
    if not sub_ids:
        print("Сабмишены не найдены — нечего качать.", file=sys.stderr)
        return 2
    harvest(api, sub_ids, args.out, sleep=args.sleep,
            max_replays=args.max_replays, per_sub=args.per_sub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
