r"""Оркестратор пересборки SFT-датасета Orbit Wars — все стадии одной командой.

Прогоняет офлайн-ETL целиком, с баннером ``=== [k/N] стадия ===`` перед каждой
стадией (макро-прогресс) и собственным tqdm-баром внутри каждой (микро-прогресс):

    download -> convert -> filter -> BALANCE(по команде) -> preprocess
    replays/   samples    samples.<keep>   samples.<keep>.balanced   sft.<keep>.jsonl
                .jsonl       .jsonl              .jsonl               (читает тренер)

Стадия BALANCE (``dataprep/balance.py``) выравнивает число сэмплов по командам
(``meta.team``), убирая доминирование частых победителей. Балансим ПОСЛЕ фильтра —
тогда баланс в итоговом обучающем файле точный.

OVERWRITE / RESUME
------------------
``--overwrite`` форсит convert собрать датасет С НУЛЯ (иначе convert докачивает только
новые реплеи через манифест — дёшево добавить реплеев и прогнать заново). Производные
файлы (выходы filter/balance/preprocess) ВСЕГДА пересобираются — они полностью
определяются своим входом.

ИСПОЛЬЗОВАНИЕ
-------------
  # докачать реплеи топ-80 команд и пересобрать всё:
  python -m dataprep.build --top 80 --min-team 200 --seed 0

  # только пересборка из уже скачанных replays/ (без обращения к Kaggle):
  python -m dataprep.build --skip-download --min-team 200
"""

from __future__ import annotations

# --- bootstrap: корень репо в sys.path (для запуска и как `-m`, и как файл) ---
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
from types import SimpleNamespace
from typing import List, Optional

from dataprep import balance, convert, download
from dataprep import filter as filt
from dataprep import preprocess


def _banner(step: int, total: int, name: str) -> None:
    """Макро-прогресс: заметная шапка перед стадией."""
    print(f"\n{'=' * 60}\n=== [{step}/{total}] {name}\n{'=' * 60}")


def run_pipeline(args: argparse.Namespace) -> str:
    """Прогоняет все стадии, возвращает путь к итоговому обучающему файлу."""
    keep_tag = "_".join(args.keep)                       # full_send -> "full_send"
    data_dir = args.data_dir
    samples = os.path.join(data_dir, "samples.jsonl")
    filtered = os.path.join(data_dir, f"samples.{keep_tag}.jsonl")
    balanced = os.path.join(data_dir, f"samples.{keep_tag}.balanced.jsonl")
    final = os.path.join(data_dir, f"sft.{keep_tag}.jsonl")

    stages: List[str] = []
    if not args.skip_download:
        stages.append("download")
    stages += ["convert", "filter", "balance", "preprocess"]
    total = len(stages)
    step = 0

    # 1) download (опционально) ------------------------------------------------
    if not args.skip_download:
        step += 1
        _banner(step, total, f"download -> {args.out_dir}/")
        api = download.get_api()
        ns = SimpleNamespace(subs=args.subs, teams=args.teams, top=args.top,
                             comp=args.comp, sleep=args.sleep)
        sub_ids = download.collect_sub_ids(api, ns)
        download.harvest(api, sub_ids, args.out_dir, sleep=args.sleep,
                         max_replays=args.max_replays, per_sub=args.per_sub)

    # 2) convert ---------------------------------------------------------------
    step += 1
    _banner(step, total, f"convert ({args.who}) -> {samples}")
    files = convert._expand_inputs([args.out_dir])
    if not files:
        raise SystemExit(f"в {args.out_dir}/ нет реплеев (.json) — нечего конвертировать")
    n_written = convert.convert(files, samples, who=args.who, overwrite=args.overwrite)
    print(f"  записано {n_written:,} семплов -> {samples}")

    # 3) filter ----------------------------------------------------------------
    step += 1
    _banner(step, total, f"filter ({', '.join(args.keep)}) -> {filtered}")
    preds = {name: filt.FILTERS[name] for name in args.keep}
    fstats = filt.filter_dataset(samples, filtered, predicates=preds, overwrite=True)
    print(f"  оставлено {fstats['kept']:,}/{fstats['total']:,} -> {filtered}")

    # 4) balance (по команде) --------------------------------------------------
    step += 1
    _banner(step, total, f"balance (по meta.{args.key}) -> {balanced}")
    bstats = balance.balance_by_team(filtered, balanced, key=args.key, tol=args.tol,
                                     center=args.center, target_n=args.target_n,
                                     cap=args.cap, min_team=args.min_team, seed=args.seed,
                                     overwrite=True)
    balance._report(bstats, balanced)

    # 5) preprocess (угол -> планета, лейблинг) --------------------------------
    step += 1
    _banner(step, total, f"preprocess -> {final}")
    preprocess.run(balanced, final, horizon=args.horizon,
                   insights_path=(args.insights or None))

    print(f"\n{'=' * 60}\nГОТОВО. Обучающий файл: {final}\n"
          f"Дальше: .venv/bin/python -m sft.check --path {final}\n{'=' * 60}")
    return final


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Пересборка SFT-датасета Orbit Wars: download -> convert -> "
                    "filter -> balance(по команде) -> preprocess")
    # источник / download
    ap.add_argument("--skip-download", action="store_true",
                    help="не качать реплеи, собрать из уже имеющихся в --out-dir")
    ap.add_argument("--comp", default="orbit-wars", help="competition slug")
    ap.add_argument("--top", type=int, help="взять реплеи top-N команд лидерборда")
    ap.add_argument("--teams", help="team_id через запятую (вместо --top)")
    ap.add_argument("--subs", help="submission_id через запятую (минуя лидерборд)")
    ap.add_argument("--out-dir", default="replays", help="папка реплеев")
    ap.add_argument("--sleep", type=float, default=1.2, help="пауза между скачиваниями, c")
    ap.add_argument("--max-replays", type=int, help="потолок скачиваний за прогон")
    ap.add_argument("--per-sub", type=int, help="макс. эпизодов на сабмишен")
    # пути / стадии
    ap.add_argument("--data-dir", default="data", help="папка промежуточных и итоговых файлов")
    ap.add_argument("--who", default="winner", help="чьи ходы брать: winner | all | top:N")
    ap.add_argument("--keep", nargs="+", choices=list(filt.FILTERS),
                    default=list(filt.DEFAULT_KEEP), help="фильтры (AND), по умолч. full_send")
    # balance
    ap.add_argument("--key", default="team", help="поле meta для группировки команд")
    ap.add_argument("--tol", type=float, default=0.1,
                    help="допуск отклонения объёма команды от центра (доля, по умолч. 0.1)")
    ap.add_argument("--center", choices=["median", "mean"], default="median",
                    help="центр распределения для коридора балансировки (по умолч. median)")
    ap.add_argument("--cap", type=int, default=None,
                    help="balance cap-only: оставить ВСЕ команды, обрезать только верх до N")
    ap.add_argument("--target-n", type=int, default=None,
                    help="ручной режим: ровно N сэмплов на команду (переопределяет --tol)")
    ap.add_argument("--min-team", type=int, default=0,
                    help="жёсткий префильтр balance: выкинуть команды с числом сэмплов < N")
    ap.add_argument("--seed", type=int, default=0, help="сид отбора в balance")
    # preprocess
    ap.add_argument("--horizon", type=int, default=preprocess.geo_lite.DEFAULT_HORIZON,
                    help="горизонт прогноза orbit_lite в preprocess")
    ap.add_argument("--insights", default="insights/sft-angle-to-planet.md",
                    help="куда положить md-сводку preprocess (пусто -> не писать)")
    # общее
    ap.add_argument("--overwrite", action="store_true",
                    help="форсить convert собрать датасет с нуля (иначе резюмирует по манифесту)")
    args = ap.parse_args(argv)

    if not args.skip_download and not (args.top or args.teams or args.subs):
        ap.error("укажи источник реплеев: --top N | --teams ids | --subs ids "
                 "(или --skip-download для сборки из имеющихся)")

    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
