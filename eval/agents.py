"""Агенты для локального турнира: единый интерфейс ``callable(obs, config) -> ходы``.

Любой бот — вызываемый объект ``agent(obs, config)``, как ждёт ``kaggle_environments``
(``env.run``). Возврат — список ходов ``[from_planet_id, angle_rad, num_ships]``.

Два вида:
  * :class:`CheckpointAgent` — SFT-чекпойнт. Режим декода (``greedy`` / ``sample`` с
    температурой) задаётся ПАРАМЕТРОМ интерфейса, а не отдельным классом — argmax и
    стохастику выбираем при создании агента.
  * :class:`HeuristicAgent`  — жадные бейзлайны (``sniper`` / ``full_send`` / ``random`` /
    ``hold``) поверх :meth:`core.geo_lite.GeoEngine.validate_launch` (угол + проверка пути).

:func:`build_agent` собирает агента из CLI-спеки ``label=spec`` (см. ниже), чтобы тяжёлые
объекты (torch-модель) создавались уже внутри воркера, а не пиклились между процессами.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from core.geo_lite import GeoEngine
from model import PolicyValueNet

Move = List[float]   # [from_planet_id, angle_rad, num_ships]


def _field(obs: Any, key: str, default: Any = None) -> Any:
    """Достать поле obs (kaggle Struct — наследник dict; на всякий случай и атрибут)."""
    if isinstance(obs, dict):
        v = obs.get(key, default)
    else:
        v = getattr(obs, key, default)
    return default if v is None else v


class Agent:
    """База: вызываемый объект, совместимый с ``env.run`` движка."""

    name: str = "agent"

    def __call__(self, obs: Any, config: Any) -> List[Move]:
        raise NotImplementedError

    def seed(self, s: int) -> None:
        """Пере-сидировать RNG к ``s`` (для воспроизводимости стохастики; по умолчанию — ничего)."""

    def reset(self) -> None:
        """Сброс к исходному посеву конструктора (используется при прямом прогоне)."""
        self.seed(getattr(self, "seed_val", 0))


# --- чекпойнт ----------------------------------------------------------------
class CheckpointAgent(Agent):
    """SFT-чекпойнт как агент. ``decode`` ∈ {``greedy``, ``sample``}; ``temperature`` для sample."""

    def __init__(self, name: str, ckpt_path: str, *, decode: str = "greedy",
                 temperature: float = 1.0, device: str = "cpu", seed: int = 0):
        self.name = name
        self.ckpt_path = ckpt_path
        self.decode = decode
        self.temperature = float(temperature)
        self.device = device
        self.seed_val = int(seed)
        self.net, self.fcfg = PolicyValueNet.load(ckpt_path, map_location=device)
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(self.seed_val)

    def seed(self, s: int) -> None:
        self._gen.manual_seed(int(s))

    def __call__(self, obs: Any, config: Any) -> List[Move]:
        return self.net.act(obs, cfg=self.fcfg, decode=self.decode,
                            temperature=self.temperature, generator=self._gen)


# --- эвристики ---------------------------------------------------------------
HEURISTIC_KINDS = ("sniper", "full_send", "random", "hold")


class HeuristicAgent(Agent):
    """Жадный бейзлайн без обучения.

    ``sniper``    — каждая своя планета шлёт ``target.ships+1`` в слабейшую захватываемую
                    и достижимую (прямой путь свободен) чужую/нейтральную цель;
    ``full_send`` — то же, но шлём весь гарнизон в слабейшую достижимую цель;
    ``random``    — весь гарнизон в случайную достижимую цель;
    ``hold``      — не ходит (опорный «ничего не делаю»).
    """

    def __init__(self, name: str, kind: str = "sniper", *, seed: int = 0):
        if kind not in HEURISTIC_KINDS:
            raise ValueError(f"неизвестная эвристика {kind!r}; есть {HEURISTIC_KINDS}")
        self.name = name
        self.kind = kind
        self.seed_val = int(seed)
        self._rng = random.Random(self.seed_val)

    def seed(self, s: int) -> None:
        self._rng = random.Random(int(s))

    def __call__(self, obs: Any, config: Any) -> List[Move]:
        if self.kind == "hold":
            return []
        player = int(_field(obs, "player", 0))
        planets = [[float(c) for c in row] for row in _field(obs, "planets", [])]
        n = len(planets)
        mine = [i for i in range(n)
                if int(planets[i][1]) == player and planets[i][5] > 0]

        geo = None                               # geo_lite-движок строим лениво (один раз)
        moves: List[Move] = []
        for fi in mine:
            garrison = int(planets[fi][5])
            cands = [ti for ti in range(n)
                     if ti != fi and int(planets[ti][1]) != player]
            if self.kind == "random":
                self._rng.shuffle(cands)
            else:                                 # sniper / full_send: слабейшие первыми
                cands.sort(key=lambda ti: planets[ti][5])
            for ti in cands:
                tgt_ships = int(planets[ti][5])
                if self.kind == "sniper":
                    send = tgt_ships + 1          # ровно на захват (бой: нужно > гарнизона)
                    if send > garrison:
                        continue                  # не вытянем — следующая цель
                else:
                    send = garrison               # full_send / random — весь гарнизон
                if send <= 0:
                    continue
                if geo is None:
                    geo = GeoEngine(obs, player=player)
                plan = geo.validate_launch(int(planets[fi][0]), int(planets[ti][0]), send)
                if plan.reaches and plan.safe:
                    moves.append([float(planets[fi][0]), float(plan.angle), int(send)])
                    break                         # один залп с планеты за ход
        return moves


# --- спека агента / фабрика --------------------------------------------------
@dataclass(frozen=True)
class AgentSpec:
    """Разобранная CLI-спека агента (создаётся фабрикой, конструируется в воркере)."""

    label: str
    kind: str                       # 'checkpoint' | 'heuristic' | 'scripted'
    ckpt_path: Optional[str] = None
    decode: str = "greedy"
    temperature: float = 1.0
    heuristic: Optional[str] = None
    scripted: Optional[str] = None  # имя в agents.SCRIPTED_AGENTS


def parse_spec(token: str) -> AgentSpec:
    """Разобрать ``label=spec``.

    Чекпойнт:  ``label=path/to.pt``  |  ``label=path/to.pt:sample:0.7``
    Эвристика: ``label=heuristic:sniper`` (kind ∈ sniper/full_send/random/hold)
    Скрипт:    ``label=scripted:apex_master`` (имя ∈ agents.SCRIPTED_AGENTS)
    Если в токене нет ``=`` — label выводится из спеки.
    """
    if "=" in token:
        label, spec = token.split("=", 1)
    else:
        label, spec = "", token
    spec = spec.strip()
    if spec.startswith("heuristic:"):
        kind = spec.split(":", 1)[1] or "sniper"
        return AgentSpec(label=label or kind, kind="heuristic", heuristic=kind)
    if spec.startswith("scripted:"):
        name = spec.split(":", 1)[1]
        return AgentSpec(label=label or name, kind="scripted", scripted=name)
    parts = spec.split(":")
    path = parts[0]
    decode = parts[1] if len(parts) > 1 and parts[1] else "greedy"
    temp = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
    label = label or path.split("/")[-1]
    return AgentSpec(label=label, kind="checkpoint", ckpt_path=path,
                     decode=decode, temperature=temp)


def spec_from_config(entry: Any) -> AgentSpec:
    """AgentSpec из записи конфига Hydra — строка ``label=spec`` ИЛИ маппинг.

    Чекпойнт:  ``{label, ckpt, decode?, temperature?}`` (decode=greedy|sample)
    Эвристика: ``{label?, heuristic}`` (heuristic=sniper|full_send|random|hold)
    Скрипт:    ``{label?, scripted}`` (scripted ∈ agents.SCRIPTED_AGENTS)
    Строка разбирается через :func:`parse_spec` (тот же синтаксис, что в argparse-CLI).
    """
    if isinstance(entry, str):
        return parse_spec(entry)
    get = entry.get
    if get("heuristic"):
        kind = str(get("heuristic"))
        return AgentSpec(label=str(get("label") or kind), kind="heuristic", heuristic=kind)
    if get("scripted"):
        name = str(get("scripted"))
        return AgentSpec(label=str(get("label") or name), kind="scripted", scripted=name)
    ckpt = get("ckpt") or get("ckpt_path") or get("path")
    if not ckpt:
        raise ValueError(f"спека агента без 'heuristic' и без 'ckpt': {entry}")
    decode = str(get("decode") or "greedy")
    temp = float(get("temperature") if get("temperature") is not None else 1.0)
    label = str(get("label") or str(ckpt).split("/")[-1])
    return AgentSpec(label=label, kind="checkpoint", ckpt_path=str(ckpt),
                     decode=decode, temperature=temp)


def build_agent(spec: AgentSpec, *, device: str = "cpu", seed: int = 0) -> Agent:
    """Сконструировать агента из спеки (вызывается внутри воркера)."""
    if spec.kind == "heuristic":
        return HeuristicAgent(spec.label, spec.heuristic or "sniper", seed=seed)
    if spec.kind == "scripted":
        from agents import SCRIPTED_AGENTS    # ленивый импорт: тянет torch+orbit_lite только если нужен
        return SCRIPTED_AGENTS[spec.scripted](spec.label, seed=seed)
    return CheckpointAgent(spec.label, spec.ckpt_path, decode=spec.decode,
                           temperature=spec.temperature, device=device, seed=seed)
