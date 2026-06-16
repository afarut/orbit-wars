# Orbit Wars — Competition Rules & Specs

> Featured Simulation Competition · Kaggle
> Conquer planets rotating around a sun in continuous 2D space. A real-time strategy game for 2 or 4 players.
> Source: `Orbit Wars _ Kaggle.html` (extracted)

---

## Overview

The goal of this competition is to create and/or train AI bots to play a novel multi-agent **1v1** or **4p FFA** game against other submitted agents.

Orbit Wars resurrects the strategy of the **2010 Planet Wars** challenge with fresh mechanics. Launch swarms of ships across the solar system, outmaneuver enemies, and claim orbital supremacy.

---

## Evaluation

- Each team may submit up to **5 agents (bots) per day**.
- Each submission plays **Episodes (games)** against other bots on the ladder with **similar skill rating**.
- Skill rating goes **up on wins**, **down on losses**, **evens out on ties**.
- Only the **latest 2 submissions** are tracked for final submissions (to reduce bot count and increase episodes/team).
- Every submitted bot keeps playing episodes until the competition ends; **newer bots play more frequently**.
- Leaderboard shows only your **best-scoring bot**; track all submissions on your Submissions page.

### Skill Rating Model
- Modeled by a Gaussian **N(μ, σ²)**: μ = estimated skill, σ = uncertainty (decreases over time).
- On upload: a **Validation Episode** runs the submission against copies of itself. If it fails → marked **Error** (download agent logs to debug).
- On success: initialized at **μ₀ = 600**, joins the pool of All Submissions.
- Episodes pair submissions with similar ratings for fair matches. New agents get an increased episode rate for faster feedback.

### Ranking System
- After each Episode, ratings update for all participating Submissions.
- Winner's μ increases, loser's μ decreases; a draw moves the two μ values toward their mean.
- Update magnitude is relative to the deviation from the expected result AND to each Submission's σ.
- σ is reduced relative to the information gained.
- **The margin of victory (score) does NOT affect rating updates** — only win/loss/draw.

### Final Evaluation
- **Submission deadline: June 23, 2026** — further submissions locked.
- From June 23, 2026 for ~2 weeks, games keep running.
- At the end of that period the leaderboard is **final**.

---

## Timeline

| Date | Event |
|------|-------|
| **April 16, 2026** | Start Date |
| **June 16, 2026** | Entry Deadline (must accept rules before this date) |
| **June 16, 2026** | Team Merger Deadline |
| **June 23, 2026** | Final Submission Deadline |
| **June 24 → ~July 8, 2026** | Games continue until leaderboard converges, then final |

All deadlines at 11:59 PM UTC unless noted.

---

## Prizes

$50,000 total. **1st–10th Place: $5,000 each.**

---

## How to Play Orbit Wars

### Overview
Players start with a **single home planet** and compete to control the map by sending fleets to capture neutral and enemy planets.
- Board: **100×100 continuous space**, sun at the center.
- Planets **orbit** the sun; comets fly through on **elliptical trajectories**; fleets travel in **straight lines**.
- Game lasts **500 turns**.
- **Winner = player with the most total ships (on planets + in fleets) at the end.**

### Board Layout
- **Board**: 100×100 continuous space, origin at top-left.
- **Sun**: centered at **(50, 50)**, radius **10**. Fleets that cross the sun are **destroyed**.
- **Symmetry**: all planets/comets placed with **4-fold mirror symmetry** around center: `(x, y)`, `(100-x, y)`, `(x, 100-y)`, `(100-x, 100-y)`. Ensures fairness regardless of start position.

### Planets
Represented as `[id, owner, x, y, radius, ships, production]`.
- **owner**: Player ID (0–3), or **-1 for neutral**.
- **radius**: `1 + ln(production)` — higher-production planets are physically larger.
- **production**: integer **1–5**. Each turn an owned planet generates this many ships.
- **ships**: current garrison. Starts between **5 and 99** (skewed toward lower values).

#### Planet Types
- **Orbiting planets**: those with `orbital_radius + planet_radius < 50` rotate around the sun at constant angular velocity (**0.025–0.05 rad/turn**, randomized per game). Use `initial_planets` and `angular_velocity` to predict positions.
- **Static planets**: further from center, do not rotate.
- Map has **20–40 planets** (5–10 symmetric groups of 4). **At least 3 groups static**, **at least 1 group orbiting**.

#### Home Planets
- One symmetric group is randomly chosen as starting planets.
- **2-player**: players start on diagonally opposite planets (Q1 and Q4).
- **4-player**: each player gets one planet from the group.
- Home planets start with **10 ships**.

### Fleets
Represented as `[id, owner, x, y, angle, from_planet_id, ships]`.
- **angle**: direction of travel in radians.
- **ships**: number in the fleet (does NOT change during travel).

#### Fleet Speed
Scales with size on a logarithmic curve:
```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```
- 1 ship → 1.0 units/turn.
- Larger fleets move faster, approaching max speed (default **6.0**).
- ~500 ships → ~5; ~1000 ships → max.

#### Fleet Movement
- Travel in a straight line at computed speed each turn.
- A fleet is **removed** if it:
  - Goes **out of bounds** (leaves the 100×100 field).
  - **Crosses the sun** (path segment within sun radius).
  - **Collides with any planet** (path segment within planet radius) → triggers **combat**.
- **Continuous collision detection**: the entire path segment from old → new position is checked, not just the endpoint.

#### Fleet Launch
Each turn the agent returns moves: `[from_planet_id, direction_angle, num_ships]`.
- Launch only from planets **you own**.
- Cannot launch more ships than the planet currently has.
- Fleet spawns just **outside the planet's radius** in the given direction.
- Multiple launches per turn allowed (same or different planets).

### Comets
Temporary extra-solar objects on highly elliptical orbits around the sun. Spawn in **groups of 4** (one per quadrant) at steps **50, 150, 250, 350, 450**.
- **Radius**: 1.0 (fixed).
- **Production**: 1 ship/turn when owned.
- **Starting ships**: random, skewed low (min of 4 rolls from 1–99). All 4 comets in a group share the same starting count.
- **Speed**: `cometSpeed` (default **4.0** units/turn).
- **Identification**: check `comet_planet_ids` in the observation. Comets also appear in the `planets` array and follow all normal planet rules (capture, production, launch, combat).
- When a comet leaves the board, it's removed along with its garrisoned ships. **Comets are removed before fleet launches each turn**, so you cannot launch from a departing comet.
- The `comets` field has `paths` (full trajectory per comet) and `path_index` (current position), usable to predict future positions.

### Turn Order
Each turn executes in this order:
1. **Comet expiration** — remove comets that have left the board.
2. **Comet spawning** — spawn new comet groups at designated steps.
3. **Fleet launch** — process all player actions, create new fleets.
4. **Production** — all owned planets (including comets) generate ships.
5. **Fleet movement** — move all fleets; check out-of-bounds, sun collision, planet collision. Fleets hitting planets are queued for combat.
6. **Planet rotation & comet movement** — orbiting planets rotate, comets advance. Any fleet caught by a moving planet/comet is swept into combat.
7. **Combat resolution** — resolve all queued planet combats.

### Combat
When one or more fleets collide with a planet (by flying into it or being swept by a moving one):
1. All arriving fleets grouped by owner; same-owner ships are summed.
2. The **largest attacking force fights the second largest**; the difference survives.
3. If there is a surviving attacker:
   - Same owner as planet → surviving ships added to garrison.
   - Different owner → surviving ships fight the garrison. If attackers **exceed** the garrison, the planet **changes ownership** and garrison becomes the surplus.
4. If two attackers **tie**, all attacking ships are destroyed (no survivors).

### Scoring and Termination
Game ends when:
- **Step limit** reached (500 turns), OR
- **Elimination**: only one player (or zero) remains with planets or fleets.

**Final score = total ships on owned planets + total ships in owned fleets. Highest wins.**

---

## Observation Reference

| Field | Type | Description |
|-------|------|-------------|
| `planets` | `[[id, owner, x, y, radius, ships, production], ...]` | All planets including comets |
| `fleets` | `[[id, owner, x, y, angle, from_planet_id, ships], ...]` | All active fleets |
| `player` | int | Your player ID (0–3) |
| `angular_velocity` | float | Planet rotation speed (radians/turn) |
| `initial_planets` | `[[id, owner, x, y, radius, ships, production], ...]` | Planet positions at game start |
| `comets` | `[{planet_ids, paths, path_index}, ...]` | Active comet group data |
| `comet_planet_ids` | `[int, ...]` | Planet IDs that are comets |
| `remainingOverageTime` | float | Remaining overage time budget (seconds) |

### Action Format
Return a list of moves:
```
[[from_planet_id, direction_angle, num_ships], ...]
```
- `from_planet_id`: ID of a planet you own.
- `direction_angle`: angle in radians (**0 = right, π/2 = down**).
- `num_ships`: integer number of ships to send.
- Return `[]` to take no action.

### Agent Convenience
The module exports named tuples for easier field access:
```python
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

def agent(obs):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets  = [Fleet(*f)  for f in obs.get("fleets", [])]
    player  = obs.get("player", 0)
    for p in planets:
        print(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
    return []  # list of [from_planet_id, angle, num_ships]
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodeSteps` | 500 | Maximum number of turns |
| `actTimeout` | 1 | Seconds per turn |
| `shipSpeed` | 6.0 | Maximum fleet speed |
| `sunRadius` | 10.0 | Radius of the sun |
| `boardSize` | 100.0 | Board dimensions |
| `cometSpeed` | 4.0 | Comet speed (units/turn) |

---

## Getting Started (AGENTS.md)

Your agent is a function that receives an observation and returns a list of moves.

**Example — Nearest Planet Sniper:**
```python
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return moves

    for mine in my_planets:
        # Find nearest planet we don't own
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        # Send exactly enough ships to capture it
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])
    return moves
```

### Test Locally
```bash
pip install "kaggle-environments>=1.28.0"
```
```python
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 42}, debug=True)
env.run(["main.py", "random"])
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")
env.render(mode="ipython", width=800, height=600)
```

### Submit
```bash
pip install kaggle
# Accept rules at https://www.kaggle.com/competitions/orbit-wars  ("Join Competition")

# Single file agent (main.py with an `agent` function at root):
kaggle competitions submit orbit-wars -f main.py -m "Nearest planet sniper v1"

# Multi-file agent — bundle into tar.gz with main.py at root:
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit orbit-wars -f submission.tar.gz -m "Multi-file agent v1"
```

### Monitor
```bash
kaggle competitions submissions orbit-wars
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0
kaggle competitions leaderboard orbit-wars -s
```

---

## Citation
Bovard Doerschuk-Tiberi, Walter Reade, and Addison Howard. *Orbit Wars.* https://kaggle.com/competitions/orbit-wars, 2026. Kaggle.

## Participation Stats (at extraction)
9,640 Entrants · 4,627 Participants · 4,286 Teams · 8,141 Submissions
Tags: Games · Artificial Intelligence · Reinforcement Learning · Custom Metric
