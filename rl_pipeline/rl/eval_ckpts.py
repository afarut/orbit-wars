"""Скоринг спреда чекпоинтов rl6b vs Rust-PL (1v1 + FFA), argmax, на сервере (A100).
Выбор финального чекпоинта по реальной цели турнира, а не по шумной opp_wr из лога.
"""
import sys, glob, re
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0, ".")
import ow_rs
from model import PolicyValueNet, ModelConfig
from core.features import FeatureConfig
from rl.encode_bridge import batch_encode_rust, decode_batch

MCFG = ModelConfig(d_model=128, d_k=64, n_layers=2, n_heads=4, ffn=512,
                   dropout=0.0, enc_hidden=128, head_hidden=128)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(p):
    ck = torch.load(p, map_location=DEV, weights_only=False)
    m = PolicyValueNet(MCFG).to(DEV)
    m.load_state_dict(ck.get("model_state", ck.get("model")))
    m.eval()
    return m


@torch.no_grad()
def _moves(model, env, player):
    bobs, meta = batch_encode_rust(env, player, DEV)
    out = model(bobs)
    moves = decode_batch(out, meta, player=player, deterministic=True)[0]
    return moves


def winrate(model, n_agents, N=256, target=1200):
    env = ow_rs.VecEnv(N, n_agents)
    env.reset(list(range(N)))
    env.set_opponents([ow_rs.AGENT_PRODUCER_LITE] * (N * (n_agents - 1)))
    W = E = 0
    while E < target:
        rew, done = env.step_p0_ids(_moves(model, env, 0))
        dp = done.astype(bool)
        if dp.any():
            W += int((rew[dp, 0] > 0).sum())
            E += int(dp.sum())
    return W / E * 100, E


def main():
    steps = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else None
    allck = {}
    for p in glob.glob("checkpoints/rl6b/rl_*.pt"):
        mm = re.search(r"rl_(\d+)\.pt", p)
        if mm:
            allck[int(mm.group(1))] = p
    if steps:
        cks = [(s, allck[min(allck, key=lambda x: abs(x - s))]) for s in steps]
    else:
        cks = sorted(allck.items())
    print(f"device={DEV}  кандидатов={len(cks)}\n")
    print(f"{'step':>11} {'vs RustPL 1v1':>14} {'FFA vs 3xPL':>13}")
    rows = []
    for step, path in cks:
        w1, e1 = winrate(load(path), 2, target=1500)
        wf, ef = winrate(load(path), 4, target=1000)
        rows.append((step, w1, wf))
        print(f"{step:>11} {w1:>11.1f}% {wf:>11.1f}%", flush=True)
    print("\n— по vs Rust-PL 1v1 —")
    for s, w1, wf in sorted(rows, key=lambda r: -r[1])[:3]:
        print(f"  step {s}: 1v1={w1:.1f}%  ffa={wf:.1f}%")
    print("— по сумме 1v1+FFA —")
    for s, w1, wf in sorted(rows, key=lambda r: -(r[1] + r[2]))[:3]:
        print(f"  step {s}: 1v1={w1:.1f}%  ffa={wf:.1f}%  sum={w1+wf:.1f}")


if __name__ == "__main__":
    main()
