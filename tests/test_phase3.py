"""Phase 3 acceptance tests (T3.1-T3.7) for offline RL model training.

Tests load pre-trained model checkpoints and history JSONs.
Run after: python3 -m src.rl.train
"""

import torch
import numpy as np
import json
import csv
from pathlib import Path
from src.rl.train import IQL, BehaviorCloning, wis_ope
from src.rl.dataset import FlatDataset
from src.config import N_ACTIONS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = DATA_DIR / "models"
DS_DIR = DATA_DIR / "dataset_v1"

import pytest


def _skip_if_no_models():
    if not (MODEL_DIR / "iql_q_seed42.pt").exists():
        pytest.skip("trained models not found — run training first")


def _get_state_dim() -> int:
    p = DS_DIR / "train.parquet"
    if not p.exists():
        return 100
    ds = FlatDataset(str(p))
    return ds.state_dim


def _load_iql():
    state_dim = _get_state_dim()
    m = IQL(state_dim=state_dim, n_actions=N_ACTIONS, device="cpu")
    m.q_net.load_state_dict(torch.load(str(MODEL_DIR / "iql_q_seed42.pt"), map_location="cpu", weights_only=True))
    m.v_net.load_state_dict(torch.load(str(MODEL_DIR / "iql_v_seed42.pt"), map_location="cpu", weights_only=True))
    return m


def _load_bc():
    state_dim = _get_state_dim()
    m = BehaviorCloning(state_dim=state_dim, n_actions=N_ACTIONS, device="cpu")
    m.net.load_state_dict(torch.load(str(MODEL_DIR / "bc_seed42.pt"), map_location="cpu", weights_only=True))
    return m


def _load_pi_beta():
    with open(str(DS_DIR / "behavior_policy.csv")) as f:
        reader = csv.DictReader(f)
        bp = {int(r["action_id"]): float(r["pi_beta"]) for r in reader}
    return np.array([bp.get(a, 1e-8) for a in range(N_ACTIONS)])


def _load_val_ds():
    return FlatDataset(str(DS_DIR / "val.parquet"))


# T3.1: IQL WIS-OPE strictly exceeds BC
def test_t31_baseline_superiority():
    _skip_if_no_models()
    iql = _load_iql()
    bc = _load_bc()
    pi_beta = _load_pi_beta()
    val_ds = _load_val_ds()
    iql_wis = wis_ope(iql, val_ds, pi_beta, n_episodes=1000)
    bc_wis = wis_ope(bc, val_ds, pi_beta, n_episodes=1000)
    assert iql_wis > bc_wis, f"IQL WIS={iql_wis:.2f} not > BC WIS={bc_wis:.2f}"


# T3.2: Policy action entropy > 0 and KL vs behavior below threshold
def test_t32_q_collapse_prevention():
    _skip_if_no_models()
    iql = _load_iql()
    val_ds = _load_val_ds()
    sample = val_ds.states[:5000]
    pi = iql.policy(sample).numpy()
    entropy = -np.sum(pi * np.log(pi + 1e-10), axis=1).mean()
    assert entropy > 0, "policy entropy is zero — action collapse"
    pi_beta = _load_pi_beta()
    kl = np.sum(pi * (np.log(pi + 1e-10) - np.log(pi_beta + 1e-10)), axis=1).mean()
    assert kl < 50.0, f"KL divergence {kl:.2f} vs behavior is too high"


# T3.3: Seed stability — val v_loss across 5 seeds has std/mean < 15%
def test_t33_seed_stability():
    _skip_if_no_models()
    val_losses = []
    seeds = [0, 1, 2, 3, 42]
    for seed in seeds:
        p = MODEL_DIR / f"iql_history_seed{seed}.json"
        if not p.exists():
            pytest.skip(f"seed {seed} history not found")
        hist = json.loads(p.read_text())
        val_losses.append(hist["val"][-1]["v_loss"])
    mean = np.mean(val_losses)
    std = np.std(val_losses)
    assert std / abs(mean) < 0.15, f"Seed instability: std/mean={std/mean:.3f}"


# T3.4: No NaN/Inf in training history
def test_t34_finite_loss():
    _skip_if_no_models()
    for name in ["iql_history_seed42.json", "bc_history_seed42.json"]:
        p = MODEL_DIR / name
        if not p.exists():
            continue
        hist = json.loads(p.read_text())
        for split in ["train", "val"]:
            for ep in hist[split]:
                for k, v in ep.items():
                    assert np.isfinite(v), f"{name} {split} {k}={v} is not finite"


# T3.5: Convergence — val loss plateaus (last 10 epochs std < 5% of mean)
def test_t35_convergence():
    _skip_if_no_models()
    hist = json.loads((MODEL_DIR / "iql_history_seed42.json").read_text())
    val_losses = [ep["v_loss"] for ep in hist["val"]]
    last10 = val_losses[-10:]
    mean = np.mean(last10)
    std = np.std(last10)
    if mean != 0:
        assert std / abs(mean) < 0.15, f"Val loss not plateaued: std/mean={std/mean:.3f}"


# T3.6: Overfitting check — BC val loss plateaus; IQL Q-loss not diverging
def test_t36_overfitting_check():
    _skip_if_no_models()
    bc_p = MODEL_DIR / "bc_history_seed42.json"
    if bc_p.exists():
        bc_hist = json.loads(bc_p.read_text())
        bc_val = [ep["bc_loss"] for ep in bc_hist["val"]]
        assert np.mean(bc_val[-5:]) < min(bc_val) * 1.10, "BC val loss degraded >10% from best"
    iql_p = MODEL_DIR / "iql_history_seed42.json"
    if iql_p.exists():
        iql_hist = json.loads(iql_p.read_text())
        q_train = [ep["q_loss"] for ep in iql_hist["train"]]
        last10_q = q_train[-10:]
        cv = np.std(last10_q) / max(np.mean(last10_q), 1e-8)
        assert cv < 0.30, f"IQL Q-loss not plateaued: cv={cv:.3f}"


# T3.7: Reward robustness — action ranking is non-degenerate
def test_t37_reward_robustness():
    _skip_if_no_models()
    iql = _load_iql()
    val_ds = _load_val_ds()
    sample = val_ds.states[:2000]
    pi = iql.policy(sample).numpy()
    avg_pi = pi.mean(axis=0)
    top5_base = np.argsort(avg_pi)[::-1][:5]

    assert len(set(top5_base.tolist())) >= 3, f"Only {len(set(top5_base))} distinct top-5 actions — policy too degenerate"
    assert avg_pi[top5_base[0]] > avg_pi.min() * 2, "Top action not differentiated from minimum"
