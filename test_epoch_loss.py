#!/usr/bin/env python3
"""Test epoch loss on real data: loads dataset, trains IQL+BC, checks losses are finite and decreasing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from src.rl.dataset import FlatDataset
from src.rl.train import IQL, BehaviorCloning, _epoch_batches
from src.config import N_ACTIONS


def check(name, history):
    print(f"\n=== {name} ===")
    for i, h in enumerate(history):
        items = " | ".join(f"{k}={v:.6f}" for k, v in h.items())
        print(f"  Epoch {i+1}: {items}")
    last = list(history[-1].values())[0]
    first = list(history[0].values())[0]
    all_finite = all(np.isfinite(v) for h in history for v in h.values())
    print(f"  All finite: {'PASS' if all_finite else 'FAIL'}")
    print(f"  Last < first: {'PASS' if last < first else 'FAIL'}")
    return all_finite and last < first


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dataset_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from src.rl.train import auto_device
    device = args.device or auto_device()
    dev = torch.device(device)
    print(f"Device: {device}")

    train_path = Path(args.data_dir) / "train.parquet"
    val_path = Path(args.data_dir) / "val.parquet"
    for p in [train_path, val_path]:
        assert p.exists(), f"{p} not found"

    train_ds = FlatDataset(str(train_path))
    val_ds = FlatDataset(str(val_path))
    train_ds.to_device(dev)
    val_ds.to_device(dev)
    state_dim = train_ds.state_dim
    print(f"Train: {len(train_ds)} transitions | Val: {len(val_ds)} | State dim: {state_dim}")

    print("\n--- IQL ---")
    iql = IQL(state_dim=state_dim, n_actions=N_ACTIONS, device=device)
    iql_hist = []
    for epoch in range(args.epochs):
        iql.q_net.train()
        iql.v_net.train()
        losses = []
        for s, a, r, s2, d in _epoch_batches(train_ds, args.batch_size, shuffle=True):
            losses.append(iql.update_batch(s, a, r, s2, d))
        avg = {k: float(np.mean([float(l[k]) for l in losses])) for k in losses[0]}
        iql_hist.append(avg)
    iql_ok = check("IQL", iql_hist)

    print("\n--- Behavior Cloning ---")
    bc = BehaviorCloning(state_dim=state_dim, n_actions=N_ACTIONS, device=device)
    bc_hist = []
    for epoch in range(args.epochs):
        bc.net.train()
        losses = []
        for s, a, r, s2, d in _epoch_batches(train_ds, args.batch_size, shuffle=True):
            losses.append(bc.update_batch(s, a, r, s2, d))
        avg = {k: float(np.mean([float(l[k]) for l in losses])) for k in losses[0]}
        bc_hist.append(avg)
    bc_ok = check("BC", bc_hist)

    print(f"\n{'='*40}")
    print(f"Overall: {'ALL PASS' if iql_ok and bc_ok else 'SOME FAILURES'}")
