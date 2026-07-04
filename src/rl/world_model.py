import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import polars as pl
from src.rl.dataset import FlatDataset
from src.config import n_actions

DS_DIR = Path("data/dataset_v1")
MODELS_DIR = Path("data/models")

class WorldModel(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim + 1),
        )

    def forward(self, state, action):
        one_hot = torch.zeros(state.size(0), self.net[0].in_features - state.size(1), device=state.device)
        one_hot.scatter_(1, action.unsqueeze(1).long(), 1.0)
        x = torch.cat([state, one_hot], dim=1)
        return self.net(x)

    def predict(self, state, action):
        out = self.forward(state, action)
        next_state_delta = out[:, :-1]
        reward = out[:, -1:]
        return next_state_delta, reward


def train_world_model(device="mps", epochs=20, batch_size=2048):
    train_ds = FlatDataset(str(DS_DIR / "train.parquet"))
    val_ds = FlatDataset(str(DS_DIR / "val.parquet"))
    state_dim = train_ds.states.shape[1]
    n_act = n_actions()

    model = WorldModel(state_dim, n_act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    state_loss_fn = nn.SmoothL1Loss()
    reward_loss_fn = nn.SmoothL1Loss()

    n_train = len(train_ds)
    n_val = len(val_ds)
    best_val_loss = float("inf")

    for ep in range(epochs):
        perm = torch.randperm(n_train)
        model.train()
        total_state_loss = 0.0
        total_reward_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            s = train_ds.states[idx].to(device)
            a = train_ds.actions[idx].to(device)
            s_next = train_ds.next_states[idx].to(device)
            r = train_ds.rewards[idx].to(device)

            pred_delta, pred_r = model.predict(s, a)
            s_loss = state_loss_fn(pred_delta.contiguous(), (s_next - s).contiguous())
            r_loss = reward_loss_fn(pred_r.squeeze().contiguous(), r.contiguous())
            loss = s_loss + r_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_state_loss += s_loss.item()
            total_reward_loss += r_loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_perm = torch.randperm(n_val)[:min(50000, n_val)]
            s = val_ds.states[val_perm].to(device)
            a = val_ds.actions[val_perm].to(device)
            s_next = val_ds.next_states[val_perm].to(device)
            r = val_ds.rewards[val_perm].to(device)
            pred_delta, pred_r = model.predict(s, a)
            val_s_loss = state_loss_fn(pred_delta.contiguous(), (s_next - s).contiguous()).item()
            val_r_loss = reward_loss_fn(pred_r.squeeze().contiguous(), r.contiguous()).item()

        val_loss = val_s_loss + val_r_loss
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep + 1}/{epochs} | state_loss={total_state_loss / n_batches:.4f} reward_loss={total_reward_loss / n_batches:.4f} | val_s={val_s_loss:.4f} val_r={val_r_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "world_model.pt")

    print(f"  Best val loss: {best_val_loss:.4f}")


def load_world_model(device="mps"):
    path = MODELS_DIR / "world_model.pt"
    if not path.exists():
        return None
    # Infer state_dim from a checkpoint
    from src.rl.evaluate import auto_device
    ckpt = torch.load(path, map_location="cpu")
    w_key = [k for k in ckpt.keys() if "net.0.weight" in k][0]
    in_features = ckpt[w_key].shape[1]
    state_dim = in_features - n_actions()
    model = WorldModel(state_dim, n_actions()).to(device)
    model.load_state_dict(ckpt)
    return model


def counterfactual(model, state, action_a, action_b, device="mps"):
    model.eval()
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        a_a = torch.LongTensor([action_a]).to(device)
        a_b = torch.LongTensor([action_b]).to(device)
        delta_a, r_a = model.predict(s, a_a)
        delta_b, r_b = model.predict(s, a_b)
        next_a = s + delta_a
        next_b = s + delta_b
    return {
        "action_a": int(action_a),
        "action_b": int(action_b),
        "reward_a": float(r_a.squeeze().cpu().item()),
        "reward_b": float(r_b.squeeze().cpu().item()),
        "delta": float((r_a - r_b).squeeze().cpu().item()),
        "next_state_a": next_a.squeeze().cpu().numpy().tolist(),
        "next_state_b": next_b.squeeze().cpu().numpy().tolist(),
    }


class WorldModelEnsemble:
    def __init__(self, n_ensembles=5):
        self.models = []
        for i in range(n_ensembles):
            m = load_world_model()
            if m:
                self.models.append(m)

    def predict(self, state, action):
        rewards = []
        for m in self.models:
            _, r = m.predict(state, action)
            rewards.append(r.item())
        return np.mean(rewards), np.std(rewards)
