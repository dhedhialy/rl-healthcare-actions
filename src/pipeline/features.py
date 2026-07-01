"""Phase 2: Feature engineering, data splits, and behavior policy logging."""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from src.config import LAB_FEATURES, VITAL_FEATURES, N_ACTIONS, BIN_HOURS

LAB_COLS = list(LAB_FEATURES.keys())
VITAL_COLS = list(VITAL_FEATURES.keys())
ALL_FEAT_COLS = LAB_COLS + VITAL_COLS
MISSING_COLS = [f"{c}_missing" for c in ALL_FEAT_COLS]


def join_demographics(traj: pl.DataFrame, cohort: pl.DataFrame) -> pl.DataFrame:
    return traj.join(
        cohort.select("hadm_id", "subject_id", "anchor_age", "gender"),
        on="hadm_id",
        how="left",
    )


def zscore_features(traj: pl.DataFrame, train_ids: set) -> Tuple[pl.DataFrame, Dict]:
    """Z-score both lab and vital features using train-set statistics only."""
    train = traj.filter(pl.col("subject_id").is_in(list(train_ids)))
    stats = {}
    for c in ALL_FEAT_COLS:
        if c not in train.columns:
            continue
        s = train[c]
        m = float(s.drop_nulls().mean())
        sd = float(s.drop_nulls().std())
        if sd == 0:
            sd = 1.0
        stats[c] = {"mean": m, "std": sd}
    exprs = []
    for c in ALL_FEAT_COLS:
        if c not in stats:
            continue
        exprs.append(
            ((pl.col(c) - stats[c]["mean"]) / stats[c]["std"]).alias(f"{c}_z")
        )
    return traj.with_columns(exprs), stats


def encode_time(traj: pl.DataFrame) -> pl.DataFrame:
    hours = pl.col("bin_idx") * BIN_HOURS
    return traj.with_columns(
        ((2 * np.pi * hours / 168.0).sin()).alias("time_sin"),
        ((2 * np.pi * hours / 168.0).cos()).alias("time_cos"),
    )


def compute_trend_deltas(traj: pl.DataFrame, n: int = 3) -> pl.DataFrame:
    """Compute trend deltas for key labs and vitals."""
    key_features = [
        "hemoglobin", "platelets", "pt", "inr",
        "heart_rate", "mean_bp", "spo2", "creatinine", "glucose",
    ]
    for feat in key_features:
        col = f"{feat}_z" if f"{feat}_z" in traj.columns else feat
        if col not in traj.columns:
            continue
        for lag in range(1, n + 1):
            delta = pl.col(col) - pl.col(col).shift(lag).over("hadm_id")
            traj = traj.with_columns(delta.alias(f"delta_{feat}_{lag}"))
    return traj


def build_state_vector(traj: pl.DataFrame) -> Tuple[pl.DataFrame, list]:
    z_cols = [f"{c}_z" for c in ALL_FEAT_COLS if f"{c}_z" in traj.columns]
    demo = ["anchor_age"]
    time_enc = ["time_sin", "time_cos"]
    gender_col = ["gender_male"] if "gender_male" in traj.columns else []
    delta_cols = [c for c in traj.columns if c.startswith("delta_")]
    state_cols = z_cols + MISSING_COLS + demo + time_enc + gender_col + delta_cols
    return traj, state_cols


def split_by_subject(
    traj: pl.DataFrame, seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict]:
    subj_mort = traj.select("subject_id", "hospital_expire_flag").group_by("subject_id").agg(
        pl.col("hospital_expire_flag").max().alias("mortality")
    )
    died = subj_mort.filter(pl.col("mortality") == 1).select("subject_id")
    lived = subj_mort.filter(pl.col("mortality") == 0).select("subject_id")
    rng = np.random.RandomState(seed)

    def _split(ids_df):
        ids = ids_df["subject_id"].to_list()
        idx = rng.permutation(len(ids))
        n = len(ids)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        return (
            set(np.array(ids)[idx[:n_train]].tolist()),
            set(np.array(ids)[idx[n_train:n_train + n_val]].tolist()),
            set(np.array(ids)[idx[n_train + n_val:]].tolist()),
        )

    train_d, val_d, test_d = _split(died)
    train_l, val_l, test_l = _split(lived)
    train_ids = train_d | train_l
    val_ids = val_d | val_l
    test_ids = test_d | test_l

    splits = {
        "train": traj.filter(pl.col("subject_id").is_in(list(train_ids))),
        "val": traj.filter(pl.col("subject_id").is_in(list(val_ids))),
        "test": traj.filter(pl.col("subject_id").is_in(list(test_ids))),
    }
    metadata = {"train_ids": sorted(train_ids), "val_ids": sorted(val_ids), "test_ids": sorted(test_ids)}
    return splits["train"], splits["val"], splits["test"], metadata


def compute_behavior_policy(traj: pl.DataFrame) -> pl.DataFrame:
    """Compute behavior policy over N_ACTIONS categories (0 through N_ACTIONS-1)."""
    counts = traj.group_by("action_id").agg(pl.len().alias("count"))
    total = traj.height
    # Ensure all action IDs 0..N_ACTIONS-1 are present
    all_actions = pl.DataFrame({"action_id": list(range(N_ACTIONS))})
    counts = all_actions.join(counts, on="action_id", how="left").with_columns(
        pl.col("count").fill_null(0)
    )
    return counts.with_columns(
        (pl.col("count") / total).alias("pi_beta")
    ).sort("action_id")


def compute_behavior_policy_binned(traj: pl.DataFrame, n_bins: int = 10) -> pl.DataFrame:
    traj = traj.with_columns(
        (pl.col("hemoglobin") * n_bins / 20.0).floor().cast(pl.Int32).alias("hgb_bin")
    )
    total = traj.group_by("hgb_bin", "action_id").agg(pl.len().alias("count"))
    totals = total.group_by("hgb_bin").agg(pl.col("count").sum().alias("total"))
    return total.join(totals, on="hgb_bin").with_columns(
        (pl.col("count") / pl.col("total")).alias("pi_beta")
    ).sort("hgb_bin", "action_id")


def build_dataset(
    traj_path: str = "data/trajectories_v1.parquet",
    cohort_path: str = "data/cohort.csv",
    out_dir: str = "data/dataset_v1",
    seed: int = 42,
) -> Dict:
    from src.pipeline.trajectory import load_cohort

    traj = pl.read_parquet(traj_path)
    cohort = load_cohort(cohort_path)

    traj = join_demographics(traj, cohort)
    traj = traj.with_columns(
        (pl.col("gender") == "M").cast(pl.Int8).alias("gender_male")
    )
    traj = traj.join(
        cohort.select("hadm_id", "hospital_expire_flag"), on="hadm_id", how="left"
    )

    _, _, _, split_meta = split_by_subject(traj, seed)
    train_ids = split_meta["train_ids"]

    traj, zstats = zscore_features(traj, set(train_ids))
    traj = encode_time(traj)
    traj = compute_trend_deltas(traj)

    train, val, test, split_meta = split_by_subject(traj, seed)

    pi_beta = compute_behavior_policy(train)
    pi_beta_binned = compute_behavior_policy_binned(train)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train.write_parquet(str(out / "train.parquet"))
    val.write_parquet(str(out / "val.parquet"))
    test.write_parquet(str(out / "test.parquet"))
    pi_beta.write_csv(str(out / "behavior_policy.csv"))
    pi_beta_binned.write_parquet(str(out / "behavior_policy_binned.parquet"))

    import json
    with open(out / "zscore_stats.json", "w") as f:
        json.dump(zstats, f, indent=2)
    split_counts = {k: len(v) for k, v in split_meta.items()}
    with open(out / "split_manifest.json", "w") as f:
        json.dump(split_counts, f, indent=2)
    np.save(str(out / "train_subjects.npy"), np.array(sorted(split_meta["train_ids"])))
    np.save(str(out / "val_subjects.npy"), np.array(sorted(split_meta["val_ids"])))
    np.save(str(out / "test_subjects.npy"), np.array(sorted(split_meta["test_ids"])))

    _, state_cols = build_state_vector(train)

    return {
        "train": train.height,
        "val": val.height,
        "test": test.height,
        "train_subjects": len(split_meta["train_ids"]),
        "val_subjects": len(split_meta["val_ids"]),
        "test_subjects": len(split_meta["test_ids"]),
        "state_dim": len(state_cols),
        "zstats": zstats,
    }


def batch_transform(
    input_path: str,
    output_path: str,
    transform_fn,
    chunk_size: int = 100_000,
    **kwargs,
):
    """Process a large parquet file in chunks with a transform function.

    transform_fn(df_chunk: pl.DataFrame, **kwargs) -> pl.DataFrame
    """
    import polars as pl
    from pathlib import Path

    src = pl.scan_parquet(input_path)
    n_rows = src.select(pl.len()).collect().item()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = True
    for offset in range(0, n_rows, chunk_size):
        chunk = src.slice(offset, chunk_size).collect()
        transformed = transform_fn(chunk, **kwargs)
        if first:
            transformed.write_parquet(str(out_path))
            first = False
        else:
            transformed.write_parquet(str(out_path), append=True)
        print(f"  batch_transform: {min(offset+chunk_size, n_rows)}/{n_rows}", flush=True)


if __name__ == "__main__":
    result = build_dataset()
    for k, v in result.items():
        if k != "zstats":
            print(f"  {k}: {v}")
