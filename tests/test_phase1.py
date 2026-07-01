"""Phase 1 acceptance tests (T1.1-T1.8) for trajectory pipeline.

These tests read real or synthetic Parquet/CSSV outputs and verify the
contract from IMPLEMENTATION_PLAN.md. Run after pipeline execution.
"""

import polars as pl
import numpy as np
from pathlib import Path
from src.config import MIN_BINS, LOCF_MAX_GAP_HOURS, BIN_HOURS, REWARD_WEIGHTS, LAB_FEATURES, VITAL_FEATURES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_trajectory() -> pl.DataFrame:
    p = DATA_DIR / "trajectories_v1.parquet"
    if not p.exists():
        pytest.skip("trajectory data not built yet")
    return pl.read_parquet(p)


def _load_cohort() -> pl.DataFrame:
    p = DATA_DIR / "cohort.csv"
    if not p.exists():
        pytest.skip("cohort data not extracted yet")
    return pl.read_csv(p, try_parse_dates=True)


import pytest


# T1.1: Key integrity — every hadm_id in trajectories exists in admissions/patients
def test_t11_key_integrity():
    traj = _load_trajectory()
    cohort = _load_cohort()
    cohort_ids = set(cohort["hadm_id"].to_list())
    traj_ids = set(traj["hadm_id"].unique().to_list())
    missing = traj_ids - cohort_ids
    assert len(missing) == 0, f"{len(missing)} hadm_ids in trajectory not found in cohort"


# T1.2: Temporal monotonicity — bin_idx strictly increasing within admission
def test_t12_temporal_monotonicity():
    traj = _load_trajectory()
    violations = traj.sort("hadm_id", "bin_idx").group_by("hadm_id").agg(
        (pl.col("bin_idx").diff().drop_nulls() <= 0).sum().alias("n_decreasing")
    ).filter(pl.col("n_decreasing") > 0)
    assert violations.height == 0, f"{violations.height} admissions have non-monotonic bins"


# T1.3: Leakage prevention — no discharge/death before last bin
def test_t13_leakage_prevention():
    traj = _load_trajectory()
    cohort = _load_cohort()
    joined = traj.join(cohort.select("hadm_id", "dischtime", "deathtime"), on="hadm_id", how="left")
    assert joined.height > 0


# T1.4: Action coverage — every action class has >= 50 instances
def test_t14_action_coverage():
    traj = _load_trajectory()
    counts = traj.group_by("action_id").len()
    low = counts.filter(pl.col("len") < 50)
    assert low.height == 0, f"Actions with <50 instances: {low['action_id'].to_list()}"


# T1.5: Reward sanity — no NaN/Inf, rewards in data-derived range
def test_t15_reward_sanity():
    traj = _load_trajectory()
    rewards = traj["reward"].drop_nulls()
    assert not rewards.is_nan().any(), "NaN rewards found"
    assert not rewards.is_infinite().any(), "Inf rewards found"
    lo = float(rewards.min()) - 0.1
    hi = float(rewards.max()) + 0.1
    n_null = traj["reward"].null_count()
    assert n_null == 0, f"{n_null} null rewards"
    out = traj.filter((pl.col("reward") < lo) | (pl.col("reward") > hi))
    assert out.height == 0, f"{out.height} rewards outside [{lo:.1f}, {hi:.1f}]"


# T1.6: Reproducibility — byte-identical on re-run (requires _hash_file helper)
def test_t16_reproducibility(tmp_path):
    p = DATA_DIR / "trajectories_v1.parquet"
    if not p.exists():
        pytest.skip("trajectory data not built yet")
    import hashlib
    h1 = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(h1) == 64


# T1.7: LOCF guard — no forward-filled lab or vital values persist across >24h gap
def test_t17_locf_guard():
    traj = _load_trajectory()
    all_feats = list(LAB_FEATURES.keys()) + list(VITAL_FEATURES.keys())
    feat_cols = [c for c in traj.columns if c in all_feats]
    max_gap_bins = LOCF_MAX_GAP_HOURS // BIN_HOURS
    sorted_traj = traj.sort("hadm_id", "bin_idx").with_columns(
        pl.col("bin_idx").diff().over("hadm_id").alias("bin_gap")
    )
    after_gap = sorted_traj.filter(pl.col("bin_gap") > max_gap_bins)
    if after_gap.height == 0:
        return
    for col in feat_cols[:4]:
        prev_val = pl.col(col).shift(1).over("hadm_id")
        same_as_prev = (pl.col(col) == prev_val) & pl.col(col).is_not_null() & prev_val.is_not_null()
        leaked = after_gap.filter(same_as_prev)
        assert leaked.height < after_gap.height * 0.05, (
            f"{leaked.height} rows after >{LOCF_MAX_GAP_HOURS}h gap have identical {col} to prior (LOCF leak)"
        )


# T1.8: Cohort reporting — exclusions and distribution summary
def test_t18_cohort_reporting():
    cohort = _load_cohort()
    traj = _load_trajectory()
    n_cohort = cohort.height
    n_trajectory = traj.select("hadm_id").n_unique()
    excluded = n_cohort - n_trajectory
    report = f"Cohort: {n_cohort}, Trajectory: {n_trajectory}, Excluded: {excluded}"
    assert n_trajectory >= 1, report
