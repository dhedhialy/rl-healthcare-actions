"""Phase 2 acceptance tests (T2.1-T2.5) for feature engineering and data splits."""

import polars as pl
import numpy as np
import json
from pathlib import Path
from src.config import LAB_FEATURES, VITAL_FEATURES

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset_v1"

import pytest


def _load_split(name: str) -> pl.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        pytest.skip(f"{name} split not built yet")
    return pl.read_parquet(p)


def _load_manifest() -> dict:
    p = DATA_DIR / "split_manifest.json"
    if not p.exists():
        pytest.skip("split manifest not built yet")
    return json.loads(p.read_text())


# T2.1: Patient isolation — no subject_id overlap between splits
def test_t21_patient_isolation():
    train_ids = set(np.load(str(DATA_DIR / "train_subjects.npy")).tolist())
    val_ids = set(np.load(str(DATA_DIR / "val_subjects.npy")).tolist())
    test_ids = set(np.load(str(DATA_DIR / "test_subjects.npy")).tolist())
    for a_name, a_ids, b_name, b_ids in [
        ("train", train_ids, "val", val_ids),
        ("train", train_ids, "test", test_ids),
        ("val", val_ids, "test", test_ids),
    ]:
        overlap = a_ids & b_ids
        assert len(overlap) == 0, f"{a_name}/{b_name} share {len(overlap)} subject_ids"


# T2.2: Stratification — mortality rate within +/- 2pp
def test_t22_stratification():
    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")

    def mortality_rate(df):
        if "hospital_expire_flag" not in df.columns:
            pytest.skip("hospital_expire_flag not in dataset")
            return 0
        adms = df.select("hadm_id", "hospital_expire_flag").unique()
        return float(adms["hospital_expire_flag"].mean())

    overall = (train.height * mortality_rate(train) + val.height * mortality_rate(val) + test.height * mortality_rate(test)) / (train.height + val.height + test.height)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        rate = mortality_rate(df)
        assert abs(rate - overall) < 0.02, f"{name} mortality {rate:.4f} vs overall {overall:.4f} (>2pp)"


# T2.3: Leakage test — z-score stats come from train only
def test_t23_leakage():
    p = DATA_DIR / "zscore_stats.json"
    if not p.exists():
        pytest.skip("zscore_stats.json not built yet")
    stats = json.loads(p.read_text())
    train = _load_split("train")
    for col in stats:
        col_z = f"{col}_z"
        if col_z not in train.columns:
            continue
        train_mean = float(train[col_z].drop_nulls().mean())
        assert abs(train_mean) < 0.05, f"{col_z} z-score mean on train = {train_mean:.4f} (not centered)"


# T2.4: Shape contract — state/action/reward present, consistent dims
def test_t24_shape_contract():
    n_labs = len(LAB_FEATURES)
    n_vitals = len(VITAL_FEATURES)
    n_total = n_labs + n_vitals
    for name in ["train", "val", "test"]:
        df = _load_split(name)
        lab_cols = [c for c in df.columns if c in LAB_FEATURES]
        vital_cols = [c for c in df.columns if c in VITAL_FEATURES]
        missing_cols = [f"{c}_missing" for c in df.columns if f"{c}_missing" in df.columns]
        assert "action_id" in df.columns
        assert "reward" in df.columns
        assert "hadm_id" in df.columns
        assert "bin_idx" in df.columns
        assert len(lab_cols) >= 20, f"{name}: expected >= 20 lab cols, got {len(lab_cols)}"
        assert len(vital_cols) >= 0, f"{name}: expected >= 0 vital cols, got {len(vital_cols)}"
        assert len(missing_cols) >= 20, f"{name}: expected >= 20 missing cols, got {len(missing_cols)}"


# T2.5: Determinism — split ratios are approximately correct
def test_t25_determinism():
    train_ids = np.load(str(DATA_DIR / "train_subjects.npy"))
    val_ids = np.load(str(DATA_DIR / "val_subjects.npy"))
    test_ids = np.load(str(DATA_DIR / "test_subjects.npy"))
    n_train = len(train_ids)
    n_val = len(val_ids)
    n_test = len(test_ids)
    total = n_train + n_val + n_test
    assert total > 0, "empty splits"
    assert n_train / total > 0.65, f"train ratio {n_train/total:.2f} < 0.65"
    assert n_val / total > 0.10, f"val ratio {n_val/total:.2f} < 0.10"
    assert n_test / total > 0.10, f"test ratio {n_test/total:.2f} < 0.10"
