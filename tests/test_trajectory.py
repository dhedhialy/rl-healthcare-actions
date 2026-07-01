from src.pipeline.trajectory import build_trajectories


def test_reward_bounds_synthetic(tmp_path):
    """Self-check: synthetic 2-admission data produces rewards in expected range."""
    import polars as pl

    cohort = pl.DataFrame({
        "hadm_id": [1, 2],
        "subject_id": [100, 200],
        "admittime": ["2024-01-01 08:00", "2024-01-02 10:00"],
        "dischtime": ["2024-01-03 08:00", "2024-01-04 10:00"],
        "deathtime": [None, "2024-01-04 10:00"],
        "hospital_expire_flag": [0, 1],
        "los_days": [2.0, 2.0],
    })
    cohort_csv = tmp_path / "cohort.csv"
    cohort.write_csv(cohort_csv)

    labs = pl.DataFrame({
        "hadm_id": [1, 1, 2, 2] * 3,
        "label": ["hemoglobin"] * 4 + ["wbc"] * 4 + ["platelets"] * 4,
        "valuenum": [10.0, 9.5, 8.0, 7.5, 5.0, 6.0, 12.0, 15.0, 200.0, 180.0, 50.0, 30.0],
        "charttime": [
            "2024-01-01 08:00", "2024-01-01 12:00",
            "2024-01-02 10:00", "2024-01-02 14:00",
        ] * 3,
        "bin_idx": [0, 1, 0, 1] * 3,
    })
    labs_path = tmp_path / "labs.parquet"
    labs.write_parquet(labs_path)

    actions = pl.DataFrame({
        "hadm_id": [1, 2],
        "bin_idx": [0, 0],
        "action_id": [0, 1],
    })
    actions_path = tmp_path / "actions.parquet"
    actions.write_parquet(actions_path)

    out = tmp_path / "trajectories.parquet"
    result = build_trajectories(str(cohort_csv), str(labs_path), str(actions_path), str(out))
    assert result.height > 0
    assert not result["reward"].is_nan().any()
