"""Schema-agnostic column mapping: adapt any EHR to the RL pipeline.

Override with $RL_SCHEMA pointing to a JSON file with the same structure,
or set individual env vars (RL_COL_PATIENT_ID, RL_COL_ADMISSION_ID, …).

Example schema.json:
{
  "patient_id": "subject_id",
  "admission_id": "hadm_id",
  "admit_time": "admittime",
  "discharge_time": "dischtime",
  "death_time": "deathtime",
  "gender": "gender",
  "age": "anchor_age",
  "expire_flag": "hospital_expire_flag",
  "chart_time": "charttime",
  "item_id": "itemid",
  "value_num": "valuenum",
  "drug_name": "drug",
  "start_time": "starttime",
  "icd_code": "icd_code",
  "icd_version": "icd_version",
  "seq_num": "seq_num",
  "los_days": "los_days",
  "bin_idx": "bin_idx",
  "action_id": "action_id",
  "reward": "reward"
}
"""

import json
import os
from typing import Dict, Optional

DEFAULT_SCHEMA = {
    "patient_id": "subject_id",
    "admission_id": "hadm_id",
    "admit_time": "admittime",
    "discharge_time": "dischtime",
    "death_time": "deathtime",
    "gender": "gender",
    "age": "anchor_age",
    "expire_flag": "hospital_expire_flag",
    "chart_time": "charttime",
    "item_id": "itemid",
    "value_num": "valuenum",
    "drug_name": "drug",
    "start_time": "starttime",
    "icd_code": "icd_code",
    "icd_version": "icd_version",
    "seq_num": "seq_num",
    "los_days": "los_days",
    "bin_idx": "bin_idx",
    "action_id": "action_id",
    "reward": "reward",
    "lab_label": "label",
    "gender_male_expr": "== 'M'",
    "subject_split_col": "subject_id",
}


def _load_schema() -> Dict[str, str]:
    path = os.environ.get("RL_SCHEMA")
    if path:
        with open(path) as f:
            return json.load(f)
    overrides = {}
    for key in DEFAULT_SCHEMA:
        env_val = os.environ.get(f"RL_COL_{key.upper()}")
        if env_val is not None:
            overrides[key] = env_val
    if overrides:
        base = dict(DEFAULT_SCHEMA)
        base.update(overrides)
        return base
    return dict(DEFAULT_SCHEMA)


_SCHEMA_CACHE: Optional[Dict[str, str]] = None


def get_schema() -> Dict[str, str]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = _load_schema()
    return _SCHEMA_CACHE


def col(name: str) -> str:
    return get_schema().get(name, name)
