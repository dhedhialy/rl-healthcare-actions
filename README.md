# RL Healthcare Actions

Reinforcement Learning for Healthcare Actions: training AI models on clinician actions to predict the best next steps for patient care, optimizing treatment interventions and improving clinical outcomes through systematic evaluations.

## Summary

This system learns from 1.3 million real clinical decisions in MIMIC-IV to recommend intervention bundles for hematology and anemia patients. For each 4-hour window, the model takes a patient's current labs, vitals, and demographics — then outputs a probability-ranked list of the best next interventions, filtered by hard safety constraints.

**Offline only.** No live deployment. Trained on historical data, evaluated through off-policy estimation.

## What it does

Given a patient's current state, the model recommends one of 16 intervention bundles:

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | No intervention (watch) | 8 | Anticoagulant hold |
| 1 | RBC transfusion | 9 | Vasopressor |
| 2 | Platelet transfusion | 10 | Antibiotic |
| 3 | FFP / cryoprecipitate | 11 | Insulin |
| 4 | IV iron | 12 | Diuretic |
| 5 | ESA (erythropoietin) | 13 | Steroid |
| 6 | Fluid resuscitation | 14 | Sedation / analgesia |
| 7 | Electrolyte correction | 15 | Cardiac medication |

Safety constraints (S1-S7) are hard-coded: no ESA with cancer, no platelets if Plt >= 50K, no FFP if INR <= 2.0, no vasopressor if MAP >= 65, no insulin if glucose < 70, no diuretic if Cr > 4 + hypotension.

## Results

| Metric | IQL | Observed Clinician Behavior | Behavior Cloning |
|--------|-----|----------------------------|------------------|
| Mean return | **-0.38** | -131.96 | -97.95 |
| 95% CI | [-0.41, -0.30] | [-136.22, -127.52] | — |

- **4 independent OPE estimators** (WIS, FQE, DM, Policy Value) agree IQL > behavior
- **Non-overlapping bootstrap CIs** (1000 resamples)
- **Zero safety violations** post-masking (including adversarial test cases)
- **Phenotype equity**: IQL outperforms behavior in all 20 ICD subgroups

## Quick start

```bash
# Unified CLI (all operations)
python3 cli.py --help

# Run full pipeline
python3 cli.py pipeline

# Feature engineering + splits
python3 cli.py features

# Train with different architectures
python3 cli.py train --arch mlp          # 2-layer MLP (default)
python3 cli.py train --arch deep         # 512->256->128
python3 cli.py train --arch wide         # 512->512
python3 cli.py train --arch lstm         # LSTM encoder
python3 cli.py train --hidden-sizes 256 128 64   # custom depth

# Evaluate
python3 cli.py eval

# Batch inference on millions of states
python3 cli.py infer --input states.parquet --output results.parquet

# Start REST API server
python3 cli.py serve --preload

# Run tests (28 pass, 1 skipped — T4.5 needs clinicians)
python3 cli.py test
python3 cli.py test -v --filter phase3
```

## REST API

Start: `python3 cli.py serve --preload` or `python3 -m src.rl.server --port 8000 --preload`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, model loaded status |
| `/actions` | GET | Available action bundles |
| `/load` | POST | Load model checkpoints |
| `/predict` | POST | Single patient → recommendations + 95% CIs |
| `/predict_batch` | POST | Batch prediction |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"state": [0.0, 0.1, ... 64 floats]}'
```

## Architecture options

| Architecture | Hidden layers | Params | Use case |
|-------------|---------------|--------|----------|
| `mlp` (default) | 256 → 256 | ~200K | Balanced general purpose |
| `deep` | 512 → 256 → 128 | ~400K | Complex state interactions |
| `wide` | 512 → 512 | ~400K | High-capacity, more data needed |
| `lstm` | LSTM(256) → 256 | ~350K | Sequential trajectory modeling |

Custom: `--hidden-sizes 128 128 64 32`

## Schema-agnostic data pipeline

The pipeline works with any EHR by setting column mappings. Defaults match MIMIC-IV:

```bash
# Full schema override via JSON file
export RL_SCHEMA=/path/to/schema.json

# Or individual env var overrides
export RL_COL_PATIENT_ID=subject_id
export RL_COL_ADMISSION_ID=hadm_id
export RL_COL_CHART_TIME=charttime

# Data directories
export MIMIC_DATA_DIR=/path/to/csv
export RL_DATA_DIR=/path/to/output

python3 cli.py pipeline
```

## Per-patient confidence

Inference uses a 5-seed ensemble. For each action, the system outputs:
- **Q-value**: mean across ensemble members
- **95% CI**: 2.5/97.5 percentile interval (bootstrap)
- **Confidence**: 1 − CI_width / |Q| (0=uncertain, 1=certain)
- **Policy probability**: softmax of advantage

## Key files

| File | Purpose |
|------|---------|
| `src/config.py` | Lab itemids, action definitions, reward weights, safety constraints |
| `src/schema.py` | Schema-agnostic column mapping (override via $RL_SCHEMA) |
| `src/cohort/extract.py` | ICD + Hgb threshold cohort extraction |
| `src/extract/labs.py` | Lab extraction and 4-hour binning |
| `src/extract/actions.py` | Action extraction with precedence |
| `src/pipeline/trajectory.py` | LOCF, missingness, rewards, trajectory assembly |
| `src/pipeline/features.py` | Z-scores, time encoding, trend deltas, splits, batch_transform() |
| `src/rl/train.py` | IQL + BC training, OPE, multiple architectures |
| `src/rl/evaluate.py` | Safety audit, phenotype stratification, bootstrap CIs |
| `src/rl/inference.py` | Ensemble inference with per-patient bootstrap CIs |
| `src/rl/server.py` | FastAPI REST server |
| `cli.py` | Unified CLI entry point (9 subcommands) |

## Caveats

- **No live deployment.** Offline RL on historical data only.
- **No clinician review yet** (T4.5 skipped — no attendings available).
- **RBC transfusion is inferred** from Hgb jumps (no inputevents table).
- **Reward is synthetic.** Composite (survival, LOS, lab deviation) approximates clinical utility.
- **Action space is coarse.** 16 bundles, not individual dose decisions.
