# Reinforcement Learning for Healthcare Actions: Implementation Plan

## Phase 0: Problem Formulation & MDP Specification (Week 1)

Before any code is written, the Markov Decision Process (MDP) must be strictly defined. An incorrect formulation at this stage will result in wasted engineering efforts later.

- **0.1 Clinical Scope:** Confine the first version to the hematology/anemia/obstetric-bleeding phenotype currently covered by the data (defined as "anemia and obstetric complications"). The scope must be concrete, narrow, and measurable. Do not attempt to encompass "all clinical decisions."
- **0.2 State Space (s_t):** Defined by 25 lab biomarkers, a phenotype embedding derived from `diagnoses_text` and the 359-cluster assignment, patient demographics (age/sex from MIMIC), and time-since-admission. The observation window must be strictly defined (e.g., 4-hour bins).
- **0.3 Action Space (a_t):** Discrete, clinically meaningful intervention bundles in the hematology domain. Do not use individual drugs due to sparsity. Bundling keeps action cardinality learnable across the 2,744 admissions. Document mapping to MIMIC source tables. Allowed actions include RBC transfusion, platelet transfusion, fresh frozen plasma/cryoprecipitate, IV iron therapy, erythropoiesis-stimulating agent (ESA), fluid resuscitation (crystalloid), electrolyte correction, anticoagulation hold/adjust, and no intervention/monitor.
- **0.4 Reward Function (r_t):** A composite, outcome-shaped reward function:

  r_t = w_1 _ 1[survival to discharge] + w_2 _ (-LOS_normalized) + w_3 _ (-|delta lab deviation from normal|) - w_4 _ 1[adverse event]

  Includes a terminal bonus/penalty at discharge or death. Weights (w_i) must be set in consultation with a clinician and undergo sensitivity analysis in Phase 4.

- **0.5 Transition Dynamics:** Next 4-hour lab bin accompanied by new event flags.
- **0.6 Deliverable:** A formal MDP specification document, requiring sign-off from at least one attending hematologist and one ML lead.

---

## Phase 1: Data Acquisition & Trajectory Construction (Weeks 1-3)

_Note: This phase runs in parallel with Phase 0 and represents the highest-leverage, riskiest portion of the project._

- **1.1 Source Linked Tables:** Extract data from MIMIC-IV utilizing existing `hadm_id` keys. Required tables: `labevents` (temporal lab trajectory), `prescriptions` and `emar` (medication actions), `inputevents` (transfusions, fluids, electrolytes), `procedures_icd` (procedural actions), `admissions` (mortality, LOS, readmission), `patients` (age, sex), and lookup tables (`d_labitems`, `d_icd_diagnoses`).
- **1.2 Join Strategy:** Key all data on `hadm_id`/`subject_id` and `charttime`. Verify complete presence of the 2,744 admission IDs.
- **1.3 Temporal Binning:** Resample lab events into fixed 4-hour bins per admission. Apply Last-Observation-Carried-Forward (LOCF) within admissions. Flag bins lacking data. Admissions with fewer than 6 bins must be excluded and the dropout rate documented.
- **1.4 Action Extraction:** Map source rows to the 9-action taxonomy per bin. Define and document a strict precedence rule or multi-hot vector strategy for bins containing multiple actions.
- **1.5 Reward Computation:** Compute per-bin and terminal rewards using the Phase 0 formula.
- **1.6 Trajectory Assembly:** Produce a list of (s*t, a_t, r_t, s*{t+1}) tuples per admission. Persist as a columnar Parquet dataset partitioned by `hadm_id`.
- **1.7 Phenotype Linkage:** Attach the Mantis cluster label and diagnosis text as static state context to condition the policy.
- **1.8 Deliverable:** A versioned, reproducible trajectory dataset (`trajectories_v1.parquet`) accompanied by a comprehensive data card.

### Phase 1 Testing Requirements

| Test ID | Description           | Acceptance Criteria                                                                     |
| ------- | --------------------- | --------------------------------------------------------------------------------------- |
| T1.1    | Key Integrity         | Every `hadm_id` in trajectories exists in `admissions` and `patients` (Fail count = 0). |
| T1.2    | Temporal Monotonicity | `charttime` is strictly increasing within each admission bin sequence.                  |
| T1.3    | Leakage Prevention    | No discharge/death timestamp appears before the last bin's `charttime`.                 |
| T1.4    | Action Coverage       | Every action class contains >= 50 instances (merge classes if necessary).               |
| T1.5    | Reward Sanity         | Terminal rewards fall within [-w_4, w_1+w_2+w_3] with no NaN/Inf values.                |
| T1.6    | Reproducibility       | Re-running the pipeline yields a byte-identical Parquet file (hash check).              |
| T1.7    | LOCF Guard            | Bins are never forward-filled across a gap > 24 hours (flagged as missing instead).     |
| T1.8    | Cohort Reporting      | Automatic Markdown dump of exclusions and distributions passes clinician review.        |

---

## Phase 2: Feature Engineering & Dataset Finalization (Weeks 2-3)

- **2.1 State Vector Construction:** Concatenate the following features into final state dimension d_s: 25 z-scored labs, one-hot demographics, sin/cos encoded time-since-admission, one-hot phenotype cluster (or 2D embedding), and last-3-bin deltas for trend capture.
- **2.2 Action Encoding:** Map to integers 0-8 (or multi-hot). Maintain an inverse map for system interpretability.
- **2.3 Normalization & Missingness:** Verify z-score scalers are fitted exclusively on training data. Add explicit missingness mask bits for labs.
- **2.4 Data Splits:** Split 70/15/15 by `subject_id` to prevent data leakage. Stratify splits by phenotype cluster and mortality.
- **2.5 Behavior Policy Logging:** Record empirical action frequencies pi_beta(a | s) conditioned on state bins for downstream off-policy evaluation.
- **2.6 Deliverable:** `dataset_v1` (train/val/test splits), feature specifications, and split manifest.

### Phase 2 Testing Requirements

| Test ID | Description       | Acceptance Criteria                                                                       |
| ------- | ----------------- | ----------------------------------------------------------------------------------------- |
| T2.1    | Patient Isolation | `subject_id` intersection across all data splits is absolutely empty.                     |
| T2.2    | Stratification    | Mortality rate in each split is within +/- 2pp of the overall cohort rate.                |
| T2.3    | Leakage Test      | Scaler is fit on train only; refitting on full data does not alter train features.        |
| T2.4    | Shape Contract    | `state.shape == (N, T, d_s)`, `action.shape == (N, T)`, `reward.shape == (N, T)`.         |
| T2.5    | Determinism       | Two builds from identical configurations produce mathematically identical seeded tensors. |

---

## Phase 3: Offline RL Model Training (Weeks 3-6)

_Note: Due to ethical and legal constraints, online RL on active patients is strictly prohibited. All training utilizes historical trajectories._

- **3.1 Baselines:** The model must demonstrably outperform Behavior Cloning (supervised classification of clinician actions) and the empirical Clinician Policy (pi_beta).
- **3.2 Core Algorithm:** Implement Implicit Q-Learning (IQL) as the primary algorithm to avoid querying out-of-distribution actions. Conservative Q-Learning (CQL) and Batch-Constrained Q-Learning (BCQ) should be implemented for baseline comparison.

  Q(s,a) <- fit via expectile regression of V(s) on tau-expectile of Q(s, pi_beta)
  pi(a | s) proportional to exp(beta \* (Q(s,a) - V(s)))

- **3.3 Network Architecture:** Start with an MLP Q-network (state -> 256 -> 256 -> |A|). A recurrent variant (LSTM) is considered a stretch goal. Strict regularization is required to prevent overfitting on the small dataset.
- **3.4 Training Regimen:** Utilize IQL value loss, policy loss, and entropy regularization. Optimize with Adam (lr: 3e-4, batch: 256, dropout: 0.2). Run across 5 random seeds and track Q-value/action distributions to monitor for action collapse.
- **3.5 Reward-Shaping Sensitivity:** Re-train models under three distinct reward-weight profiles to document policy stability.
- **3.6 Deliverable:** Trained `policy_v1` across all seeds, complete training logs, model card, and checkpoints.

### Phase 3 Testing Requirements

| Test ID | Description           | Acceptance Criteria                                                                  |
| ------- | --------------------- | ------------------------------------------------------------------------------------ |
| T3.1    | Baseline Superiority  | IQL's validation OPE score strictly exceeds Behavior Cloning scores.                 |
| T3.2    | Q-Collapse Prevention | Learned policy action entropy > 0; KL divergence vs. behavior is below threshold.    |
| T3.3    | Seed Stability        | Standard deviation of validation OPE across 5 seeds is < 15% of the mean.            |
| T3.4    | Finite Loss           | No NaN/Inf values present in any epoch; gradient norms remain bounded.               |
| T3.5    | Convergence           | Validation loss plateaus within budget; flag if still decreasing at timeout.         |
| T3.6    | Overfitting Check     | Discrepancies where Train OPE >> Val OPE automatically trigger added regularization. |
| T3.7    | Reward Robustness     | Rank correlation of top-5 actions across the 3 reward settings is > 0.6.             |

---

## Phase 4: Validation & Off-Policy Evaluation (Weeks 5-7)

_Evaluation is strictly limited to off-policy techniques, simulated environments, and expert review._

- **4.1 Off-Policy Evaluation (OPE):** Utilize three independent estimators requiring statistical agreement: Weighted Importance Sampling (WIS), Fitted Q-Evaluation (FQE), and Direct Method (DM).

  V*hat_WIS(pi_e) = sum_i(w_i * sum_t(gamma^t * r*{i,t})) / sum*i(w_i)
  w_i = product_t(pi_e(a*{i,t} | s*{i,t}) / pi_beta(a*{i,t} | s\_{i,t}))

- **4.2 Simulator Review:** Present tabular trajectories to >= 3 ICU attendings on 50 held-out admissions. Score based on agreement rate and clinical outcomes relative to historical choices.
- **4.3 Phenotype Stratification:** Evaluate policy performance independently across top clusters to ensure a globally positive mean does not obscure localized failures.
- **4.4 Safety Constraint Audit:** Institute absolute hard rules (e.g., no ESA with active malignancy). The policy must run against all test admissions with zero violations.
- **4.5 Interpretability:** Generate SHAP/attention vectors for sample recommendations. Explanations must achieve an actionability score of >= 4/5 from clinicians.
- **4.6 Deliverable:** Comprehensive evaluation report, OPE scorecard, safety audit logs, and interpretability artifacts.

### Phase 4 Testing Requirements

| Test ID | Description           | Acceptance Criteria                                                                                   |
| ------- | --------------------- | ----------------------------------------------------------------------------------------------------- | --------- | ---------------- | -------- | ------------- |
| T4.1    | OPE Agreement         | Estimators agree within variance:                                                                     | WIS - FQE | <= sigma_WIS and | FQE - DM | <= sigma_FQE. |
| T4.2    | Policy Efficacy       | V_hat(pi_e) > V_hat(pi_beta) with non-overlapping bootstrap confidence intervals.                     |
| T4.3    | Safety Zero-Tolerance | 0 constraint violations on test sets, including adversarially engineered data.                        |
| T4.4    | Phenotype Equity      | No top-20 cluster exists where V_hat(pi_e) < V_hat(pi_beta). Gate failing clusters out of deployment. |
| T4.5    | Clinician Agreement   | >= 70% of recommendations rated "reasonable or better" by at least 2 of 3 attendings.                 |
| T4.6    | Plausibility          | Spot-checks of high-reward trajectories reveal clinically plausible action sequences.                 |
| T4.7    | Statistical Rigor     | 1000-resample bootstrap CIs calculated on the headline metric to report effect size.                  |

---

## Phase 5: Integration into Platform (Weeks 6-8)

_The RL model trains offline; the platform consumes model outputs strictly as static metadata fields to surface counterfactuals and recommendations. There is no live inference loop._

- **5.1 Schema Definition:** Generate an `rl_outputs_v1` table mapping `hadm_id` to: recommended action, policy entropy, estimated value, top 3 actions (JSON), safety flags (JSON), semantic explanation, and confidence.
- **5.2 Enrichment:** Join outputs back to the admissions dataset and re-ingest as new columns, preserving existing cluster mapping for UI filtering.
- **5.3 UI Surfacing:** Deploy visualization tools comparing recommended actions against actual historical actions per cluster. Color maps based on recommendations.
- **5.4 Counterfactual Workflows:** Create specific review queues prioritizing admissions where the model most strongly diverges from clinical practice.
- **5.5 Point Detail Panels:** Render pre-computed text explanations and suggested steps upon point inspection.
- **5.6 Access Control:** Visibly label all outputs as "Decision Support." Gate features behind clinician roles and mandate rigorous audit logging.
- **5.7 Deliverable:** Enriched dataset map, updated UI surfaces, and finalized audit log specification.

### Phase 5 Testing Requirements

| Test ID | Description          | Acceptance Criteria                                                                            |
| ------- | -------------------- | ---------------------------------------------------------------------------------------------- |
| T5.1    | Enrichment Integrity | Every `hadm_id` output has a corresponding platform point; null counts on required fields = 0. |
| T5.2    | Schema Validation    | Data types are strictly enforced upon platform ingest.                                         |
| T5.3    | Static Execution     | Tests confirm zero live model artifacts are loaded or executed by the frontend.                |
| T5.4    | Distribution Parity  | UI-displayed cluster action distributions perfectly match offline Phase 4 reports.             |
| T5.5    | Labeling Compliance  | Automated checks confirm the presence of decision-support disclaimers on all surfaces.         |
| T5.6    | Audit Completeness   | Every recommendation view systematically writes to an immutable audit log.                     |
