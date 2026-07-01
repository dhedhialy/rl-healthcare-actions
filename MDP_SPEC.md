# MDP Specification: Hematology/Anemia/Obstetric-Bleeding RL

**Version:** 0.2 (itemids verified, data limitations documented)  
**Scope:** MIMIC-IV v2.2, hematology/anemia/obstetric-bleeding phenotype  
**Admission Count:** 114,540 (after filtering ≥6 bins); 236,324 (ICD + Hgb<10 cohort)

---

## 1. State Space (s_t)

State at each 4-hour bin is a fixed-dimension vector of d_s features.

### 1.1 Lab Biomarkers (24 items from `labevents`)

All itemids verified against `d_labitems.csv`. No asterisked items remain.

| #   | Biomarker          | itemid | Source      | Units     | Normal Range | Sparsity |
| --- | ------------------ | ------ | ----------- | --------- | ------------ | -------- |
| 1   | WBC                | 51301  | Hematology  | K/uL      | 4.5–11.0     | 18%      |
| 2   | RBC                | 51279  | Hematology  | M/uL      | 4.0–5.5      | 18%      |
| 3   | Hemoglobin         | 51222  | Hematology  | g/dL      | 12.0–16.0    | 17%      |
| 4   | Hematocrit         | 51221  | Hematology  | %         | 36–46        | 12%      |
| 5   | MCV                | 51250  | Hematology  | fL        | 80–100       | 18%      |
| 6   | MCH                | 51248  | Hematology  | pg        | 27–33        | 18%      |
| 7   | MCHC               | 51249  | Hematology  | g/dL      | 32–36        | 18%      |
| 8   | Platelets          | 51265  | Hematology  | K/uL      | 150–400      | 16%      |
| 9   | Serum Iron         | 50952  | Chemistry   | ug/dL     | 50–170       | 98%      |
| 10  | Ferritin           | 50924  | Chemistry   | ng/mL     | 10–200       | 98%      |
| 11  | TIBC               | 50953  | Chemistry   | ug/dL     | 240–450      | 98%      |
| 12  | Transferrin Sat    | 51746  | Chemistry   | %         | 20–50        | ~100%    |
| 13  | PT                 | 51274  | Coagulation | sec       | 11–13.5      | 58%      |
| 14  | PTT                | 51275  | Coagulation | sec       | 25–35        | 55%      |
| 15  | INR                | 51237  | Coagulation | ratio     | 0.8–1.2      | 58%      |
| 16  | Fibrinogen         | 51214  | Coagulation | mg/dL     | 200–400      | 95%      |
| 17  | D-Dimer            | 51196  | Coagulation | ng/mL FEU | <500         | ~100%    |
| 18  | Reticulocyte Count | 51283  | Hematology  | %         | 0.5–2.0      | 99%      |
| 19  | Haptoglobin        | 50935  | Chemistry   | mg/dL     | 30–200       | 99%      |
| 20  | LDH                | 50954  | Enzyme      | U/L       | 100–250      | 83%      |
| 21  | Bilirubin, Total   | 50885  | Enzyme      | mg/dL     | 0.1–1.2      | 72%      |
| 22  | Bilirubin, Direct  | 50883  | Enzyme      | mg/dL     | 0.0–0.3      | 99%      |
| 23  | Vitamin B12        | 51010  | Chemistry   | pg/mL     | 200–900      | 99%      |
| 24  | Folate             | 50925  | Chemistry   | ng/mL     | >3.0         | ~100%    |

Transferrin Saturation uses direct measurement itemid 51746; derived (Serum Iron/TIBC) used as fallback.

**Removed from v0.1:** ABO/Rh (itemids 51344/51345 not present in d_labitems).

### 1.2 Non-Lab State Features

| Feature                  | Source                    | Encoding                 |
| ------------------------ | ------------------------- | ------------------------ |
| Age                      | `patients.anchor_age`     | Z-scored                 |
| Sex                      | `patients.gender`         | One-hot (M/F)            |
| Time-since-admission     | Computed from `admittime` | sin/cos of hours         |
| Phenotype cluster        | Mantis 359-cluster label  | One-hot or 2D embedding  |
| Diagnosis text embedding | `diagnoses_text`          | 2D embedding from Mantis |
| Last-3-bin lab deltas    | Computed from lab history | Raw trend values         |
| Missingness mask         | Computed per bin          | Binary flags (24 bits)   |

### 1.3 Observation Window

Fixed 4-hour bins starting from `admittime`. LOCF within admissions, but never forward-filled across a gap >24h (flagged missing instead).

---

## 2. Action Space (a_t)

Discrete, 9 clinically meaningful intervention bundles.

| Action ID | Bundle                                | Primary MIMIC Source                        | Key itemids / Drug Names                                                                                              |
| --------- | ------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 0         | **No intervention / Monitor**         | Derived by absence                          | No active intervention in bin                                                                                         |
| 1         | **RBC Transfusion**                   | Hgb jump heuristic (_see data limitations_) | Hgb increase ≥1.5 g/dL between bins (proxy; itemids 220997/226267 in inputevents, unavailable)                        |
| 2         | **Platelet Transfusion**              | `inputevents`                               | 225075 (Platelet Pheresis), 225076 (Platelet)                                                                         |
| 3         | **FFP / Cryoprecipitate**             | `inputevents`                               | 220989 (FFP), 225771 (FFP PACU), 224929 (Cryo)                                                                        |
| 4         | **IV Iron Therapy**                   | `prescriptions`                             | iron sucrose, ferric carboxymaltose, iron dextran, ferumoxytol (route=IV)                                             |
| 5         | **ESA**                               | `prescriptions`                             | epoetin, epogen, procrit, aranesp, darbepoetin, erythropoietin                                                        |
| 6         | **Fluid Resuscitation (Crystalloid)** | `inputevents`                               | 225158, 225159 (NS 1L/500mL), 226391, 226392 (LR 1L/500mL), 220862 (NS generic), 220986 (LR generic), 223258 (D5W)    |
| 7         | **Electrolyte Correction**            | `inputevents`                               | 227970, 228008 (KCl), 225866 (K-Phos), 228009 (Na-Phos), 225833 (Ca gluconate), 225831, 224062 (MgSO4)                |
| 8         | **Anticoagulation Hold/Adjust**       | `inputevents` + `prescriptions`             | 225152 (Heparin drip) + warfarin, coumadin, rivaroxaban, apixaban, dabigatran, enoxaparin, lovenox from prescriptions |

### Precedence Rule

When multiple actions occur in the same 4-hour bin, use the **highest-acuity** action:
1 (RBC) > 2 (Plt) > 3 (FFP) > 5 (ESA) > 4 (Fe) > 8 (Anticoag) > 7 (Lyte) > 6 (Fluid) > 0 (None)

Alternatively, encode as multi-hot vector for downstream architectural flexibility.

### Validation SQL

```sql
-- Verify ICU blood product itemids
SELECT itemid, label, category, linksto
FROM mimiciv_icu.d_items
WHERE LOWER(label) LIKE ANY (
  '%packed red blood%', '%prbc%', '%platelet%',
  '%fresh frozen%', '%cryoprecipitate%', '%ffp%'
)
AND linksto = 'inputevents'
ORDER BY label;
```

---

## 3. Reward Function (r_t)

Composite, outcome-shaped:

```
r_t = w1 * 1[survival to discharge]
    + w2 * (-LOS_normalized)
    + w3 * (-|Δlab deviation from normal|)
    - w4 * 1[adverse event]
```

### 3.1 Proposed Initial Weights

| Weight | Component              | Initial Value | Rationale                                      |
| ------ | ---------------------- | ------------- | ---------------------------------------------- |
| w1     | Survival to discharge  | +10.0         | Dominant positive signal; binary               |
| w2     | Length of stay penalty | +1.0          | Shorter stays preferred; LOS in days, z-scored |
| w3     | Lab normalization      | +0.5          | Penalize deviation from normal ranges per bin  |
| w4     | Adverse event penalty  | -5.0          | Transfusion reaction, re-bleed, thrombosis     |

### 3.2 Adverse Event Definition

- Transfusion reaction (ICD: D70-D77 complications)
- Re-bleed within 24h of RBC transfusion (>2g/dL Hgb drop)
- Thrombotic event on anticoagulation hold
- Hospital-acquired infection during pancytopenia

### 3.3 Terminal Reward

Applied only at the final bin (discharge or death):

- Survival: +w1 - w2 \* LOS_norm - sum(lab_deviations)
- Death: -w4 (replaces +w1)

### 3.4 Sensitivity Analysis

Retrain under three profiles in Phase 3:

- **Conservative:** w1=15, w2=0.5, w3=0.3, w4=8
- **Balanced (default):** w1=10, w2=1.0, w3=0.5, w4=5
- **Lab-focused:** w1=8, w2=0.5, w3=1.0, w4=3

---

## 4. Transition Dynamics

s\_{t+1} = next 4-hour bin of labs + updated event flags + upweighted time-since-admission.

Non-stationary: patient state evolves deterministically from EHR data. No environment simulation — transition model is purely empirical from MIMIC trajectories.

---

## 5. Discount Factor

gamma = 0.99 (long-horizon; admissions span up to ~60 bins for 10-day stays).

---

## 6. Safety Constraints (Hard Rules)

These are inviolable regardless of learned policy:

1. **No ESA with active malignancy** (ICD-10 C00-C97)
2. **No anticoagulation hold + surgery within 48h** (procedures_icd)
3. **Platelet transfusion only if Plt < 50K** or active bleeding
4. **FFP only if INR > 2.0 or active coagulopathic bleeding**

Failures on these are zero-tolerance (T4.3).

---

## 7. Data Limitations (MIMIC-IV CSV Extraction)

| Issue                                 | Impact                                                                                                  | Mitigation                                                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **No `inputevents` table**            | RBC transfusion (action 1) cannot be directly extracted; itemids 220997, 226267 absent from chartevents | Inferred from Hgb jumps ≥1.5 g/dL between bins (~44K instances); false positives possible (e.g., error correction, hemoconcentration) |
| **Truncated `chartevents.csv.gz`**    | ~35M readable rows before EOF; platelet/FFP counts may be underestimated                                | Pre-extracted blood product events (5,385 rows) from readable portion                                                                 |
| **Mixed ICD-9/10 codes**              | ~55% ICD-10, ~45% ICD-9 in diagnoses table                                                              | Both code sets included in cohort extraction                                                                                          |
| **Sparse bins**                       | Mean ~10 bins per admission; labs drawn 1-2x daily in ICU                                               | LOCF with 24h gap limit; missing labs treated as zero deviation                                                                       |
| **RBC transfusion: no direct record** | Only inferred, not observed                                                                             | Hgb jump threshold (1.5 g/dL) is clinically conservative; if inputevents obtained, replace heuristic                                  |

### Verified Action Counts (from pipeline run)

| Action                   | Count     | % of Transitions |
| ------------------------ | --------- | ---------------- |
| 0 (No intervention)      | 1,538,315 | 82.3%            |
| 1 (RBC Transfusion)      | 39,361    | 2.1%             |
| 2 (Platelet Transfusion) | 1,204     | 0.06%            |
| 3 (FFP/Cryo)             | 165       | 0.01%            |
| 4 (IV Iron)              | 101       | 0.005%           |
| 5 (ESA)                  | 8,454     | 0.45%            |
| 6 (Fluid)                | 148,932   | 7.97%            |
| 7 (Electrolyte)          | 72,232    | 3.87%            |
| 8 (Anticoag)             | 59,393    | 3.18%            |

---

## Sign-Off

| Role                   | Name | Date | Signature |
| ---------------------- | ---- | ---- | --------- |
| Attending Hematologist |      |      |           |
| ML Lead                |      |      |           |
