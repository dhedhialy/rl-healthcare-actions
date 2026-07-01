-- Cohort extraction: hematology/anemia/obstetric-bleeding phenotype
-- Target: ~2,744 hadm_ids from MIMIC-IV v2.2
-- Two inclusion paths: (1) ICD-10 diagnosis codes, (2) lab threshold on hemoglobin
-- NOTE: Python implementation in src/cohort/extract.py is the source of truth.
-- This SQL mirrors the original MIMIC direct-database approach (v0.1 spec).

WITH icd_cohort AS (
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN mimiciv_hosp.d_icd_diagnoses dic ON d.icd_code = dic.icd_code
        AND d.icd_version = dic.icd_version
    WHERE d.icd_version = 10
        AND (
            -- Anemias D50-D64
            d.icd_code LIKE 'D50%'
            OR d.icd_code LIKE 'D51%'
            OR d.icd_code LIKE 'D52%'
            OR d.icd_code LIKE 'D53%'
            OR d.icd_code LIKE 'D55%'
            OR d.icd_code LIKE 'D56%'
            OR d.icd_code LIKE 'D57%'
            OR d.icd_code LIKE 'D58%'
            OR d.icd_code LIKE 'D59%'
            OR d.icd_code LIKE 'D60%'
            OR d.icd_code LIKE 'D61%'
            OR d.icd_code LIKE 'D62%'
            OR d.icd_code LIKE 'D63%'
            OR d.icd_code LIKE 'D64%'
            -- Coagulation defects D65-D69
            OR d.icd_code LIKE 'D65%'
            OR d.icd_code LIKE 'D66%'
            OR d.icd_code LIKE 'D67%'
            OR d.icd_code LIKE 'D68%'
            OR d.icd_code LIKE 'D69%'
            -- Other diseases of blood D70-D77
            OR d.icd_code LIKE 'D70%'
            OR d.icd_code LIKE 'D71%'
            OR d.icd_code LIKE 'D72%'
            OR d.icd_code LIKE 'D73%'
            OR d.icd_code LIKE 'D74%'
            OR d.icd_code LIKE 'D75%'
            OR d.icd_code LIKE 'D76%'
            OR d.icd_code LIKE 'D77%'
            -- Obstetric hemorrhage O44-O46, O67, O72
            OR d.icd_code LIKE 'O44%'
            OR d.icd_code LIKE 'O45%'
            OR d.icd_code LIKE 'O46%'
            OR d.icd_code LIKE 'O67%'
            OR d.icd_code LIKE 'O72%'
        )
),

lab_cohort AS (
    SELECT DISTINCT le.hadm_id
    FROM mimiciv_hosp.labevents le
    WHERE le.itemid = 51222  -- Hemoglobin
        AND le.valuenum IS NOT NULL
        AND le.hadm_id IS NOT NULL
        AND le.valuenum < 10.0
),

raw_cohort AS (
    SELECT hadm_id FROM icd_cohort
    UNION
    SELECT hadm_id FROM lab_cohort
),

cohort_meta AS (
    SELECT
        c.hadm_id,
        a.subject_id,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.hospital_expire_flag,
        p.anchor_age,
        p.gender,
        EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400.0 AS los_days
    FROM raw_cohort c
    JOIN mimiciv_hosp.admissions a ON c.hadm_id = a.hadm_id
    JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
    WHERE a.dischtime > a.admittime
)

SELECT
    hadm_id,
    subject_id,
    admittime,
    dischtime,
    deathtime,
    hospital_expire_flag,
    anchor_age,
    gender,
    los_days,
    'icd_or_lab' AS inclusion_criteria
FROM cohort_meta
ORDER BY hadm_id;
