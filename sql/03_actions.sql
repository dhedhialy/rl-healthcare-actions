-- Action extraction: map MIMIC events to discrete action bundles per 4-hour bin
-- Uses precedence rule: RBC(1) > Plt(2) > FFP(3) > ESA(5) > Fe(4) > Anticoag(8) > Lyte(7) > Fluid(6) > None(0)
-- NOTE: Python config (src/config.py ACTION_BUNDLES + PRECEDENCE) is authoritative.
-- This SQL reflects the original v0.1 9-action scheme; Python now supports 16 action bundles.

WITH cohort AS (
    SELECT hadm_id, subject_id, admittime
    FROM cohort_admissions
),

-- ICU input events: blood products, fluids, electrolytes, heparin
icu_actions AS (
    SELECT
        c.hadm_id,
        ie.starttime AS event_time,
        CASE
            WHEN ie.itemid IN (220997, 226267) THEN 1   -- RBC transfusion
            WHEN ie.itemid IN (225075, 225076) THEN 2   -- Platelet transfusion
            WHEN ie.itemid IN (220989, 225771, 224929) THEN 3  -- FFP/Cryo
            WHEN ie.itemid IN (225158, 225159, 226391, 226392, 220862, 220986, 223258) THEN 6  -- Fluid
            WHEN ie.itemid IN (227970, 228008, 225866, 228009, 225833, 225831, 224062) THEN 7  -- Electrolyte
            WHEN ie.itemid = 225152 THEN 8               -- Heparin drip (anticoag)
            ELSE NULL
        END AS action_id,
        FLOOR(EXTRACT(EPOCH FROM (ie.starttime - c.admittime)) / 3600.0 / 4) AS bin_idx
    FROM mimiciv_icu.inputevents ie
    JOIN cohort c ON ie.hadm_id = c.hadm_id
    WHERE ie.itemid IN (
        220997, 226267,           -- RBC
        225075, 225076,           -- Platelets
        220989, 225771, 224929,   -- FFP/Cryo
        225158, 225159, 226391, 226392, 220862, 220986, 223258,  -- Fluids
        227970, 228008, 225866, 228009, 225833, 225831, 224062,  -- Electrolytes
        225152                    -- Heparin
    )
),

-- Prescription-based actions: IV iron, ESA, anticoagulation meds
rx_actions AS (
    SELECT
        c.hadm_id,
        p.starttime AS event_time,
        CASE
            WHEN LOWER(p.drug) ~ 'iron\s+sucrose|ferric\s+carboxymaltose|iron\s+dextran|ferumoxytol' THEN 4
            WHEN LOWER(p.drug) ~ 'epoetin|epogen|procrit|aranesp|darbepoetin|erythropoietin' THEN 5
            WHEN LOWER(p.drug) ~ 'warfarin|coumadin|rivaroxaban|apixaban|dabigatran|enoxaparin|lovenox' THEN 8
            ELSE NULL
        END AS action_id,
        FLOOR(EXTRACT(EPOCH FROM (p.starttime - c.admittime)) / 3600.0 / 4) AS bin_idx
    FROM mimiciv_hosp.prescriptions p
    JOIN cohort c ON p.hadm_id = c.hadm_id
    WHERE p.starttime IS NOT NULL
        AND (
            LOWER(p.drug) ~ 'iron\s+sucrose|ferric\s+carboxymaltose|iron\s+dextran|ferumoxytol'
            OR LOWER(p.drug) ~ 'epoetin|epogen|procrit|aranesp|darbepoetin|erythropoietin'
            OR LOWER(p.drug) ~ 'warfarin|coumadin|rivaroxaban|apixaban|dabigatran|enoxaparin|lovenox'
        )
),

all_actions AS (
    SELECT hadm_id, event_time, action_id, bin_idx FROM icu_actions WHERE action_id IS NOT NULL
    UNION ALL
    SELECT hadm_id, event_time, action_id, bin_idx FROM rx_actions WHERE action_id IS NOT NULL
),

-- Apply precedence: pick highest-acuity action per (hadm_id, bin_idx)
precedenced AS (
    SELECT DISTINCT ON (hadm_id, bin_idx)
        hadm_id, bin_idx, action_id
    FROM all_actions
    ORDER BY hadm_id, bin_idx, action_id ASC  -- lower action_id = higher acuity in precedence list
)

SELECT hadm_id, bin_idx, action_id
FROM precedenced
WHERE bin_idx >= 0
ORDER BY hadm_id, bin_idx;
