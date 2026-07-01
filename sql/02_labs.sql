-- Lab extraction: 25 biomarkers binned into 4-hour intervals per admission
-- Input: cohort table (hadm_ids from 01_cohort.sql)
-- Uses CTE to define itemids; LOCF applied in Python, not SQL
-- NOTE: Python config (src/config.py LAB_FEATURES + VITAL_FEATURES) is authoritative.
-- This SQL uses the original v0.1 itemid set (includes ABO/Rh later removed).

WITH cohort AS (
    -- Replace with actual cohort table or CTE from 01_cohort.sql
    SELECT hadm_id, subject_id, admittime
    FROM cohort_admissions
),

lab_items AS (
    SELECT itemid, label
    FROM (VALUES
        (51301, 'wbc'),             (51279, 'rbc'),
        (51222, 'hemoglobin'),      (51221, 'hematocrit'),
        (51250, 'mcv'),             (51248, 'mch'),
        (51249, 'mchc'),            (51265, 'platelets'),
        (50908, 'serum_iron'),      (50924, 'ferritin'),
        (51010, 'tibc'),            (51274, 'pt'),
        (51275, 'ptt'),            (51237, 'inr'),
        (51214, 'fibrinogen'),      (51196, 'd_dimer'),
        (51287, 'reticulocyte'),    (50934, 'haptoglobin'),
        (50954, 'ldh'),             (50885, 'bilirubin_total'),
        (50883, 'bilirubin_direct'),(51138, 'b12'),
        (51224, 'folate'),          (51344, 'abo'),
        (51345, 'rh')
    ) AS t(itemid, label)
),

raw_labs AS (
    SELECT
        c.hadm_id,
        c.subject_id,
        li.label,
        le.charttime,
        le.valuenum,
        le.refitemid,
        -- Bin index: hours since admission / 4
        FLOOR(EXTRACT(EPOCH FROM (le.charttime - c.admittime)) / 3600.0 / 4) AS bin_idx
    FROM mimiciv_hosp.labevents le
    JOIN cohort c ON le.hadm_id = c.hadm_id
    JOIN lab_items li ON le.itemid = li.itemid
    WHERE le.valuenum IS NOT NULL
        AND le.hadm_id IS NOT NULL
),

-- Take the last value per (hadm_id, label, bin_idx) within each bin
binned AS (
    SELECT DISTINCT ON (hadm_id, label, bin_idx)
        hadm_id, label, valuenum, charttime, bin_idx
    FROM raw_labs
    ORDER BY hadm_id, label, bin_idx, charttime DESC
)

SELECT hadm_id, label, valuenum, charttime, bin_idx
FROM binned
WHERE bin_idx >= 0
ORDER BY hadm_id, bin_idx, label;
