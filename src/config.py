import os

BIN_HOURS = 4
MIN_BINS = 6
LOCF_MAX_GAP_HOURS = 24

LAB_FEATURES = {
    # Hematology (original 24)
    "wbc": {"itemid": 51301, "unit": "K/uL", "lo": 4.5, "hi": 11.0},
    "rbc": {"itemid": 51279, "unit": "M/uL", "lo": 4.0, "hi": 5.5},
    "hemoglobin": {"itemid": 51222, "unit": "g/dL", "lo": 12.0, "hi": 16.0},
    "hematocrit": {"itemid": 51221, "unit": "%", "lo": 36.0, "hi": 46.0},
    "mcv": {"itemid": 51250, "unit": "fL", "lo": 80.0, "hi": 100.0},
    "mch": {"itemid": 51248, "unit": "pg", "lo": 27.0, "hi": 33.0},
    "mchc": {"itemid": 51249, "unit": "g/dL", "lo": 32.0, "hi": 36.0},
    "platelets": {"itemid": 51265, "unit": "K/uL", "lo": 150.0, "hi": 400.0},
    "serum_iron": {"itemid": 50952, "unit": "ug/dL", "lo": 50.0, "hi": 170.0},
    "ferritin": {"itemid": 50924, "unit": "ng/mL", "lo": 10.0, "hi": 200.0},
    "tibc": {"itemid": 50953, "unit": "ug/dL", "lo": 240.0, "hi": 450.0},
    "pt": {"itemid": 51274, "unit": "sec", "lo": 11.0, "hi": 13.5},
    "ptt": {"itemid": 51275, "unit": "sec", "lo": 25.0, "hi": 35.0},
    "inr": {"itemid": 51237, "unit": "ratio", "lo": 0.8, "hi": 1.2},
    "fibrinogen": {"itemid": 51214, "unit": "mg/dL", "lo": 200.0, "hi": 400.0},
    "d_dimer": {"itemid": 51196, "unit": "ng/mL FEU", "lo": 0.0, "hi": 500.0},
    "reticulocyte": {"itemid": 51283, "unit": "%", "lo": 0.5, "hi": 2.0},
    "haptoglobin": {"itemid": 50935, "unit": "mg/dL", "lo": 30.0, "hi": 200.0},
    "ldh": {"itemid": 50954, "unit": "U/L", "lo": 100.0, "hi": 250.0},
    "bilirubin_total": {"itemid": 50885, "unit": "mg/dL", "lo": 0.1, "hi": 1.2},
    "bilirubin_direct": {"itemid": 50883, "unit": "mg/dL", "lo": 0.0, "hi": 0.3},
    "b12": {"itemid": 51010, "unit": "pg/mL", "lo": 200.0, "hi": 900.0},
    "folate": {"itemid": 50925, "unit": "ng/mL", "lo": 3.0, "hi": 20.0},
    "transferrin_sat": {"itemid": 51746, "unit": "%", "lo": 20.0, "hi": 50.0},
    # Renal
    "creatinine": {"itemid": 50912, "unit": "mg/dL", "lo": 0.5, "hi": 1.3},
    "bun": {"itemid": 51006, "unit": "mg/dL", "lo": 7.0, "hi": 20.0},
    "potassium": {"itemid": 50971, "unit": "mEq/L", "lo": 3.5, "hi": 5.0},
    "sodium": {"itemid": 50983, "unit": "mEq/L", "lo": 136.0, "hi": 145.0},
    "chloride": {"itemid": 50902, "unit": "mEq/L", "lo": 98.0, "hi": 106.0},
    "bicarbonate": {"itemid": 50882, "unit": "mEq/L", "lo": 22.0, "hi": 26.0},
    # Hepatic
    "alt": {"itemid": 50861, "unit": "IU/L", "lo": 7.0, "hi": 56.0},
    "ast": {"itemid": 50878, "unit": "IU/L", "lo": 10.0, "hi": 40.0},
    "albumin": {"itemid": 50862, "unit": "g/dL", "lo": 3.5, "hi": 5.5},
    # Cardiac / metabolic
    "troponin_t": {"itemid": 50931, "unit": "ng/mL", "lo": 0.0, "hi": 0.01},
    "lactate": {"itemid": 50813, "unit": "mmol/L", "lo": 0.5, "hi": 2.0},
    # ABG
    "ph": {"itemid": 50820, "unit": "", "lo": 7.35, "hi": 7.45},
    "pco2": {"itemid": 50818, "unit": "mmHg", "lo": 35.0, "hi": 45.0},
    "po2": {"itemid": 50817, "unit": "mmHg", "lo": 80.0, "hi": 100.0},
    "spo2_lab": {"itemid": 50821, "unit": "%", "lo": 95.0, "hi": 100.0},
    # Inflammatory / coagulation (already have most above)
    "crp": {"itemid": 50889, "unit": "mg/L", "lo": 0.0, "hi": 10.0},
}

# Vitals from chartevents (separate from labs — different table, higher frequency)
VITAL_FEATURES = {
    "heart_rate": {"itemid": 220045, "unit": "bpm", "lo": 60.0, "hi": 100.0},
    "systolic_bp": {"itemid": 220050, "unit": "mmHg", "lo": 90.0, "hi": 140.0},
    "diastolic_bp": {"itemid": 220051, "unit": "mmHg", "lo": 60.0, "hi": 90.0},
    "mean_bp": {"itemid": 220052, "unit": "mmHg", "lo": 70.0, "hi": 105.0},
    "resp_rate": {"itemid": 220210, "unit": "/min", "lo": 12.0, "hi": 20.0},
    "spo2": {"itemid": 220277, "unit": "%", "lo": 95.0, "hi": 100.0},
    "temperature": {"itemid": 223761, "unit": "F", "lo": 97.0, "hi": 99.5},
}

TRANSFERRIN_SAT = {"numerator": "serum_iron", "denominator": "tibc", "lo": 20.0, "hi": 50.0}

LAB_FEATURES["glucose"] = {"itemid": 50893, "unit": "mg/dL", "lo": 70.0, "hi": 100.0}

ACTION_BUNDLES = {
    0: {"name": "no_intervention", "source": "derived", "itemids": [], "drugs": []},
    # Blood products
    1: {
        "name": "rbc_transfusion",
        "source": "prescriptions+chartevents",
        "itemids": [220997, 226267],
        "drugs": ["packed red blood", "prbc", "red blood cell"],
    },
    2: {
        "name": "platelet_transfusion",
        "source": "prescriptions+chartevents",
        "itemids": [225075, 225076],
        "drugs": ["platelet pheresis", "platelet"],
    },
    3: {
        "name": "ffp_cryo",
        "source": "prescriptions+chartevents",
        "itemids": [220989, 225771, 224929],
        "drugs": ["fresh frozen plasma", "ffp", "cryoprecipitate"],
    },
    4: {
        "name": "iv_iron",
        "source": "prescriptions",
        "drugs": [
            "iron sucrose",
            "ferric carboxymaltose",
            "iron dextran",
            "ferumoxytol",
            "iron (iv)",
        ],
    },
    5: {
        "name": "esa",
        "source": "prescriptions",
        "drugs": ["epoetin", "epogen", "procrit", "aranesp", "darbepoetin", "erythropoietin"],
    },
    # Fluids + electrolytes
    6: {
        "name": "fluid_resuscitation",
        "source": "prescriptions+chartevents",
        "itemids": [225158, 225159, 226391, 226392, 220862, 220986, 223258],
        "drugs": ["sodium chloride 0.9%", "lactated ringer", "normal saline", "d5w", "dextrose 5%"],
    },
    7: {
        "name": "electrolyte_correction",
        "source": "prescriptions",
        "drugs": [
            "potassium chloride",
            "magnesium sulfate",
            "calcium gluconate",
            "sodium phosphate",
            "potassium phosphate",
            "calcium chloride",
            "sodium bicarbonate",
        ],
    },
    # Anticoagulation
    8: {
        "name": "anticoag_hold",
        "source": "prescriptions+chartevents",
        "itemids": [225152],
        "drugs": [
            "warfarin",
            "coumadin",
            "rivaroxaban",
            "apixaban",
            "dabigatran",
            "enoxaparin",
            "lovenox",
            "heparin",
        ],
    },
    # Vasopressors / inotropes
    9: {
        "name": "vasopressor",
        "source": "prescriptions",
        "drugs": [
            "norepinephrine",
            "levophed",
            "vasopressin",
            "phenylephrine",
            "neosynephrine",
            "dobutamine",
            "dopamine",
            "epinephrine",
        ],
    },
    # Antibiotics
    10: {
        "name": "antibiotic",
        "source": "prescriptions",
        "drugs": [
            "vancomycin",
            "meropenem",
            "piperacillin",
            "cefepime",
            "ceftriaxone",
            "cefazolin",
            "ampicillin",
            "metronidazole",
            "azithromycin",
            "levofloxacin",
            "ciprofloxacin",
            "zosyn",
        ],
    },
    # Insulin
    11: {
        "name": "insulin",
        "source": "prescriptions",
        "drugs": [
            "insulin regular",
            "insulin glargine",
            "insulin lispro",
            "insulin aspart",
            "insulin humalog",
            "insulin novolog",
            "humulin",
            "novolin",
        ],
    },
    # Diuretics
    12: {
        "name": "diuretic",
        "source": "prescriptions",
        "drugs": [
            "furosemide",
            "lasix",
            "bumetanide",
            "mannitol",
            "spironolactone",
            "acetazolamide",
        ],
    },
    # Steroids
    13: {
        "name": "steroid",
        "source": "prescriptions",
        "drugs": [
            "prednisone",
            "methylprednisolone",
            "hydrocortisone",
            "dexamethasone",
            "solumedrol",
            "prednisolone",
        ],
    },
    # Sedation / analgesia
    14: {
        "name": "sedation_analgesia",
        "source": "prescriptions",
        "drugs": [
            "propofol",
            "midazolam",
            "versed",
            "fentanyl",
            "dexmedetomidine",
            "precedex",
            "hydromorphone",
            "dilaudid",
            "morphine",
            "ketamine",
        ],
    },
    # Cardiac
    15: {
        "name": "cardiac_rxn",
        "source": "prescriptions",
        "drugs": [
            "amiodarone",
            "metoprolol",
            "lopressor",
            "diltiazem",
            "cardizem",
            "nitroglycerin",
            "nitroprusside",
            "dobutamine",
        ],
    },
}

# Full hospital: all admissions (ICU + floor)
COHORT_MODE = "all_hosp"

COHORT_ICD10 = []
COHORT_ICD9 = []

LAB_THRESHOLD_COHORT = {"hgb_below": 10.0, "hgb_itemid": 51222}

PRECEDENCE = [1, 2, 3, 9, 5, 4, 10, 11, 15, 8, 13, 14, 12, 7, 6, 0]

REWARD_WEIGHTS = {
    "balanced": {"w1": 10.0, "w2": 1.0, "w3": 0.5, "w4": 5.0},
    "conservative": {"w1": 15.0, "w2": 0.5, "w3": 0.3, "w4": 8.0},
    "lab_focused": {"w1": 8.0, "w2": 0.5, "w3": 1.0, "w4": 3.0},
}

SAFETY_CONSTRAINTS = [
    {"id": "S1", "rule": "No ESA with active malignancy", "icd_prefix": ["C00", "C97"]},
    {"id": "S2", "rule": "No anticoag hold + surgery within 48h"},
    {
        "id": "S3",
        "rule": "Platelet transfusion only if Plt < 50K",
        "action": 2,
        "lab": "platelets",
        "threshold": 50.0,
        "direction": "below",
    },
    {
        "id": "S4",
        "rule": "FFP only if INR > 2.0",
        "action": 3,
        "lab": "inr",
        "threshold": 2.0,
        "direction": "above",
    },
    {
        "id": "S5",
        "rule": "No vasopressor if MAP >= 65 mmHg",
        "action": 9,
        "lab": "mean_bp",
        "threshold": 65.0,
        "direction": "above",
    },
    {
        "id": "S6",
        "rule": "No insulin if glucose < 70 mg/dL",
        "action": 11,
        "lab": "glucose",
        "threshold": 70.0,
        "direction": "below",
    },
    {"id": "S7", "rule": "No diuretic if creatinine > 4.0 + hypotension", "action": 12},
]

MIMIC_DATA_DIR = os.environ.get("MIMIC_DATA_DIR", "/Users/farasatdhedhi/mimic_pipeline/data")

N_ACTIONS = len(ACTION_BUNDLES)
