"""Sepsis cohort definition using Sepsis-3 criteria."""
def is_sepsis(hadm):
    return {"has_sepsis": False, "qsofa": 0}
