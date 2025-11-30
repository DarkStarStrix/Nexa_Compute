"""Optional categorical feature enrichment."""

from typing import Dict

ADDUCT_TABLE = {
    "[M+H]+": 1,
    "[M+Na]+": 2,
    "[M-H]-": 3,
    "unknown": 0,
}

INSTRUMENT_TABLE = {
    "Orbitrap": 1,
    "QTOF": 2,
    "FTICR": 3,
    "unknown": 0,
}


def enrich_features(rec: dict) -> dict:
    """Enrich record with categorical feature IDs.

    Args:
        rec: Record dictionary

    Returns:
        Record with added adduct_id and instrument_id
    """
    rec["adduct_id"] = ADDUCT_TABLE.get(rec["adduct"], 0)
    rec["instrument_id"] = INSTRUMENT_TABLE.get(rec["instrument_type"], 0)
    return rec

