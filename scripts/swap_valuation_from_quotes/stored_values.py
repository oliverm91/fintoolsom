"""MTMs stored in ``derivatives.swaps_mtm``, kept as constants to compare the valuations against."""

from datetime import date

# Swap 10 (CLP; the DB signs the pay leg negative).
STORED_MTM = {
    (date(2026, 11, 6), "SLMNID5G"): 16_955_420_167.0,  # receive leg, fixed 4.92%
    (date(2026, 11, 6), "SLMNID5H"): -17_015_550_924.0,  # pay leg, ICP
    (date(2026, 11, 9), "SLMNID5G"): 16_966_709_576.0,  # receive leg
    (date(2026, 11, 9), "SLMNID5H"): -17_011_472_728.0,  # pay leg
}


def stored_mtm(t: date, leg_ids: list[str]) -> dict[str, float]:
    """Stored MTM per leg id on ``t``; a leg without a stored value is left out."""
    return {
        leg_id: STORED_MTM[(t, leg_id)]
        for leg_id in leg_ids
        if (t, leg_id) in STORED_MTM
    }
