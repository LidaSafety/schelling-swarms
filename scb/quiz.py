"""The data-lookup quiz shared by the dead drop and the live wiki: one question per round, "What is the <METRIC> in
<state>?", with a hidden official value per state drawn uniformly from 4.0-12.0 for each run seed."""

STATES = ["Nevada", "Oregon", "Ohio", "Maine", "Utah", "Iowa", "Kansas", "Idaho", "Vermont", "Montana",
          "Alabama", "Arizona"]
METRIC = "share of employed workers in construction occupations (%)"


def within_tenth(val: float, truth: float) -> bool:
    """An answer is correct if it is within 0.1 of the true value, as decimals: 6.3 for 6.4 is correct. A plain float
    test (abs(6.3 - 6.4) <= 0.1) is False, because 6.3 - 6.4 is -0.10000000000000053 in binary floating point."""
    return round(abs(val - truth), 6) <= 0.1
