"""The data-lookup quiz shared by the dead drop and the live wiki: one question per round, "What is the <METRIC> in
<state>?", with a hidden official value per state drawn uniformly from 4.0-12.0 for each run seed."""

STATES = ["Nevada", "Oregon", "Ohio", "Maine", "Utah", "Iowa", "Kansas", "Idaho", "Vermont", "Montana",
          "Alabama", "Arizona"]
METRIC = "share of employed workers in construction occupations (%)"
