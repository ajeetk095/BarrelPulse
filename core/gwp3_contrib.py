"""Assessment of the eight contributions claimed in Alvi (2018), Chapter 5 (pp. 66-67).

Status vocabulary (evidence-based, deliberately conservative):
  Replicated in spirit — the group's pipeline implements the idea (not a like-for-like numeric match)
  Design claim — true by construction; no empirical test possible with our data
  Not testable here — needs data/analysis outside GWP3 (e.g. portfolio, policy use)
  Unsupported by our results — our evidence does not back the claim
"""
CONTRIBUTIONS = [
    ("C1", "Replace EGARCH-M views in Black-Litterman with Bayesian-model views",
     "Not testable here", "No Black-Litterman allocation was built in GWP3.",
     "Needs portfolio optimisation and a benchmark; outside scope."),
    ("C2", "Novel use of HMM-discretised series as inputs to belief networks",
     "Replicated in spirit", "The pipeline decodes HMM regimes and feeds them to a Bayesian network.",
     "Novelty is Alvi's claim; we only show the construction is implementable."),
    ("C3", "Deployable, autonomous trading mechanism",
     "Unsupported by our results", "Our network does not beat the majority baseline; the illustrative rule is evaluated on 23 months only.",
     "Costs, sizing and risk limits are absent; claim of direct deployability is not supported here."),
    ("C4", "Structure learning needing no expert prior knowledge",
     "Replicated in spirit", "Hill climbing runs with no seed (BIC/K2/BDeu) and with expert-seeded start (V4).",
     "The learned graph is thin (few edges), so the benefit is limited on this data."),
    ("C5", "Python abstraction aids understanding of the design process",
     "Design claim", "The pipeline is a few hundred lines on open-source libraries.",
     "Subjective; not an empirical result."),
    ("C6", "Systematic event-driven global-macro strategy with higher return than HFT/fixed-income funds",
     "Not testable here", "No fund-level comparison was run.",
     "Comparative-return claim needs return data we do not have."),
    ("C7", "Better models of energy markets for policy makers",
     "Not testable here", "A 5-node monthly model is a proof of concept.",
     "Policy use needs a larger, validated model."),
    ("C8", "Amalgamates multi-disciplinary research to raise alpha for commodity traders",
     "Unsupported by our results", "No alpha is demonstrated; baseline not beaten.",
     "Aspirational; the evidence here does not show alpha."),
]
NOTE = ("Chapter 5 of the dissertation lists these in running prose (the enumeration repeats 'Secondly' and "
        "restates the HMM→BN novelty twice); the eight above merge duplicates. The submitted notebook's 'C8' "
        "(long-term trend forecasting) does not correspond to Alvi's eighth statement.")
