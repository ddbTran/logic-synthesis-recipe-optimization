"""Shared action indices and default RL/metaheuristic action budget."""

TRANSFORMS = [
    # Rewrite
    "rewrite",
    "rewrite -z",
    "rewrite -l",
    "rewrite -z -l",

    # Refactor
    "refactor",
    "refactor -z",
    "refactor -l",
    "refactor -z -l",

    # Resub
    "resub",
    "resub -z",
    "resub -l",
    "resub -z -l",

    # Balance / GIA
    "balance",
    "balance -l",
    "&get -n; &dsdb; &put",
    "&get -n; &sopb; &put",
]

MAX_TOOL_CALLS = 4000
