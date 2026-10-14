"""Experiment suites. Each suite is a function returning a list of RunSpec.

The confirmatory suites below are frozen in docs/PREREGISTRATION.md; changing
them after results exist requires a new suite name, never an edit in place.
"""
from __future__ import annotations

from arithscope.train import RunSpec

MAIN_VARIANTS = ("nope", "alibi", "rope", "arith", "rope+arith", "rope+fourier_fund",
                 "rope+fourier_full", "rope+fourier_learned", "rope+relbias")

# Development data never overlaps the confirmatory test data (data_seed 20261014)
# and pilot runs use seed 0, which no confirmatory run uses.
DEV_DATA_SEED = 7


def pilot() -> list[RunSpec]:
    specs = []
    for period in (12, 13):
        for variant in MAIN_VARIANTS:
            for lr, mult in ((1e-3, 1.0), (3e-3, 10.0)):
                specs.append(RunSpec(experiment=f"pilot_p{period}_lr{lr:g}_m{mult:g}",
                                     task={"kind": "periodic", "periods": [period]}, variant=variant, seed=0,
                                     lr=lr, pos_lr_mult=mult, eval_lengths=(64, 256, 1024), eval_tokens=16384,
                                     data_seed=DEV_DATA_SEED))
    return specs


def pilot_controls() -> list[RunSpec]:
    """Feasibility of the control tasks and the alias fixture (development data only)."""
    specs = []
    for task in ({"kind": "tagged", "num_tags": 12}, {"kind": "mqar"}):
        for variant in ("nope", "alibi", "rope", "arith", "rope+arith"):
            for lr, mult in ((1e-3, 1.0), (3e-3, 10.0)):
                specs.append(RunSpec(experiment=f"pilotc_{task['kind']}_lr{lr:g}_m{mult:g}", task=task,
                                     variant=variant, seed=0, lr=lr, pos_lr_mult=mult,
                                     eval_lengths=(64, 256), eval_tokens=16384, data_seed=DEV_DATA_SEED))
    for bank in ((2, 3, 4, 6, 12), (3, 8)):
        specs.append(RunSpec(experiment=f"pilotc_alias_{'-'.join(map(str, bank))}",
                             task={"kind": "periodic", "periods": [24]}, variant="arith", orders=bank, seed=0,
                             lr=3e-3, pos_lr_mult=10.0, eval_lengths=(64, 256, 1024), eval_tokens=16384,
                             data_seed=DEV_DATA_SEED))
    return specs


def pilot_alias_query() -> list[RunSpec]:
    """Feasibility of the query-token alias fixture adopted for E2 (development data only)."""
    return [RunSpec(experiment=f"pilotq_alias_{'-'.join(map(str, bank))}",
                    task={"kind": "periodic", "periods": [24], "query_rate": 0.25}, variant="arith", orders=bank,
                    seed=0, lr=3e-3, pos_lr_mult=10.0, eval_lengths=(64, 256, 1024), eval_tokens=16384,
                    data_seed=DEV_DATA_SEED)
            for bank in ((2, 3, 4, 6, 12), (3, 8))]


def pilot_alias_design() -> list[RunSpec]:
    """Design check for E2 after pilot_alias_query (development data only).

    (b) P = 12 with lcm-6 vs lcm-12 banks at training length 64; (a') P = 24 trained at length 128.
    """
    specs = [RunSpec(experiment=f"pilotd_p12_{'-'.join(map(str, bank))}",
                     task={"kind": "periodic", "periods": [12], "query_rate": 0.25}, variant="arith", orders=bank,
                     seed=0, lr=3e-3, pos_lr_mult=10.0, eval_lengths=(64, 256, 1024), eval_tokens=16384,
                     data_seed=DEV_DATA_SEED)
             for bank in ((2, 3), (3, 4))]
    specs.append(RunSpec(experiment="pilotd_p24_n128_3-8", task={"kind": "periodic", "periods": [24], "query_rate": 0.25},
                         variant="arith", orders=(3, 8), seed=0, lr=3e-3, pos_lr_mult=10.0, train_length=128,
                         eval_lengths=(128, 512, 2048), eval_tokens=16384, data_seed=DEV_DATA_SEED))
    return specs


def pilot_mqar_easy() -> list[RunSpec]:
    """MQAR feasibility after pilot_controls showed a floor effect (development data only)."""
    return [RunSpec(experiment="pilotm_mqar_k64_v16_4000", task={"kind": "mqar", "num_keys": 64, "num_values": 16},
                    variant=v, seed=0, lr=3e-3, pos_lr_mult=10.0, steps=4000, eval_lengths=(64, 128, 256),
                    eval_tokens=16384, data_seed=DEV_DATA_SEED)
            for v in ("rope", "rope+arith")]


# ----------------------------------------------------------------------------------------------
# Confirmatory suites (docs/PREREGISTRATION.md). Hyperparameters come from the preregistered
# pilot rule (experiments/select_hparams.py -> results/summary/hparam_selection.json).
CONFIRM_SEEDS = (1, 2, 3, 4, 5)
SECONDARY_SEEDS = (1, 2, 3)
LONG_EVAL = (64, 128, 256, 512, 1024, 2048)
E1_EVAL = LONG_EVAL + (4096, 8192)  # up to 128x the training length
SHIFTS = (0, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9)
CONTROL_VARIANTS = ("nope", "alibi", "rope", "arith", "rope+arith", "rope+fourier_fund", "rope+relbias")
# E2 (period 12): banks with lcm 6 alias the 12-comb; banks with lcm 12 resolve it (docs/MATH.md, T4/P1).
ALIAS_PERIOD = 12
ALIAS_BANKS = ((2, 3), (1, 2, 3, 6), (3, 4), (4, 6), (12,), (1, 2, 3, 4, 6, 12))

# Frozen output of the preregistered pilot rule (experiments/select_hparams.py ->
# results/summary/hparam_selection.json): (task family, variant) -> (lr, positional-LR multiplier).
# Variants without a pilot for a family inherit their periodic choice.
HPARAMS: dict[tuple[str, str], tuple[float, float]] = {
    **{("periodic", v): (3e-3, 10.0) for v in MAIN_VARIANTS},
    ("periodic", "alibi"): (1e-3, 1.0),
    **{("tagged", v): (3e-3, 10.0) for v in ("nope", "alibi", "rope", "arith", "rope+arith")},
}


def _hp(family: str, variant: str) -> dict:
    lr, mult = HPARAMS.get((family, variant), HPARAMS[("periodic", variant)])
    return {"lr": lr, "pos_lr_mult": mult}


def e1_periodic() -> list[RunSpec]:
    return [RunSpec(experiment=f"e1_p{period}", task={"kind": "periodic", "periods": [period]}, variant=v,
                    seed=seed, eval_lengths=E1_EVAL, shift_evals=SHIFTS, **_hp("periodic", v))
            for period in (12, 13) for v in MAIN_VARIANTS for seed in CONFIRM_SEEDS]


def e2_alias() -> list[RunSpec]:
    return [RunSpec(experiment="e2_alias", task={"kind": "periodic", "periods": [ALIAS_PERIOD], "query_rate": 0.25},
                    variant="arith",
                    orders=bank, seed=seed, eval_lengths=LONG_EVAL, **_hp("periodic", "arith"))
            for bank in ALIAS_BANKS for seed in CONFIRM_SEEDS]


def e3_tagged() -> list[RunSpec]:
    return [RunSpec(experiment="e3_tagged", task={"kind": "tagged", "num_tags": 12}, variant=v, seed=seed,
                    eval_lengths=LONG_EVAL, **_hp("tagged", v))
            for v in CONTROL_VARIANTS for seed in CONFIRM_SEEDS]


# e4_mqar was dropped before confirmatory runs (floor effect in pilots; docs/PREREGISTRATION.md, Section 6).


def e5_mixed() -> list[RunSpec]:
    return [RunSpec(experiment="e5_mixed", task={"kind": "periodic", "periods": [7, 11, 13]}, variant=v,
                    seed=seed, eval_lengths=LONG_EVAL, **_hp("periodic", v))
            for v in MAIN_VARIANTS for seed in SECONDARY_SEEDS]


SUITES = {"pilot": pilot, "pilot_controls": pilot_controls, "pilot_alias_query": pilot_alias_query,
          "pilot_alias_design": pilot_alias_design, "pilot_mqar_easy": pilot_mqar_easy,
          "e1_periodic": e1_periodic, "e2_alias": e2_alias,
          "e3_tagged": e3_tagged, "e5_mixed": e5_mixed}
