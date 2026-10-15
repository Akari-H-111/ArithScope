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


# ----------------------------------------------------------------------------------------------
# E6 (docs/PREREGISTRATION.md, Section 9): two remedies for the NLL rise of RoPE models at long
# lengths (HANDOFF open question 3). Every arm uses E1's architecture and its preregistered periodic
# hyperparameters; nothing is tuned. Within a period and seed, all arms share the experiment name,
# hence the initial weights and the training-batch generator.
E6_DATA_SEED = 20261016  # fresh test data: E1-E5 used 20261014, the post-hoc NLL diagnosis 20261015
E6_WINDOW = 64           # second-layer key window = the training length (offsets seen in training)
E6_EVAL_TOKENS_LONG = 32768  # 4 test sequences at N = 8192 (E1: 2)
E6_ARMS: dict[str, dict] = {
    "reference": {"variant": "arith"},                                  # no RoPE: flat NLL in E1
    "baseline": {"variant": "rope+arith"},                              # E1's rope+arith
    "window": {"variant": "rope+arith", "attn_windows": (None, E6_WINDOW)},
    "long_train": {"variant": "rope+arith", "train_length": 256, "batch_size": 8},  # same tokens per step
}


def e6_arm(spec: RunSpec) -> str:
    """The E6 arm a run belongs to."""
    for arm, overrides in E6_ARMS.items():
        if all(getattr(spec, k) == v for k, v in overrides.items()) and \
                all(getattr(spec, k) == getattr(RunSpec, k) for k in ("attn_windows", "train_length", "batch_size")
                    if k not in overrides):
            return arm
    raise ValueError(f"not an E6 arm: {spec.filename}")


def _e6(experiment: str, seeds, data_seed: int, eval_lengths, eval_tokens_long: int, periods=(12, 13)) -> list[RunSpec]:
    return [RunSpec(experiment=f"{experiment}_p{period}", task={"kind": "periodic", "periods": [period]}, seed=seed,
                    eval_lengths=eval_lengths, eval_tokens_long=eval_tokens_long, data_seed=data_seed,
                    **overrides, **_hp("periodic", overrides["variant"]))
            for period in periods for overrides in E6_ARMS.values() for seed in seeds]


def pilot_e6() -> list[RunSpec]:
    """Feasibility of the E6 arms on development data: training only, evaluated at length 64 (no extrapolation)."""
    return _e6("pilot_e6", (0,), DEV_DATA_SEED, (64,), E6_EVAL_TOKENS_LONG, periods=(12,))


def e6_calibration() -> list[RunSpec]:
    return _e6("e6", CONFIRM_SEEDS, E6_DATA_SEED, E1_EVAL, E6_EVAL_TOKENS_LONG)


# ----------------------------------------------------------------------------------------------
# E7 (docs/PREREGISTRATION.md, Section 10): is ArithScope's own NLL floor partly its broad second
# layer? E6's window arm ended below ArithScope alone (its S4 failed). E7 windows ArithScope alone.
# Same architecture, hyperparameters, seeds, and window as E6; a new experiment name (new training
# batches, same initial weights per seed) and fresh test data.
E7_DATA_SEED = 20261017  # fresh: E1-E5 used 20261014, the post-hoc diagnosis 20261015, E6 20261016
E7_ARMS: dict[str, dict] = {
    "reference": {"variant": "arith"},                                      # E6's reference
    "arith_window": {"variant": "arith", "attn_windows": (None, E6_WINDOW)},  # the new arm
    "window": {"variant": "rope+arith", "attn_windows": (None, E6_WINDOW)},   # E6's window arm
}


def e7_arm(spec: RunSpec) -> str:
    """The E7 arm a run belongs to."""
    for arm, overrides in E7_ARMS.items():
        if spec.variant == overrides["variant"] and tuple(spec.attn_windows) == overrides.get("attn_windows", ()) \
                and spec.train_length == RunSpec.train_length and spec.batch_size == RunSpec.batch_size:
            return arm
    raise ValueError(f"not an E7 arm: {spec.filename}")


def e7_floor() -> list[RunSpec]:
    return [RunSpec(experiment=f"e7_p{period}", task={"kind": "periodic", "periods": [period]}, seed=seed,
                    eval_lengths=E1_EVAL, eval_tokens_long=E6_EVAL_TOKENS_LONG, data_seed=E7_DATA_SEED,
                    **overrides, **_hp("periodic", overrides["variant"]))
            for period in (12, 13) for overrides in E7_ARMS.values() for seed in CONFIRM_SEEDS]


# ----------------------------------------------------------------------------------------------
# E8 (docs/PREREGISTRATION.md, Section 11): a better-powered retest of E7's W1. E7's two
# ArithScope-only arms, with fresh initial weights (seeds that no earlier suite used; E1, E6, and E7
# share seeds 1-5 and hence their initial weights), a new experiment name (new training batches), and
# fresh test data. The seed count comes from experiments/e8_power.py (results/summary/e8_power.json).
E8_DATA_SEED = 20261018  # fresh: E1-E5 20261014, post-hoc diagnosis 20261015, E6 20261016, E7 20261017
E8_SEEDS = tuple(range(6, 27))  # 21 seeds, the n of experiments/e8_power.py; seed 0 is for pilots, 1-5 are used
E8_ARMS: dict[str, dict] = {k: E7_ARMS[k] for k in ("reference", "arith_window")}


def e8_arm(spec: RunSpec) -> str:
    """The E8 arm a run belongs to."""
    arm = e7_arm(spec)
    if arm not in E8_ARMS:
        raise ValueError(f"not an E8 arm: {spec.filename}")
    return arm


def e8_power_retest() -> list[RunSpec]:
    return [RunSpec(experiment=f"e8_p{period}", task={"kind": "periodic", "periods": [period]}, seed=seed,
                    eval_lengths=E1_EVAL, eval_tokens_long=E6_EVAL_TOKENS_LONG, data_seed=E8_DATA_SEED,
                    **overrides, **_hp("periodic", overrides["variant"]))
            for period in (12, 13) for overrides in E8_ARMS.values() for seed in E8_SEEDS]


SUITES = {"pilot": pilot, "pilot_controls": pilot_controls, "pilot_alias_query": pilot_alias_query,
          "pilot_alias_design": pilot_alias_design, "pilot_mqar_easy": pilot_mqar_easy,
          "e1_periodic": e1_periodic, "e2_alias": e2_alias,
          "e3_tagged": e3_tagged, "e5_mixed": e5_mixed, "pilot_e6": pilot_e6, "e6_calibration": e6_calibration,
          "e7_floor": e7_floor, "e8_power_retest": e8_power_retest}
