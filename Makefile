# Canonical commands. PY can point at any interpreter with the needed extras, e.g.
#   make test PY=/path/to/env/bin/python
PY ?= python3
WORKERS ?= 4
SUITES = e1_periodic e2_alias e3_tagged e5_mixed

.PHONY: help oracle test lint theory pilots experiments e6 e7 analyze lab bench all

help:  ## list targets
	@grep -E '^[a-z0-9_]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

oracle:  ## stdlib-only exact checks (no torch needed; torch tests are skipped)
	$(PY) -m unittest discover -s tests -v

test:  ## full test suite (needs torch)
	$(PY) -m pytest

lint:  ## static checks (ruff)
	$(PY) -m ruff check arithscope experiments tests examples

theory:  ## theory figures and tables (no training)
	$(PY) experiments/theory_figures.py

pilots:  ## development pilots (dev data seed; used only for the preregistered hyperparameter rule)
	$(PY) experiments/run_suite.py pilot pilot_controls pilot_alias_query pilot_alias_design pilot_mqar_easy --workers $(WORKERS)
	$(PY) experiments/select_hparams.py results/runs/pilot results/runs/pilot_controls

experiments:  ## confirmatory suites in one worker pool (resumable; finished runs are skipped)
	$(PY) experiments/run_suite.py $(SUITES) --workers $(WORKERS)

e6:  ## E6 long-length calibration suite (docs/PREREGISTRATION.md, Section 9; resumable), then its verdicts
	$(PY) experiments/run_suite.py e6_calibration --workers $(WORKERS)
	$(PY) experiments/e6_verdicts.py

e7:  ## E7 ArithScope-only NLL floor suite (docs/PREREGISTRATION.md, Section 10; resumable), then its verdicts
	$(PY) experiments/run_suite.py e7_floor --workers $(WORKERS)
	$(PY) experiments/e7_verdicts.py

analyze:  ## regenerate every results table/figure from run JSON files
	$(PY) experiments/analyze.py

lab:  ## build the interactive explainer docs/lab/index.html from committed summaries
	$(PY) experiments/lab/build.py

bench:  ## runtime / memory overhead of the bias backends (run on an idle machine)
	$(PY) experiments/bench_overhead.py

all: test theory experiments analyze bench
