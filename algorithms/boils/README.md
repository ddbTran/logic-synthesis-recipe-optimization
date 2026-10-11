# BOiLS baseline

Run from the repository root with Python 3.11 and the shared requirements
(`torch==2.7.1+cpu`, NumPy; no additional dependencies):

```sh
.venv/bin/python run.py --algorithm boils \
  --circuit benchmarks/converted/iscas85/c17.aig \
  --recipe-length 20 --seed 0 --max-tool-calls 100 \
  --weight-area 0.5 --weight-delay 0.5 --lut-k 6
```

BOiLS requires an explicit `max_tool_calls`, via CLI or a JSON-subset YAML
config. All fields in `config.yaml` have corresponding hyphenated CLI
options. It supports `weighted_sum`; it does not implement Pareto selection.
Recipes are ordered fixed-length lists over the shared 16 actions, with
repetitions allowed. Every candidate starts from the original circuit and
uses the shared ABC LUT mapping flow and normalized `weighted_fitness()`.

The budget counts complete candidate recipe invocations, including failed
launches once ABC starts; cache hits are free. As with the other scalar
engines, original-circuit baseline measurement is separately accounted as
`baseline_tool_calls=1`, outside the recipe budget. The run-local cache is
shared `RecipeEvaluator`; proposals exclude previously evaluated recipes.
A fully enumerated finite search space ends with `search_space_exhausted`.
Budget termination uses `budget_exhausted`. ABC errors and timeouts stop the
run with diagnostics and retain `best_so_far`; `best` becomes null and no
success CSV is written, following the shared failure convention.

Output follows `results/boils/<suite>/<circuit>/len<L>/seed_<S>/` with
`config.json`, `result.json`, `best.json`, `best.csv`, and `progress.jsonl`.
Reruns at that path replace known artifacts, just like existing engines;
changing other hyperparameters requires archiving prior results first.
Results include all observed recipes/fitnesses, region/restart/fit counters,
effective configuration, CPU/dtype/thread settings, upstream commit, and
source/circuit/ABC fingerprints. Initialization uses a private Python RNG;
the GP has deterministic initial parameters and no random training steps.
The engine temporarily uses one CPU thread and restores the previous setting.

See [PROVENANCE.md](PROVENANCE.md) for algorithm details and deviations.
Tests: `.venv/bin/python -m unittest discover -s tests -p test_boils.py -v`.

## Validation in this checkout

Python 3.11, `torch==2.7.1+cpu`:

- `python -m unittest discover -s tests -v`: 86 tests run: 85 passed, one existing
  CUDA hardware test skipped. Includes 11 BOiLS tests covering kernel values,
  GP fitting/prediction, acquisition, region adaptation/restart, deterministic
  initialization, retained training data, fitness/best-so-far, cache/budgets,
  failure diagnostics, CLI overrides and JSON/CSV/progress serialization.
- Real `yosys-abc` CLI smoke: c17, length 20, seed 123, budget 24, default
  optimizer settings. Produced four surrogate fits, 24 successful complete
  recipe evaluations, one baseline call, zero cache hits, and
  `budget_exhausted`; best QoR was 2 LUTs / 1 level, normalized fitness 1.0.
  This checks execution and accounting, not optimization quality on a tiny
  circuit. Tests also repeat a real CLI run with acquisition enabled and
  verify identical observations and best results.
