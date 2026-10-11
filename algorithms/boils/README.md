# BOiLS baseline

Run from the repository root with Python 3.11 and the shared requirements
(`torch==2.7.1+cpu`, NumPy; no additional dependencies):

```sh
.venv/bin/python run.py --algorithm boils \
  --circuit benchmarks/converted/iscas85/c17.aig \
  --recipe-length 20 --seed 0 --max-tool-calls 200 \
  --weight-area 0.5 --weight-delay 0.5 --lut-k 6
```

The default recipe budget is **200**, including the 20 initial evaluations.
For preliminary runs, use `--max-tool-calls 100`. Default settings are length
20, normalized order-three SSK, augmented EI, 500 GP training steps, full
current-region training history, target rank transformation followed by
standardization, and three 200-step acquisition walks. Initial radius is
`max(1,int(.666*recipe_length))`, or 13 at length 20. The reference experiment
uses 40 failures / three successes and multiplier 1.5 for adaptation.

These settings follow the local HEBO SSK experiment scripts. The paper and
released code have trust-region differences; see the full original/previous/
corrected comparison and intentional deviations in [PROVENANCE.md](PROVENANCE.md).

All config fields can be supplied through `--config` using JSON-subset YAML.
Numerical/search fields also accept hyphenated CLI options; target
standardization accepts `--standardise` / `--no-standardise`.
`candidate_pool_size=null` means the upstream restart candidate count
`min(100*recipe_length,5000)`. `tr_initial_radius=null` derives the radius from
length. `max_training_points` must remain null: old numeric truncation settings
are rejected with an explicit migration error. Reduced iteration overrides
are useful for diagnostics but must be reported as experimental deviations.
Only scalar `weighted_sum` optimization is supported.

Recipes are ordered fixed-length lists over the shared 16 actions, with
repetitions allowed. Every candidate starts from the original circuit and
uses the shared ABC LUT mapping flow and normalized `weighted_fitness()`.
The budget counts complete **uncached** candidate recipe invocations,
including failures once ABC starts. Cache hits, GP training, acquisition
queries and restart searches consume no recipe calls. Original-circuit
baseline measurement is separately accounted as `baseline_tool_calls=1`,
outside the recipe budget, following the shared scalar engines.

The shared run-local `RecipeEvaluator` handles caching. Normal proposals
exclude already evaluated recipes; any cache hits do not add duplicate GP
observations or advance region counters. A fully enumerated finite space
ends with `search_space_exhausted`; budget termination uses `budget_exhausted`.
ABC errors/timeouts and surrogate errors stop with diagnostics and retain
`best_so_far`; `best` becomes null and no success CSV is written.

Output follows `results/boils/<suite>/<circuit>/len<L>/seed_<S>/` with
`config.json`, `result.json`, `best.json`, `best.csv`, and `progress.jsonl`.
Reruns at that path replace known artifacts. Archive previous results before
changing other hyperparameters. Results contain observed recipes/fitnesses,
region/restart/fit counters, effective config, kernel/ARD/initial-radius
metadata, CPU/dtype/thread settings, upstream commit and source/tool hashes.
Private Python and NumPy RNGs are seeded; training has deterministic initial
parameters. The engine temporarily uses one CPU thread and restores the
previous setting. Full-history, 500-step fitting can be expensive; no speed
cap is applied. This is a sequential, nonresumable code-based baseline.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -p test_boils.py -v
.venv/bin/python -m unittest discover -s tests -v
```

The focused tests cover upstream SSK recurrence equivalence, independent
subsequence enumeration, GP fitting/prediction, target transformations and
hyperparameters, acquisition and region/restart behavior, full histories
above 128 observations, deterministic initialization, failure retention,
CLI overrides and JSON/CSV/progress serialization.

The 200-call accounting test uses real initialization, acquisition walks,
region updates and caching with GP/ABC spies: 200 uncached recipes plus 199
injected cache hits produce exactly 200 recipe calls and one baseline call.
It verifies 180 model refits, each requesting 500 steps on every observation
in the current region (20 through 199), without duplicate training rows.
This test verifies accounting, not numerical GP performance over 200 trials.

Real `yosys-abc` smoke: c17, length 20, seed 124, budget 24, all default
optimizer settings. Four numerical 500-step GP fits and three 200-step walks
per acquisition produced 24 successful recipe evaluations plus one baseline
call, zero cache hits, and `budget_exhausted`. Best QoR was 2 LUTs / 1 level,
normalized fitness 1.0. This checks execution rather than optimization quality.
Repeated short real-ABC CLI runs also verify identical observations/results.

After alignment, the focused suite passed all 15 tests. The complete
regression suite ran 90 tests: 89 passed and the existing CUDA hardware test
was skipped. Python 3.11 and `torch==2.7.1+cpu` were used. The full 200-trial
numerical real-ABC optimization was not run; the 200-call test substitutes
GP/ABC spies, while the 24-trial smoke exercises actual numerical training
and ABC with the aligned defaults.
