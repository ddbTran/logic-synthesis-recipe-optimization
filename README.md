# Logic synthesis recipe optimization benchmark

Minimal experimental Python repository for comparable QoR measurements in a
research paper. All nine algorithms, including AST-GWO, are comparison baselines.
NSGA-II and Sequential Forward Greedy are integrated for reproducible LUT QoR
collection; the other seven engines remain placeholders. Only small smoke runs
have been performed.
Benchmark environment smoke checks are documented in [benchmarks/README.md](benchmarks/README.md).

## Layout and status

```text
algorithms/{ast_gwo,drills,nsga2,erl_ls,boils,flowtune,cb_evo,rl4ls,greedy}/
common/actions.py   # implemented: action indices and default budget
common/abc.py       # implemented: serial ABC evaluation and typed QoR parsing
benchmarks/        # pinned EPFL/ISPRAS submodules; 11 ISCAS85 AIG conversions and validation notes
results/           # ignored smoke evidence and future experiment CSV outputs
run.py             # NSGA-II / Greedy runner; other engines report not implemented
requirements.txt   # standard library only, including NSGA-II; Python >= 3.10
```

NSGA-II and Greedy each have `engine.py` and `config.yaml`. Empty algorithm
directories contain `.gitkeep`. Each future algorithm gets `engine.py` and
`config.yaml` for its own hyperparameters when integrated. No dummy engines or
speculative configurations are supplied. Every engine must import the shared
pool and use shared evaluation; no configuration framework is needed.

## Shared actions and budget

`common/actions.py` is the authority for these exact indices:

| Index | Action | Index | Action |
| --- | --- | --- | --- |
| 0 | `rewrite` | 8 | `resub` |
| 1 | `rewrite -z` | 9 | `resub -z` |
| 2 | `rewrite -l` | 10 | `resub -l` |
| 3 | `rewrite -z -l` | 11 | `resub -z -l` |
| 4 | `refactor` | 12 | `balance` |
| 5 | `refactor -z` | 13 | `balance -l` |
| 6 | `refactor -l` | 14 | `&get -n; &dsdb; &put` |
| 7 | `refactor -z -l` | 15 | `&get -n; &sopb; &put` |

`MAX_TOOL_CALLS = 4000` is the default for RL and metaheuristic runs. BOiLS's
budget remains undecided and must be explicitly configured. The utility's
`ActionBudget(limit)` requires an explicit limit; future adapters select the
appropriate default without changing the core search algorithm.

One tool call is one launched ABC sequence (full recipe evaluation), regardless
of recipe length or how many commands an action contains. An empty-recipe
baseline is one ABC invocation, but the runner accounts for it separately and
excludes it from the candidate `tool_calls` budget. A cache hit launches no ABC
and costs zero.
Replaying L prefixes in separate evaluations costs L calls. Loading, `strash`,
transformations, mapping and statistics all share the same sequence invocation.
Action completion counts are diagnostics, not budget charges.

The serial evaluator checks one call before launch and never truncates recipes.
A launched sequence costs one call even if it fails or times out. Invalid inputs
and definite launch failures cost zero. An unexpected launch exception can leave
invocation accounting uncertain (`budget.tool_calls = None`); abort the run and
refuse further evaluations. Never reset the budget to continue a failed run.
Parallel accounting is not implemented.

## ABC and QoR

`evaluate_recipe(circuit, recipe, budget=..., metric=..., ...)` uses ABC's batch
script mode (`-s -f`), with initialization aliases disabled. It loads the original
input, runs `strash`, applies action indices, and calls `print_stats`. Completion
markers detect interrupted scripts even when ABC returns exit code zero. It
handles missing executables, command failures, malformed metrics and timeouts,
and returns raw output with measurements and runtime. No shell is invoked.

For a run using the 4,000-sequence budget, call the existing API from the project
root and reuse **the same budget object** for every candidate:

```python
from common.abc import ActionBudget, evaluate_recipe
from common.actions import MAX_TOOL_CALLS

budget = ActionBudget(MAX_TOOL_CALLS)  # BOiLS must supply its own selected limit.
result = evaluate_recipe(
    "benchmarks/epfl/random_control/ctrl.aig",
    [0, 4, 14],  # Ordered action indices 0-15; repeats and [] are supported.
    budget=budget, metric="lut", lut_k=6,
    abc_binary="yosys-abc", timeout=None,
)
print(result["metrics"])          # {"lut_count": int, "lut_levels": int}
print(result["tool_calls"])       # Sequence invocations in this evaluation (1 here).
print(result["total_tool_calls"]) # Cumulative launched sequences in this run.
```

`timeout` defaults to 60 seconds; `None` disables the deadline. Change
`abc_binary` to an executable name or path when needed. The result also includes
the recipe, raw ABC output, runtime, and separate process/evaluation counts.
Input loading, ordered actions, `if -K 6` mapping and measurement share one
invocation. `BudgetExceeded` rejects an evaluation before launch when no calls
remain. `ABCError` reports execution/parsing failures with raw output and
confirmed action completions; missing inputs raise `FileNotFoundError`.
Failed evaluations return no QoR but retain the known sequence charge, including
timeouts and interrupted actions. No candidate caching or fitness/reward
calculation is performed by this utility.

Run the focused evaluator checks without benchmark conversion or equivalence:

```bash
python tests/smoke_abc.py --evaluator-only
```

Choose the metric explicitly:

- `metric="aig"`: unmapped `and` and `lev`, returned as `aig_nodes`, `aig_levels`.
- `metric="lut", lut_k=K`: `if -K K` followed by `nd` and `lev`, returned as
  `lut_count`, `lut_levels`. K must be explicitly selected and supported by ABC.

These metrics are distinct from ASIC area/delay. ASIC mapping is **not
implemented**: select a technology library, mapping commands, timing constraints,
units, and parsing policy first. No target library has been configured or validated.
No normalized fitness/reward is imposed by the utility.

Supply an ABC-readable `.aig`, `.aiger`, `.blif`, `.bench`, or supported `.v` file;
ABC is not a general RTL compiler. The default executable is `yosys-abc`; override
`abc_binary` with an executable name or path. The inspected environment has
`/usr/bin/yosys-abc` (ABC 1.01), and its command help exposes `&dsdb` and `&sopb`.
All 16 actions and LUT-6 mapping now pass smoke checks on EPFL ctrl and converted
ISCAS85 c17; see [benchmark validation](benchmarks/README.md) for measurements and
provenance limitations. All 11 ISCAS85 AIG conversions pass source equivalence;
EPFL adder also passes read/mapping checks. Full-suite action compatibility is
still pending.

## NSGA-II runner and results

The integrated discrete NSGA-II variant uses binary rank/crowding tournaments,
uniform crossover, random-reset mutation and elitist nondominated/crowding
survival. Both LUT count and LUT levels are minimized separately. See
[provenance and adaptations](algorithms/nsga2/PROVENANCE.md) for source inspection,
licensing decisions, precise operators and deterministic tie policies. This is
an independent implementation of the supplied example's variant, not BOiLS's
integer SBX/polynomial-mutation variant. No external source was copied.

From the project directory, these small real-ABC smoke runs use existing inputs:

```bash
python run.py --algorithm nsga2 --circuit benchmarks/epfl/random_control/ctrl.aig --seed 7 --recipe-length 3 --population-size 4 --offspring-count 3 --generations 2 --max-tool-calls 30
python run.py --algorithm nsga2 --circuit benchmarks/converted/iscas85/c17.aig --seed 7 --recipe-length 3 --population-size 4 --offspring-count 3 --generations 2 --max-tool-calls 30
python tests/test_nsga2.py
python tests/smoke_abc.py --evaluator-only
```

Python `Config` defaults: seed 0, recipe length 20, population 40, offspring count 40,
99 offspring generations after initialization, crossover probability 0.9,
per-gene mutation probability 0.05, LUT K=6, timeout 60 seconds,
`yosys-abc`, and **4000 launched sequences**. CLI defaults come from [config.yaml](algorithms/nsga2/config.yaml), which may
override the Python defaults. CLI options override
[config.yaml](algorithms/nsga2/config.yaml); `--config PATH` accepts JSON syntax,
a YAML subset parsed by the standard library (no PyYAML dependency). Unknown
keys and invalid settings fail before evaluation. `--timeout none` (or JSON
`null`) disables the deadline. `--objective lut_count,lut_levels` is the only
supported objective pair. The default weighted-sum mode minimizes normalized
LUT count/levels with weights 0.5/0.5; `--optimization-mode pareto` selects
two-objective NSGA-II. ASIC/AIG objectives are rejected explicitly.

Ordered full recipes, including repeats, are cached only within a run. Every
cache miss uses the shared evaluator and the same budget object; each launched
sequence costs one tool call regardless of recipe length. Cache hits launch no ABC and cost zero.
Stop at the first cache miss with no remaining calls; never truncate recipes.
Fixed seed/config/input/ABC bytes give reproducible search and best selection.

Results use the compact deterministic layout
`results/<algorithm>/<benchmark>/<circuit>/<config>/seed_<seed>/`, for example:
`results/nsga2/epfl/adder/pop40_off40_gen99_len20/seed_7/`.
The config label contains only the declared varying parameters: population,
offspring, generations and recipe length. Seed remains in the final directory.
Other settings are held fixed for data collection and recorded in `config.json`;
they do not affect the directory name. Rerunning the same declared parameters
and seed replaces that directory's known artifacts. Other files and older-layout
directories are preserved. No hashes or random run IDs appear in directory names.
Terminal output shows the circuit path once at the beginning, generation
updates, and the results path once at the end. Generation 0 displays the measured
original circuit `print_stats` QoR in weighted mode, before any recipe actions
(empty recipe with the existing LUT mapping). Generation 1 is the initial
candidate population, followed by the configured offspring generations. Pareto
mode starts at generation 1 because it performs no baseline evaluation.
Improvement, checkpoint and summary events are recorded in JSONL without extra
terminal lines. JSONL retains complete metadata on every event.

- `config.json`: effective configuration including seed and input/tool identities.
- `progress.jsonl`: flushed configuration, baseline (weighted mode), improvement,
  generation, checkpoint and summary events.
- `result.json`: status, best/archive, counters, config, provenance and failures.
- `best.csv`: measured scalar winner in weighted mode, or the legacy lexicographic
  `(LUT count, LUT levels, recipe)` archive representative in Pareto mode.
- `best.json` (weighted mode) or `pareto.json` (Pareto mode): winner or complete
  nondominated archive, including candidates from partial generations.

`completed` and `budget_exhausted` distinguish full and partial runs. Zero
successful candidates produce no CSV and exit 1. Evaluation failures abort,
retain diagnostics/partial Pareto archive, set legacy `best` to null and write
no successful CSV. New `best_*` fields retain previously measured scalar
best-so-far even on failure. Results are ignored by git. Runtimes vary; search
and paths are reproducible. NSGA-II, its weighted-sum GA variant, and Sequential
Forward Greedy are implemented; DRiLLS, BOiLS, AST-GWO and the other engines remain
placeholders (exit 2, no ABC execution).

## Sequential Forward Greedy

Greedy is implemented in [engine.py](algorithms/greedy/engine.py), with one
[config.yaml](algorithms/greedy/config.yaml) and no extra dependencies. It is an
independent **Sequential Forward Greedy** implementation; equivalence to any
paper's Greedy baseline has not been verified.

Start with an empty recipe. At each depth, append each of the 16 shared actions
to the selected prefix, evaluate each entire candidate from the original
circuit through the shared ABC evaluator, and choose the lowest weighted fitness.
Equal fitness selects the lowest action index. Actions may repeat. Continue even
when the selected extension is worse than the preceding prefix. There is no
randomness, restart, beam, local search or evolutionary operation; seed is
retained for experiment identity. Fixed configuration, circuit and ABC bytes
produce the same search decisions.

Only `weighted_sum` is supported. The engine reuses NSGA-II's normalized formula:

```text
fitness = weight_area * (lut_count / baseline_lut_count)
        + weight_delay * (lut_levels / baseline_lut_levels)
```

Lower is better. Weights must be finite, nonnegative and sum to 1 within 1e-9.
The baseline reloads the original circuit with an empty recipe and the existing
LUT mapping. Its one invocation uses a separate budget, recorded as
`baseline_tool_calls=1` and `baseline_evaluations=1`, outside the candidate
`max_tool_calls` limit, exactly as in the current NSGA-II engine. Total
`abc_process_invocations` includes both baseline and candidates.

Each uncached candidate costs **one tool call regardless of recipe length**.
Ordered recipe tuples are cached within a fixed run; hits cost zero calls and
increase only `cache_hits` and `candidate_attempts`. Failed or invalid QoR is
never cached as a success. `evaluations` / `successful_evaluations` count only
valid, uncached candidate measurements. `candidate_attempts` also counts a
budget-rejected proposal. No evaluation exceeds the remaining candidate budget.

Defaults: recipe length 20, seed 0, max tool calls 4000, checkpoint interval 200,
mode `weighted_sum`, LUT K=6, weights 0.5/0.5, binary `yosys-abc`, timeout 60 seconds.
A complete search makes 16 evaluations per depth, so the default stops after
**320 candidate calls plus one baseline**, leaving the rest of the budget unused.
It finishes after reaching its target length; it does not spend spare calls on
additional search methods.

```bash
# Small real-ABC smoke run; not a full benchmark experiment.
python run.py --algorithm greedy --circuit benchmarks/epfl/arithmetic/adder.aig --recipe-length 2 --max-tool-calls 32 --checkpoint-interval 16 --seed 0
python -m unittest discover -s tests -p test_greedy.py -v
```

CLI overrides: `--config PATH` (JSON-subset YAML), `--recipe-length`, `--seed`,
`--max-tool-calls`, `--checkpoint-interval`, `--optimization-mode weighted_sum`,
`--lut-k`, `--weight-area`, `--weight-delay`, `--abc-binary` and `--timeout` (seconds,
or `none` to disable). Pareto mode, unknown configuration keys, and NSGA-II-only
search options are rejected before evaluation.

Results follow the shared directory policy, for example
`results/greedy/epfl/adder/len2/seed_0/`. Only recipe length and seed vary this
path; other settings are recorded in the configuration. A rerun replaces the
known artifacts in that directory and preserves unrelated files:

- `config.json`: complete effective configuration, seed, circuit and ABC identities.
- `progress.jsonl`: configuration, baseline, improvement, checkpoint, step and
  summary events, including counters, best QoR/fitness, elapsed time and status.
  `depth`, `target_depth`, `completed_depth` and `depth_candidates_evaluated`
  describe search progress. Checkpoints use cumulative candidate tool calls.
  Terminal output shows the circuit, baseline, one line per completed depth and
  final result path; individual candidates are never logged as separate events.
- `result.json`: configuration, provenance (including input/tool hashes),
  counters, termination status and failures. `best` / `best_so_far` track the
  globally best valid candidate, including shorter prefixes and alternatives
  seen at a partial depth. `best_complete` is the best evaluated target-length
  candidate, or null. `constructed_recipe` / `constructed_solution` preserve
  the prefix selected after fully comparing all actions at each completed depth.
- `best.json` and `best.csv`: the global scalar winner. No Pareto archive is
  generated. With no valid candidate, `best.json` is null and no CSV is written.

`completed` means all target depths compared all 16 actions. `budget_exhausted`
may leave a partial depth: retain every valid candidate seen, but do not commit
that depth's selection or pad the recipe. A target-length candidate can therefore
exist while `constructed_recipe` is shorter and status is `budget_exhausted`.
ABC errors or invalid metrics abort with `failed`, diagnostics and null `best`;
previous valid measurements remain in `best_so_far`, and no successful CSV is
written. Baseline metrics must be positive integers; candidate metrics must be
nonnegative integers. Keyboard interruption propagates as in NSGA-II; old result
artifacts are removed before execution and the evaluator stops its child process.

Greedy makes irreversible prefix choices and evaluates only single-action
extensions. It cannot revisit earlier decisions or explore combinations that
initially look worse; it has no global-optimality guarantee or Pareto tradeoff
search. This is a deterministic reference method for the shared objective.

## Common convergence progress

Every `checkpoint` includes these fields (also present on other engine events):

| Field | Meaning |
| --- | --- |
| `event` | `"checkpoint"` |
| `evaluations` | Integer count of successful, non-cached candidate recipe evaluations |
| `tool_calls` | Cumulative launched candidate sequences, excluding generation 0; integer, or null on uncertain launches |
| `elapsed_seconds` | Cumulative wall-clock runtime in seconds |
| `best_fitness` | Lowest measured candidate weighted fitness so far, or null |
| `best_lut_count`, `best_lut_levels` | QoR of that same scalar winner, or null |

Weighted fitness is `weight_area * LUTs / baseline_LUTs + weight_delay * levels /
baseline_levels` (lower is better). The original circuit's empty-recipe baseline
is a normalization reference, excluded from `evaluations` and counted separately
as `baseline_evaluations`. Its invocation is tracked as `baseline_tool_calls`
and excluded from candidate `tool_calls` and `max_tool_calls`, including failure
or timeout. Generation 0 therefore displays `ABC calls=0/<budget>`.
`abc_process_invocations` includes both baseline and candidate launches. The
baseline is measured even with a zero candidate budget; baseline failure still
aborts the run. Initial candidate population evaluations remain budgeted.
Cache hits, failed evaluations and unevaluated proposals never increment
`evaluations`. `candidate_attempts` includes cache hits and a budget-rejected
candidate; `cache_hits` and process invocations are separate counters.
`result.json` retains `successful_evaluations` as an alias of `evaluations`.

`checkpoint_interval` defaults to **200 sequence calls**, configurable in JSON/YAML or
with `--checkpoint-interval`. `common.progress.CheckpointScheduler` is reusable
by future engines: supply a snapshot after updating measured best/archive and
call once at termination with `final=True`. Scheduling is independent of
generations. The evaluator returns only completed full recipes: no intermediate
QoR inside a recipe is measured. At cumulative call 200 the snapshot records
`tool_calls: 200` and `checkpoint_thresholds: [200]`. Each launched recipe
increments calls by one, so NSGA-II normally crosses one threshold at a time.
The reusable scheduler can group multiple crossed thresholds into one measured
snapshot if an adapter reports several calls together. Every threshold is
represented once; use the actual **`tool_calls` as the convergence X-axis**,
never generations or individual action counts. No interpolation within recipes
is justified.

A final checkpoint is emitted when counters or measured best/archive differ
from the last checkpoint, including early termination and cache-only progress.
Elapsed time, generation and status changes alone do not create duplicate
checkpoints; the final `summary` always carries termination status and runtime.
Failed recipes yield no new QoR. Known final sequence counts can cross thresholds;
uncertain counts remain null and cannot define additional thresholds.

In Pareto mode all three scalar `best_*` fields are null. Checkpoints retain
`pareto_size` and the full `pareto` archive; legacy `best` remains a CSV
representative, not a scalar convergence criterion. Compare archives with an
explicit multi-objective metric rather than treating this representative as a
scalar optimum.

Sequence-call accounting uses one invocation per sequence. The runner excludes
the generation-0 measurement from the candidate budget.
Commands, LUT mapping, fitness and search operators are unchanged. The earlier
action-counting revision used a different budget unit; do not mix its results
with sequence-based curves. `checkpoint_interval` is recorded in
`config.json` and held fixed alongside the other settings outside the path label.

Validation: `python -m unittest discover -s tests -v` and
`python tests/smoke_abc.py --evaluator-only`; the small CLI checks run both modes
on ctrl/c17 and verify artifacts. `python tests/smoke_abc.py` additionally checks
c17 conversion, all 16 actions and adder baseline mapping.

## Inspected integration sources

Local sibling checkouts were inspected read-only. Dependencies below describe
those sources, not installed requirements of this repository.

| Planned baseline | Source and main entry points | Dependencies / integration notes |
| --- | --- | --- |
| AST-GWO | [ast-gwo-aicas2026](https://github.com/ddbTran/ast-gwo-aicas2026): `scripts/run_single.sh`, `algorithms/ast_gwo/ast_gwo.py`, `utils.py`, `evaluator.py` | NumPy; pandas/matplotlib for outputs; ABC LUT mapping. Existing opcode dictionary differs from this required pool. |
| DRiLLS | [DRiLLS](https://github.com/scale-lab/DRiLLS): `drills.py`, `drills/model.py`, `fpga_session.py`, `scl_session.py` | TensorFlow 1.12, NumPy, PyYAML, ABC/Yosys; documented Python 3.6 environment. A2C; separate FPGA and ASIC sessions. |
| NSGA-II | [BOiLS checkout](https://github.com/SZU-AdvTech-2023/050-BOiLS-Bayesian-Optimisation-for-Logic-Synthesis): `GA/main_nsga2.py`, `GA/nsga2_exp.py` | pymoo 0.5.0, NumPy, pandas, joblib; this fork evaluates through iMAP. The integrated engine independently implements the supplied example's discrete uniform-crossover/random-reset variant with shared ABC evaluation; see provenance above. |
| ERL-LS | [ERL-LS](https://github.com/Intelligent-Computing-Research-Group/ERL-LS): `ERL-abcRL/testReinforce_aig.py`, `reinforce_aig.py`, `env_aig.py` | PyTorch, DGL, NumPy, Numba, `abc_py`/`mtlPy`. Imports reference absent helper modules (`perf_analysis`, `env_forLargeFile`, `convert_list2actions`, `env`); restoration and variant selection needed. `ERL-DRiLLS` has no implementation in this checkout. |
| BOiLS | [BOiLS checkout](https://github.com/SZU-AdvTech-2023/050-BOiLS-Bayesian-Optimisation-for-Logic-Synthesis): `Boils/main_boils.py`, `boils_exp.py`, `optimizer.py`, `sessions/utils_eval.py` | PyTorch, GPyTorch, sentence-transformers, NumPy, pandas, joblib; Python 3.7 documented; iMAP backend here. Budget undecided. |
| FlowTune | [FlowTune](https://github.com/Yu-Maryland/FlowTune): `FlowTune-AIG-Optimization/single_design.sh`, `src/src/base/abci/abc.c` (`ftune`) | Custom ABC C/C++ command, readline, OpenMP, build tools; optional Yosys/VTR flows. Internal actions need observable accounting. |
| CB-EVO | [CB-EVO](https://github.com/sallyliu921/CB-EVO): `src/main.cpp`, `bandit.hpp`, `evolution.hpp` | C++17, CMake, OpenMP, Eigen, ABC; optional Yosys/ccirc feature extraction. Native implementation, not a Python engine. |
| RL4LS | [RL4LS](https://github.com/Gabriel-in-Toronto/RL4LS): `RL/PPO.py`, `RL/Env.py`, `RL/Feature_extractor.py` | Stable Baselines3, PyTorch, Gym, NumPy, PyYAML, icecream, ABC, modified ccirc. `PPO.py` constructs A2C despite importing PPO; resolve source inconsistency. Modified ccirc requires obtaining the authors' implementation. |
| Greedy | [DRiLLS greedy](https://github.com/scale-lab/DRiLLS/tree/master/baseline/greedy): `baseline/greedy/greedy.py` | Historical source reference only. The integrated Sequential Forward Greedy is independent and does not claim equivalence to this baseline. |

The supplied `../example/drill_logic_synthesis.py` also uses the exact shared pool
and ABC LUT parsing, but explicitly substitutes linear-policy REINFORCE for
DRiLLS's A2C. It is a useful utility reference, not an equivalent original DRiLLS
implementation. Neither example source is copied or executed. The NSGA-II adaptation adds the
shared sequence-invocation budget missing from the example.

Before integrating another engine, select its faithful source/variant, QoR
track and objective (including Pareto reporting), LUT K or ASIC library/constraints,
circuit suite, recipe lengths, seeds, per-algorithm hyperparameters, and policies
for training, cache hits and failures. Preserve core algorithms and document the
required shared-pool/evaluator adaptation. Confirm source licenses before copying.
