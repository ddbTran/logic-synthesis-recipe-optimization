# Logic synthesis recipe optimization benchmark

Minimal experimental Python repository for comparable QoR measurements in a
research paper. All nine algorithms, including AST-GWO, are comparison baselines.
NSGA-II is integrated for reproducible LUT QoR collection; the other eight
engines remain placeholders. Only small smoke runs have been performed.
Benchmark environment smoke checks are documented in [benchmarks/README.md](benchmarks/README.md).

## Layout and status

```text
algorithms/{ast_gwo,drills,nsga2,erl_ls,boils,flowtune,cb_evo,rl4ls,greedy}/
common/actions.py   # implemented: action indices and default budget
common/abc.py       # implemented: serial ABC evaluation and typed QoR parsing
benchmarks/        # pinned EPFL/ISPRAS submodules; 11 ISCAS85 AIG conversions and validation notes
results/           # ignored smoke evidence and future experiment CSV outputs
run.py             # NSGA-II runner; all other engines report not implemented
requirements.txt   # standard library only, including NSGA-II; Python >= 3.10
```

NSGA-II has `engine.py` and `config.yaml`. Empty algorithm directories contain
`.gitkeep`. Each future algorithm gets `engine.py` and
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

One call is one executed pool action. Each bundled GIA entry costs one call.
A recipe of length L costs L calls each time ABC executes it. Replaying prefixes
of lengths 1 through L costs L(L+1)/2, not L. A cache hit that executes no actions
costs zero. Process invocations and full-recipe evaluations are separate counts.
Circuit loading, `strash`, measurement mapping, and statistics are not pool calls.
Feature extraction or training that executes pool actions must share the run's
budget too.

The serial evaluator checks the whole recipe against the remaining budget before
launch; it rejects an oversized recipe instead of truncating it. Successful calls
report per-evaluation and cumulative action counts. An error/timeout preserves raw
output and confirmed completions as a lower bound. If the exact count is uncertain,
`budget.tool_calls` becomes `None` and subsequent evaluations are refused: abort
that run and do not publish an exact action count or successful CSV row. Never
reset the budget to continue the same failed run. Parallel accounting is not
implemented.

## ABC and QoR

`evaluate_recipe(circuit, recipe, budget=..., metric=..., ...)` uses ABC's batch
script mode (`-s -f`), with initialization aliases disabled. It loads the original
input, runs `strash`, applies action indices, and calls `print_stats`. Completion
markers detect interrupted scripts even when ABC returns exit code zero. It
handles missing executables, command failures, malformed metrics and timeouts,
and returns raw output with measurements and runtime. No shell is invoked.

For a run using the 4,000-action budget, call the existing API from the project
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
print(result["tool_calls"])       # Actions in this evaluation (3 here).
print(result["total_tool_calls"]) # Cumulative executed actions in this run.
```

`timeout` defaults to 60 seconds; `None` disables the deadline. Change
`abc_binary` to an executable name or path when needed. The result also includes
the recipe, raw ABC output, runtime, and separate process/evaluation counts.
Input loading and `if -K 6` mapping cost zero pool calls; each GIA bundle costs
one. `BudgetExceeded` rejects an oversized recipe before launching ABC.
`ABCError` reports execution/parsing failures with raw output and confirmed
completions; missing input files raise `FileNotFoundError`. Failed evaluations
return no QoR. If a timeout occurs after all action completion markers, their
exact count is retained; otherwise the run budget becomes uncertain as described
above. No candidate caching or fitness/reward calculation is performed.

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

Defaults: seed 0, recipe length 20, population 30, offspring count 30,
50 offspring generations after initialization, crossover probability 0.9,
per-gene mutation probability 0.05, LUT K=6, timeout 60 seconds,
`yosys-abc`, and **4000 executed pool actions**. CLI options override
[config.yaml](algorithms/nsga2/config.yaml); `--config PATH` accepts JSON syntax,
a YAML subset parsed by the standard library (no PyYAML dependency). Unknown
keys and invalid settings fail before evaluation. `--timeout none` (or JSON
`null`) disables the deadline. `--objective lut_count,lut_levels` is the only
supported objective; scalar/ASIC/AIG objectives are rejected explicitly.

Ordered full recipes, including repeats, are cached only within a run. Every
cache miss uses the shared evaluator and the same budget object; each actually
executed length-L recipe costs L actions. Cache hits launch no ABC and cost zero.
Stop at the first unaffordable cache miss without truncating the recipe.
Fixed seed/config/input/ABC bytes give reproducible search and best selection.

Each invocation writes a unique `results/nsga2-*/` directory:

- `result.json`: status, best solution, Pareto archive, cumulative `tool_calls`,
  runtime, successful evaluation/cache counts, completed generations, full config,
  provenance hashes (input, executable, engine, evaluator, source reference),
  Python version, action pool and failure diagnostics.
- `pareto.json`: separate nondominated archive over all successful evaluations,
  including distinct recipes with identical QoR and partial-generation candidates.
- `best.csv`: one measured representative chosen by lexicographic
  **(LUT count, LUT levels, recipe indices)**. `best_recipe` is a CSV-quoted JSON
  action-index list; config and provenance are JSON fields too.

`completed` and `budget_exhausted` are distinct statuses; budget exhaustion may
produce a measured partial-run CSV. Zero evaluated candidates produce no CSV and
exit 1. Any evaluation failure aborts, writes diagnostic JSON and partial archive,
sets best to null, writes no CSV, and exits 1. Uncertain counts remain JSON null.
No fake objective values, budget reset or persistent caches are used. Runtime and
unique output paths vary between otherwise reproducible runs. Results are ignored
by git. Other engines still print `not implemented`, exit 2, and execute no ABC.

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
| Greedy | [DRiLLS greedy](https://github.com/scale-lab/DRiLLS/tree/master/baseline/greedy): `baseline/greedy/greedy.py` | PyYAML, joblib, ABC; decide tie-breaking, stopping, and budget policy. |

The supplied `../example/drill_logic_synthesis.py` also uses the exact shared pool
and ABC LUT parsing, but explicitly substitutes linear-policy REINFORCE for
DRiLLS's A2C. It is a useful utility reference, not an equivalent original DRiLLS
implementation. Neither example source is copied or executed. The NSGA-II adaptation adds the
shared executed-action budget missing from the example.

Before integrating another engine, select its faithful source/variant, QoR
track and objective (including Pareto reporting), LUT K or ASIC library/constraints,
circuit suite, recipe lengths, seeds, per-algorithm hyperparameters, and policies
for training, cache hits and failures. Preserve core algorithms and document the
required shared-pool/evaluator adaptation. Confirm source licenses before copying.
