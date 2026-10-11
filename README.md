# Logic synthesis recipe optimization benchmark

Minimal experimental Python repository for comparable QoR measurements in a
research paper. All nine algorithms, including AST-GWO, are comparison baselines.
No algorithms are integrated and no optimization experiments have been run.
Benchmark environment smoke checks are documented in [benchmarks/README.md](benchmarks/README.md).

## Layout and status

```text
algorithms/{ast_gwo,drills,nsga2,erl_ls,boils,flowtune,cb_evo,rl4ls,greedy}/
common/actions.py   # implemented: action indices and default budget
common/abc.py       # implemented: serial ABC evaluation and typed QoR parsing
benchmarks/        # pinned EPFL/ISCAS85 submodules; c17 conversion and validation notes
results/           # ignored smoke evidence and future experiment CSV outputs
run.py             # argparse entry point; all engines report not implemented
requirements.txt   # standard library only; Python >= 3.10
```

Empty directories contain `.gitkeep`. Each future algorithm gets `engine.py` and
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
provenance limitations. Validation of the remaining circuits is still pending.

## Runner and future results

From the project directory:

```bash
python run.py --help
python run.py --algorithm drills --circuit benchmarks/example.aig
python run.py --algorithm nsga2 --circuit benchmarks/example.aig
python run.py --algorithm ast_gwo --circuit benchmarks/example.aig
# Future BOiLS use requires a selected --max-tool-calls value.
```

The example circuit is a placeholder path. Every algorithm currently prints
`not implemented` to stderr and exits 2, without executing ABC or writing results.
`--seed` and `--max-tool-calls` are reserved for integration; they do not enable an
engine. No ML packages need to be installed for this skeleton.

Future numerical outputs belong in `results/*.csv`. The planned columns are
`algorithm,circuit,seed,best_recipe,metric,lut_k,aig_nodes,aig_levels,lut_count,lut_levels,asic_area,asic_delay,tool_calls,runtime_seconds`.
Store `best_recipe` as a CSV-quoted JSON list of action indices, leave inapplicable
metrics blank, and record mapping/constraint units and tool/source revisions with
the experiment. No experiment CSV writer or fabricated data is included yet.

## Inspected integration sources

Local sibling checkouts were inspected read-only. Dependencies below describe
those sources, not installed requirements of this repository.

| Planned baseline | Source and main entry points | Dependencies / integration notes |
| --- | --- | --- |
| AST-GWO | [ast-gwo-aicas2026](https://github.com/ddbTran/ast-gwo-aicas2026): `scripts/run_single.sh`, `algorithms/ast_gwo/ast_gwo.py`, `utils.py`, `evaluator.py` | NumPy; pandas/matplotlib for outputs; ABC LUT mapping. Existing opcode dictionary differs from this required pool. |
| DRiLLS | [DRiLLS](https://github.com/scale-lab/DRiLLS): `drills.py`, `drills/model.py`, `fpga_session.py`, `scl_session.py` | TensorFlow 1.12, NumPy, PyYAML, ABC/Yosys; documented Python 3.6 environment. A2C; separate FPGA and ASIC sessions. |
| NSGA-II | [BOiLS checkout](https://github.com/SZU-AdvTech-2023/050-BOiLS-Bayesian-Optimisation-for-Logic-Synthesis): `GA/main_nsga2.py`, `GA/nsga2_exp.py` | pymoo 0.5.0, NumPy, pandas, joblib; this fork evaluates through iMAP. Local `../example/nsga2_logic_synthesis.py` is an alternative standard-library ABC implementation with Pareto selection and caching. Choose source/variation operators before porting. |
| ERL-LS | [ERL-LS](https://github.com/Intelligent-Computing-Research-Group/ERL-LS): `ERL-abcRL/testReinforce_aig.py`, `reinforce_aig.py`, `env_aig.py` | PyTorch, DGL, NumPy, Numba, `abc_py`/`mtlPy`. Imports reference absent helper modules (`perf_analysis`, `env_forLargeFile`, `convert_list2actions`, `env`); restoration and variant selection needed. `ERL-DRiLLS` has no implementation in this checkout. |
| BOiLS | [BOiLS checkout](https://github.com/SZU-AdvTech-2023/050-BOiLS-Bayesian-Optimisation-for-Logic-Synthesis): `Boils/main_boils.py`, `boils_exp.py`, `optimizer.py`, `sessions/utils_eval.py` | PyTorch, GPyTorch, sentence-transformers, NumPy, pandas, joblib; Python 3.7 documented; iMAP backend here. Budget undecided. |
| FlowTune | [FlowTune](https://github.com/Yu-Maryland/FlowTune): `FlowTune-AIG-Optimization/single_design.sh`, `src/src/base/abci/abc.c` (`ftune`) | Custom ABC C/C++ command, readline, OpenMP, build tools; optional Yosys/VTR flows. Internal actions need observable accounting. |
| CB-EVO | [CB-EVO](https://github.com/sallyliu921/CB-EVO): `src/main.cpp`, `bandit.hpp`, `evolution.hpp` | C++17, CMake, OpenMP, Eigen, ABC; optional Yosys/ccirc feature extraction. Native implementation, not a Python engine. |
| RL4LS | [RL4LS](https://github.com/Gabriel-in-Toronto/RL4LS): `RL/PPO.py`, `RL/Env.py`, `RL/Feature_extractor.py` | Stable Baselines3, PyTorch, Gym, NumPy, PyYAML, icecream, ABC, modified ccirc. `PPO.py` constructs A2C despite importing PPO; resolve source inconsistency. Modified ccirc requires obtaining the authors' implementation. |
| Greedy | [DRiLLS greedy](https://github.com/scale-lab/DRiLLS/tree/master/baseline/greedy): `baseline/greedy/greedy.py` | PyYAML, joblib, ABC; decide tie-breaking, stopping, and budget policy. |

The supplied `../example/drill_logic_synthesis.py` also uses the exact shared pool
and ABC LUT parsing, but explicitly substitutes linear-policy REINFORCE for
DRiLLS's A2C. It is a useful utility reference, not an equivalent original DRiLLS
implementation. Neither example is copied or executed. Their generation/episode
limits do not enforce this benchmark's executed-action budget.

Before integrating the first engine, select its faithful source/variant, QoR
track and objective (including Pareto reporting), LUT K or ASIC library/constraints,
circuit suite, recipe lengths, seeds, per-algorithm hyperparameters, and policies
for training, cache hits and failures. Preserve core algorithms and document the
required shared-pool/evaluator adaptation. Confirm source licenses before copying.
