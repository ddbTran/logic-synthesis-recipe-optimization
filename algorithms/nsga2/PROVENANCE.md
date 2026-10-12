# NSGA-II variant and source inspection

The Pareto mode implements the discrete variant described in the supplied local
`../example/nsga2_logic_synthesis.py`, SHA-256
`b1a33ad832d067f8b21927980e1518ccdbad6d85840e7f376ad1b89346e797e3`.
It is an independent implementation: **no external source text was copied**.
The example and its directory contain no license grant; therefore its source
was not eligible for copying. Algorithmic behavior was used as the reference.

Also inspected read-only: BOiLS commit
`2dc6c310c68ae622174d6174c798da1878f8397b`, `GA/main_nsga2.py`
(SHA-256 `355e3315b6b0cec3bff9460db10f8016c9bfeebefe0607ec6a74a9433600ad8a`)
and `GA/nsga2_exp.py`
(`1b10f6e3531fa1a334d28487e01070cb517544e2b380fe557c2a67d158f5c71c`).
Its root LICENSE is Mulan PSL v2; `nsga2_exp.py` additionally carries the
Huawei 2022 BSD-style three-condition notice. Nothing from that checkout is
redistributed here. Its pymoo 0.5.0 integer SBX/polynomial-mutation variant,
integer Latin-hypercube initialization, optional reference-sequence injection,
duplicate elimination, and iMAP/multiple-design plumbing would require more
dependencies and adaptations than the discrete example. This integration must
not be described as a reproduction of BOiLS's exact operators or experiments.

## Preserved Pareto algorithmic behavior (explicit mode)

- Fixed-length ordered chromosomes, uniform random initialization; repeated
  actions and duplicate individuals are allowed.
- Two minimized objectives, pairwise fast nondominated sorting, normalized
  two-objective crowding distances with infinite boundary distances.
- Binary tournaments by rank then descending crowding; exact ties retain the
  first seeded random contestant.
- Uniform crossover: mating probability 0.9, independent 50% gene exchange.
  Independent per-gene random-reset mutation (default 0.05); reset may select
  the original allele. Both probabilities are configurable.
- Elitist parent-plus-offspring survival, front order then descending crowding.
  Stable input order breaks equal crowding and objective sorting ties, including
  degenerate fronts. Rank and crowding are recomputed on survivors.
- Run-local cache keyed by the entire ordered recipe; cache hits avoid ABC.

## Adaptations for this benchmark

Import the shared 16 actions directly; replace the example's evaluator and
reporting policies, and remove its repeated-run driver.
Every candidate cache miss calls `common.abc.evaluate_recipe` with the same
`ActionBudget` object. No persistent cache, prefix evaluation, parallelism, or budget reset.
Check one sequence invocation before ABC; stop at the first unaffordable
cache miss without truncating it.
In Pareto mode, already evaluated offspring remain in the archive if a
generation stops early.
Generation count means offspring generations **after** initialization; offspring
count is configurable independently of population size, including odd counts.
Progress labels reserve generation 0 for the original circuit’s `print_stats` QoR (empty recipe, existing
LUT mapping) in weighted mode, generation
1 for initialization, and 2 through `generations + 1` for offspring. Pareto mode
starts at 1 without adding a baseline invocation. Progress `generations` is the
last display index; `offspring_generations` retains the configured search count.

Use raw LUT count and levels; only `objective: lut_count,lut_levels` is supported.
`optimization_mode: pareto` retains the two-objective NSGA-II search.
Maintain the nondominated archive over all successful evaluations, preserving
distinct recipes even when QoR is identical. Select the CSV representative by
lexicographic `(LUT count, LUT levels, recipe tuple)` across that archive.
The example reported a final-population front and weighted best; these reporting
policies are deliberately replaced. No scalar value affects Pareto selection.

## Default weighted sum variant

The default `optimization_mode: weighted_sum` selects a scalar-objective elitist
genetic algorithm using the same initialization, binary tournament sampling, uniform
crossover, random-reset mutation and generation convention. It is not unchanged
multi-objective NSGA-II. Both tournament winners and parent-plus-offspring
survivors minimize

```
weight_area * (lut_count / baseline_lut_count)
  + weight_delay * (lut_levels / baseline_lut_levels)
```

Equal fitness is resolved by recipe tuple, without a secondary LUT/level
preference. Distinct individuals with equal keys retain stable input order.
Weights default to 0.5/0.5 in both modes and must be finite, nonnegative and sum
to one within absolute tolerance 1e-9; they are validated before ABC launches.

Weighted mode first evaluates the original circuit with an empty recipe, using
a separate one-invocation `ActionBudget` with the same ABC binary, LUT K and
timeout as candidates. Both measured baseline metrics must be positive. Its
invocation is recorded as `baseline_tool_calls`, excluded from candidate
`tool_calls` and the configured budget, including failures and timeouts.
A failed baseline still aborts the search. Generation 0 displays zero candidate
calls, including when `max_tool_calls` is zero.
The baseline is a normalization reference, not a search candidate. Cache hits
remain free. Default candidate proposals can number 40 + 40 * 99 = 4000, but
all 4000 candidate invocations fit the 4000-call budget if recipes are unique,
regardless of recipe length. Initialization candidates remain budgeted.
The first unaffordable candidate attempt is recorded and stops the run with
`budget_exhausted`; the budget and generation convention are never increased.

Weighted mode tracks the scalar best across all measured candidates, including
partial generations. It does not compute or report a Pareto archive. The
`pareto` list in `result.json` is empty for compatibility; a separate `best.json`
replaces `pareto.json`. JSON and CSV identify the mode, weights, baseline QoR,
weighted fitness, best recipe and actual calls. `successful_evaluations` counts
candidate successes; `baseline_evaluations` separately counts successful
baseline measurements. `candidate_attempts` includes cache hits and the first
rejected candidate if budget exhaustion interrupts a run. Progress events use
the existing flushed logger, include mode and baseline QoR, and show scalar
fitness rather than Pareto size in weighted mode.

CLI selection (all existing overrides remain supported):

```
.venv/bin/python run.py --algorithm nsga2 --circuit <circuit.aig> \
  --optimization-mode weighted_sum --weight-area 0.8 --weight-delay 0.2
```

Omitting `--optimization-mode` uses the config mode, defaulting to `weighted_sum`.
Selecting `pareto` in the config or CLI retains the original two-objective
search and incurs no baseline invocation. Older config files without the new
fields inherit the Weighted Sum default. The default weights are provisional;
set both weights to match the exact normalized objective of the research
algorithm being compared. No particular comparison algorithm is assumed.

## Reproducible experiment directories

The runner resolves defaults and CLI overrides before opening a run directory:

```
results/nsga2/<benchmark>/<circuit-stem>/pop40_off40_gen99_len20/seed_<seed>/
```

EPFL inputs use `epfl/<circuit>`; converted ISCAS85 inputs use
`iscas85/<circuit>`. Other inputs use their parent directory name as the benchmark.
Only the declared varying parameters (population, offspring count, generations,
recipe length) appear in the config label. Seed has its own final directory.
Other settings are held fixed during data collection and stored in `config.json`;
they do not alter the directory name. Rerunning the same declared parameters and
seed replaces the known artifacts in that directory. No hashes, random IDs or
timestamps appear in directory names.

`config.json` stores the complete resolved config including seed, metric,
circuit identity/path/checksum, and requested/resolved ABC executable identity
and checksum. Circuit/executable content hashes remain provenance metadata;
replacing their bytes at the same path overwrites the same experiment directory.
Older directory layouts are preserved.

At CLI run start, only `config.json`, `result.json`, `best.csv`, `best.json`,
`pareto.json` and `progress.jsonl` in that seed directory are replaced. Other
files, seeds and configurations are preserved. Directory symlinks and artifact
directories are rejected rather than traversed or recursively removed. Thus an
interrupted rerun has no stale successful result, and a failed rerun writes no
successful CSV. Existing random-name result directories are left untouched.
Runs remain serial; simultaneous writers to the same experiment are unsupported.

Library callers may use `describe_experiment` and `prepare_run` before calling
`run`, then pass the prepared directory to `write_results`. Calling
`write_results` with only a root also uses the deterministic identity, replacing
known outputs at write time. Finalization preserves the CLI's live progress log.
The existing flushed logger records algorithm, suite-qualified circuit,
effective config and results directory in JSONL. Terminal output prints the
circuit path only at startup and the results path only after artifacts are written.

Ordinary budget exhaustion is a labeled partial run, with genuine measured best
and archive if available; insufficient budget for one recipe writes no CSV.
Any evaluation failure aborts the run, retains diagnostics and partial archive,
and writes no successful CSV even if earlier candidates succeeded. Uncertain
invocation counts remain JSON null and stop the run. Each launched sequence costs
one tool call for a candidate, including failure or timeout. The empty baseline
is counted separately and excluded from the candidate budget. Recipe length and
bundled commands do not affect the budget. Action completions are diagnostics.

Python `Config` defaults are seed 0, population 40, offspring 40, 99 offspring generations,
recipe length 20, budget 4000, and 6-LUT mapping. Initialization plus offspring
allows 40 + 40 * 99 = 4000 evaluation attempts; cache hits launch no ABC and do
not consume calls. The runner streams generation updates to the terminal; `progress.jsonl`
retains configuration, baseline, best improvements, generations, checkpoints
and final status. Checkpoints use the shared scheduler and default interval of
200 launched sequence calls. The [shared progress implementation](../../common/progress.py)
defines threshold grouping and final deduplication; counting and Pareto semantics
are described above. The `evaluations` field aliases candidate-only
`successful_evaluations`; cache hits and failures are excluded. Scalar checkpoint
fields are null in Pareto mode; its archive and legacy representative remain.

Configuration is JSON syntax (a YAML subset), loaded with Python's standard
`json` module; general YAML syntax is intentionally unsupported. Fixed seed,
configuration, input bytes and ABC executable give deterministic search and
selection; runtimes naturally vary while experiment directory names are stable.
Results include source-reference, engine, evaluator, circuit and ABC binary hashes, Python
version, full configuration and action pool for comparison provenance.
