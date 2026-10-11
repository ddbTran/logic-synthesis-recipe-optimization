# NSGA-II variant and source inspection

The integration implements the discrete variant described in the supplied local
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

## Preserved algorithmic behavior

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

Import the shared 16 actions directly; remove the example's evaluator, baseline
normalization, weighted reporting score, repeated-run driver and progress logs.
Every cache miss calls `common.abc.evaluate_recipe` with the same `ActionBudget`
object. No persistent cache, prefix evaluation, parallelism, or budget reset.
Check a full recipe before ABC; stop at the first unaffordable cache miss.
Already evaluated offspring remain in the archive if a generation stops early.
Generation count means offspring generations **after** initialization; offspring
count is configurable independently of population size, including odd counts.

Use raw LUT count and levels; only `objective: lut_count,lut_levels` is supported.
Other objectives are rejected rather than changing NSGA-II into a scalar GA.
Maintain the nondominated archive over all successful evaluations, preserving
distinct recipes even when QoR is identical. Select the CSV representative by
lexicographic `(LUT count, LUT levels, recipe tuple)` across that archive.
The example reported a final-population front and weighted best; these reporting
policies are deliberately replaced. No scalar value affects search selection.

Ordinary budget exhaustion is a labeled partial run, with genuine measured best
and archive if available; insufficient budget for one recipe writes no CSV.
Any evaluation failure aborts the run, retains diagnostics and partial archive,
and writes no successful CSV even if earlier candidates succeeded. Uncertain
action counts remain JSON null. The evaluator alone determines executed actions.

Configuration is JSON syntax (a YAML subset), loaded with Python's standard
`json` module; general YAML syntax is intentionally unsupported. Fixed seed,
configuration, input bytes and ABC executable give deterministic search and
selection; runtimes and output-directory names naturally vary. Results include
source-reference, engine, evaluator, circuit and ABC binary hashes, Python
version, full configuration and action pool for comparison provenance.
