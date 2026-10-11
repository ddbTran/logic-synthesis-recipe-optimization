# AST-GWO baseline adaptation

Inspected read-only source: local `../ast-gwo-aicas2026`, commit
`36686f1bc035d653da635c4748ecf8414e389a55` (clean at inspection).
The source README attributes this artifact to the IEEE AICAS 2026 AST-GWO paper.
No standalone license file was present in that checkout; this local adaptation
preserves attribution and does not establish redistribution permission.

Inspected `algorithms/ast_gwo/{ast_gwo.py,config.py,utils.py,evaluator.py}`,
`README.md`, `requirements.txt`, and `scripts/{run_single.sh,run_sweep.sh}`.
The source uses NumPy, ABC LUT mapping, an LRU cache and optional process-pool
population evaluation. Its single-run defaults are population 50, dimension 80,
200 iterations, LUT K=6, seed 42, k_min=6 and k_max=14. The sweep script instead
selects 50 iterations and k_min=k_max=18. This adapter uses the single-run search
hyperparameters; the constant-slope sweep variant is selectable by overrides.

## Preserved numerical mechanisms

`optimizer.py` adapts the source `Leaders` class and `AstGWO._step()` and
`_update_coefficients()` directly. It retains:

- `default_rng(seed)` initialization of independent uint8 bits with probability
  1/2, with shape `(population_size, 4 * recipe_length)`.
- Historical alpha/beta/delta selection: take each population's top three with
  `np.argsort`, merge with the previous three, then sort again. Copies own their
  bit arrays. Duplicate leaders and tied fitness values are not deduplicated;
  no recipe-based tie breaker changes the original sort behavior.
- Six full random matrices drawn in order r1_alpha, r1_beta, r1_delta,
  r2_alpha, r2_beta, r2_delta. For each leader, A=2*a*r1-a, C=2*r2,
  D=abs(C*leader-population), X=leader-A*D. Average the three X positions.
- Adaptive transfer `S=1/(1+exp(-k*(X_mean-0.5)))`; compare a seventh independent
  random matrix strictly below S to construct the next binary population.
- Initial a=2 and k=k_min, with a=2-2*t/T and
  k=k_min+(t/T)*(k_max-k_min). The source steps BEFORE updating coefficients:
  move t uses coefficients from t-1, then records coefficients for t.
- Initial population evaluation at iteration zero followed by T complete
  population moves, without crossover, mutation, elitist population insertion,
  duplicate resampling, or early convergence stopping.

The exact source equations are used even for large user-selected sigmoid slopes;
NumPy may warn about exponential overflow (the sigmoid saturates). Default
slopes do not cause this issue. Reproducibility is scoped to the recorded NumPy
version: upstream's default argsort tie behavior is intentionally preserved.

## Required framework adaptations

1. Keep the source's big-endian four-bit opcode codec. `dim` becomes
   `4 * recipe_length`, `pop_size` becomes `population_size`, and `iters` becomes
   `generations`. Replace the original opcode dictionary (including aliases and
   ifraig/dfraig) with `common.actions.TRANSFORMS` at its unchanged indices 0–15.
   Every bit pattern yields an ordered fixed-length recipe; repeats are allowed.
2. Replace the original ten-action reference script with the shared original
   circuit baseline: an empty recipe, strash and LUT mapping. Every candidate
   reloads that same original circuit. Use shared normalized `weighted_fitness`
   rather than the original unweighted sum of LUT and depth ratios. At weights
   0.5/0.5 this scales the original objective by 1/2 for a fixed reference; other
   weights deliberately change preference. The changed reference also can
   change rankings. No claim of reproducing paper results is made.
3. Reuse `RecipeEvaluator` and `common.abc` unchanged. Evaluation is serial,
   with one global candidate budget and a separate single-call baseline budget.
   One full recipe costs one launched ABC invocation. Run-local cache hits cost
   zero; duplicates within a population reuse the first result immediately.
   This replaces upstream's bounded 200,000-entry LRU and batch parallelism,
   whose pending duplicate misses could launch repeated evaluations.
4. Update best-so-far immediately after each valid evaluation using strict
   fitness improvement (first seen wins ties). Commit leader selection only
   after a complete population evaluation. Budget exhaustion preserves every
   measured valid candidate, including a partial initial population or later
   iteration. Cached recipes may still be evaluated at a full budget until an
   uncached recipe is rejected. All-cache iterations still advance the original
   finite iteration schedule; the engine does not force budget consumption.
5. Follow existing failure semantics: ABC failures/timeouts, uncertain launches,
   or invalid metrics abort with diagnostics and `status=failed`; `best` is null,
   while `best_so_far` retains previous valid measurements. No successful CSV is
   written for failed runs. Invalid individuals are not assigned fabricated
   penalties. Keyboard interruption propagates through the shared evaluator.
6. Use framework config overrides, deterministic result paths, progress JSONL,
   result/best JSON and best CSV, and tool/input/source hashes. The normal path
   varies by population, generations, recipe length and seed. Other settings
   are recorded in config.json; as with existing engines, rerunning that path
   replaces known artifacts. No dependencies beyond existing NumPy are added.

The shared recipe length 20, seed 0, candidate budget 4000, checkpoint interval
200, LUT K=6, weights 0.5/0.5, timeout 60 seconds and ABC executable defaults
match the framework metaheuristic settings. Match these explicitly across
algorithms for comparisons; finite iterations or cache reuse may finish before
the budget is spent. The normalized original baseline is excluded from the
candidate budget in every engine.

## Validation

`tests/test_ast_gwo.py` covers initialization, bit decoding, coefficient schedules,
controlled sigmoid thresholds, historical leader selection, deterministic runs,
weighted selection, free cache hits, partial iterations, errors, checkpoints,
and a small real-ABC CLI rerun with JSON/CSV/metadata assertions.
Its optional local-source test compiles the original numerical classes directly
from the sibling checkout (without executing its ABC constructor), comparing
floating-point sigmoid inputs, populations and leaders over eight steps for seeds 0, 7 and 42. This test skips
when the sibling repository is unavailable; all other tests are standalone.

Validated locally on Python 3.11 / NumPy 2.2.6 / CPU:

- Full regression suite: 104 tests, passing; one existing CUDA-only DRiLLS test
  skipped because this environment has no CUDA hardware/runtime.
- Shared evaluator smoke suite: 12 tests, passing with real ABC.
- Real Adder AST-GWO smoke: population 3, recipe length 4, generations 3,
  seed 0, budget 7. Stopped mid-iteration with `budget_exhausted`, exactly seven
  candidate calls plus one baseline call. Baseline 254 LUTs / 51 levels;
  best measured 243 LUTs / 51 levels, weighted fitness 0.9783464566929134.
  Ignored evidence: `results/ast_gwo/epfl/adder/pop3_gen3_len4/seed_0/`.

These are implementation/smoke checks, not full comparative QoR experiments.
