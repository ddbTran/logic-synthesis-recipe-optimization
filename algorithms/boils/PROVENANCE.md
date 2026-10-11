# Source and adaptation

Primary reference inspected directly at
`/home/tandat_tran/work/research/HEBO/BOiLS/`, HEBO commit
`ee6112d39d1a9e9703fecaf9057193e1ec9dae72`:

- `core/algos/bo/boils/multi_boils_exp.py`: suggest/evaluate/observe integration.
- `resources/casmopolitan/bo/optimizer.py`: Gaussian rank transformation,
  initialization, region-local histories, guided restarts.
- `resources/casmopolitan/bo/localbo_cat.py`: exact GP, augmented expected
  improvement, incumbent-centered Hamming regions, success/failure counters.
- `resources/casmopolitan/bo/localbo_utils.py`: GP marginal-likelihood training,
  random categorical perturbations and multistart acquisition hill climbing.
- `resources/casmopolitan/bo/seq_kernel_fast.py`: normalized fixed-order SSK.

The surrogate and optimizer are adaptations of these mechanisms, not a
port of HEBO's environment or its experiment framework. Applicable Huawei
2021/2022 copyright notices and BSD-3-Clause terms are retained in LICENSE
and the adapted module headers. No HEBO files are changed.

## Preserved mechanisms

An exact Gaussian process models Gaussian-rank-transformed scalar fitness,
using a normalized string-subsequence kernel. Adam maximizes the Gaussian
marginal likelihood, learning gap decay, constant mean, signal scale and
bounded observation noise. Predictions use Cholesky posterior inference.
Sequential augmented EI uses the posterior mean at the incumbent as its
threshold and the upstream noise penalty. Candidates are optimized by
multistart discrete local search, restricted to a Hamming ball centered on
the best observation in the current region. After initialization, strict
relative improvements of at least 0.001 increment a success counter;
nonimprovements increment a failure counter. Three successes expand the
radius; forty failures shrink it, using multiplier 1.5. Collapse triggers
a restart, fits an auxiliary GP to previous region optima, and selects a
new initial center by lower confidence bound (mean minus 1.96 standard
deviations). The global evaluation cache and best-so-far survive restarts.

## Intentional deviations and limits

- Uses the target's action pool, fixed-length recipes, original-circuit ABC
  evaluation, LUT QoR, baseline normalization, explicit recipe budget and
  artifacts. It does not attempt to reproduce paper numerical results.
- Implements the exact SSK feature map in PyTorch instead of GPyTorch and
  pairwise dynamic programming. For subsequence `u` of order `p`, the feature
  is the sum over occurrences of `match_decay**p * gap_decay**(span-p)`.
  Normalized feature inner products give the kernel. Order defaults to 3
  and is reduced to recipe length for lengths 1/2. Orders above 3 are
  intentionally unsupported to bound the 16**p feature dimension.
- Upstream `calc_decay_matrix()` can put nonzero entries outside its intended
  strict upper triangle because `0**0 == 1`. This implementation uses actual
  ordered subsequences, without those spurious DP transitions. Kernel tests
  independently enumerate the mathematical subsequences. Normalization at
  a single fixed order cancels match decay, so it stays fixed at 0.8; gap
  decay is constrained by a sigmoid to (0,1), avoiding invalid fitted values.
- Uses float64 on CPU with one thread. Default GP training is 50 Adam steps
  (upstream `suggest` requests 500). The training set defaults to at most
  128 current-region observations, retaining the incumbent and most recent
  observations. `max_training_points` and `gp_training_steps` are configurable.
  Exact GP inference applies to that retained subset, not all historical data.
- Initializes with independent uniform recipes (default 20) instead of
  Latin-hypercube one-hot sampling. Distinct proposals avoid wasting
  optimization steps on cache hits. Sampling inside regions changes a
  uniform number of positions in [1,radius], always to different actions.
- EI search scores 128 random candidates plus all one-edit incumbent
  neighbors, then performs up to 10 steepest-ascent steps from the three
  highest-scoring starts. These bounds are configurable. It does not port
  batch fantasizing, Thompson sampling, embedding kernels or CUDA support.
- Radius is clamped to recipe length; expansion guarantees integer progress.
  Restart occurs when shrinking below one, allowing meaningful radius-one
  search and length-one experiments. Local proposal exhaustion also restarts.
  The auxiliary restart model uses SSK and rank targets, rather than the
  upstream auxiliary model's default overlap kernel and raw targets.
- No silent fallback to an overlap kernel on numerical error. A surrogate
  failure is reported with best-so-far preserved. This keeps the baseline
  genuinely GP/SSK-based throughout its run.

These CPU bounds affect search quality; compare algorithms with the same
recipe length, seed, evaluator settings, scalar weights and recipe budget,
and report the effective BOiLS hyperparameters. Large recipe lengths can
still be costly: explicit feature construction scales with the number of
order-p position combinations, and exact GP factorization is cubic in the
retained training count. This is a sequential, nonresumable baseline.
