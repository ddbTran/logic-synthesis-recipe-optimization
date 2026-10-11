# BOiLS source and experimental alignment

Primary implementation reference: the existing local
`/home/tandat_tran/work/research/HEBO/BOiLS/`, HEBO commit
`ee6112d39d1a9e9703fecaf9057193e1ec9dae72`. No HEBO files were modified.

Inspected files:

- `core/algos/bo/boils/multi_boils_exp_2.sh`: experimental settings below.
- `core/algos/bo/boils/multiseq_boils_exp.sh`: corroborating settings.
- `core/algos/bo/boils/multi_boils_exp_transf.sh` and
  `test/test_scripts/test_multi_boils_exp.sh`: different embedding/test variants,
  not the SSK baseline configuration.
- `core/algos/bo/boils/{main_multi_boils,multi_boils_exp}.py`: CLI and the
  sequential suggest/evaluate/observe loop, including maximum radius = length.
- `resources/casmopolitan/bo/optimizer.py`: Gaussian rank targets, LHS,
  complete region histories, and guided restarts.
- `resources/casmopolitan/bo/localbo_cat.py`: standardization, augmented EI,
  three local-search walks, trust-region adaptation.
- `resources/casmopolitan/bo/localbo_utils.py`: exact GP, Adam training,
  initialization, and random-neighbor acquisition search.
- `resources/casmopolitan/bo/{seq_kernel_fast,kernels}.py`: normalized
  fixed-order SSK and the auxiliary transformed-overlap kernel.

Adapted modules retain applicable Huawei 2021/2022 copyright notices and
BSD-3-Clause terms in LICENSE. No upstream infrastructure is required.

## Original, previous adaptation, and corrected defaults

| Setting | Local upstream SSK experiment | Previous adaptation | Corrected adaptation |
| --- | --- | --- | --- |
| Total recipe trials, including initialization | 200 | Explicit budget required | 200; CLI 100 for preliminary runs |
| Initialization / recipe length | 20 / 20 | 20 / 20 | 20 / 20 |
| Main kernel | Normalized SSK, order 3 | Same fixed-order kernel via explicit features | Preserved; DP equivalence tested |
| Acquisition | Augmented EI | Augmented EI | Preserved |
| Main GP training | 500 Adam steps, lr .03 | 50 | 500, lr .03 |
| Main training data | All observations since last restart | Best + latest, capped at 128 | Entire current-region history; cap rejected |
| Targets | Gaussian maximum-tie ranks, then sample mean/std normalization | Gaussian ranks only | Both transformations |
| SSK ARD | Flag enabled but SSK has no lengthscale/ARD parameters | No ARD | Same effective behavior; metadata records distinction |
| GP outputscale / noise initialization | 1.0 / .005 | 2.75 / approximately .00474 | 1.0 / .005 |
| Outputscale / noise bounds | [.5,5] / [1e-6,.1] | Same | Preserved |
| Initial discrete radius | max(1,int(.666*length)) = 13 at length 20 | 20 | 13, derived from length |
| Success / failure tolerance | 3 / 40 | 3 / 40 | Preserved |
| Radius adjustment | int(radius*1.5), capped at length; int(radius/1.5) | Forced integer expansion progress | Upstream integer multiplication/division |
| Restart threshold | radius <= 1, tested after every observation | radius < 1 | radius <= 1 |
| Acquisition optimization | Three incumbent-started random-neighbor walks, 200 steps each | 128 random points + all incumbent neighbors; 10 steepest steps from top three | Three 200-step random-neighbor walks |
| Initial sampling | LHS in one-hot coordinates, then argmax | Independent uniform actions | LHS + argmax |
| Restart auxiliary GP | Transformed overlap, raw fitness, 400 steps | SSK, rank targets, main-step count | Transformed overlap, raw fitness, 400 steps |
| Restart candidates | LHS, min(100*length,5000) = 2000 | 128 random points | Upstream LHS count |
| Restart LCB | mean - 1.96*latent std | Included likelihood noise in std | Latent std |
| Restart initialization | Guided center, then perturb initial-radius positions with replacement actions | Uniform count of necessarily changed positions | Upstream perturbation distribution |

`candidate_pool_size` now overrides only the guided restart candidate count.
`max_training_points` is retained as a null compatibility field; a numeric
cap fails explicitly rather than silently discarding data. Reduced GP/search
iteration overrides remain available for diagnostics, but are nondefault
experimental settings and must be reported. `standardise` defaults to true;
`--no-standardise` reproduces the upstream unstandardized-rank option.

## Paper versus released code

The [BOiLS paper](https://arxiv.org/html/2111.06178), Sections III-B and IV,
uses length 20 and reports comparisons at 200 evaluations. Its pseudocode
initializes radius to the sequence length; the text describes additive radius
changes, 20 failures, and restart at zero. The released SSK experiment script
instead specifies the .666 initial factor and 40 failures; the optimizer
implements multiplier 1.5 and restart at radius <= 1. This baseline follows
those **local executable experiment settings**, not a claim of exact agreement
between the paper and its released code. The preliminary budget 100 is a
user-selected framework setting, not inferred as a universal paper default.

The paper discusses subsequences up to a maximum order, whereas the local
`FastStringKernel` computes one fixed order (default three). We retain the
local implementation's fixed-order semantics. Normalization cancels its
match-decay factor exactly, so fixing that redundant parameter at .8 does
not change kernel values. Gap decay is now learned directly with projected
Adam on [0,1], following the paper's constrained update, rather than through
the previous sigmoid reparameterization. The local kernel registers decay
constraints but accesses its parameters directly without a projected update;
we enforce valid bounds to preserve the intended constrained SSK.

## Preserved framework and intentional differences

- Shared 16 actions, ordered fixed-length recipes with repetitions, original
  circuit input for every complete evaluation, ABC LUT mapping, original
  circuit baseline, and normalized weighted fitness remain unchanged. The
  upstream action alphabet and resyn2 normalization are deliberately not used.
- Total budget counts **uncached complete recipe evaluations**, including
  initialization and restarts. One baseline invocation is separate, as in
  the shared scalar engines. GP fitting, EI queries and restart searches
  never call the evaluator or consume this budget. Unlike upstream's fixed
  number of suggestions, cache hits do not use a trial or add duplicate GP
  observations. Exhaustive finite spaces terminate explicitly.
- Uses exact dense Cholesky inference in PyTorch float64 on CPU with one
  thread instead of GPyTorch float32/CUDA. The feature map is mathematically
  equivalent to upstream's SSK recurrence. Earlier documentation incorrectly
  alleged a decay-mask bug: the upstream exponents outside the strict upper
  triangle are initialized to **one**, so zero entries stay zero. That claim
  is withdrawn; numerical tests check the actual decay construction and DP.
- Main GP refits start from the same deterministic hyperparameters, as
  upstream's sequential `hypers={}` path does. Marginal likelihood uses the
  per-observation scaling of GPyTorch; an added diagonal jitter of 1e-8 and
  a 1e-12 posterior variance floor stabilize dense CPU inference. Singleton
  standardized targets use zero variance instead of undefined sample std.
- Acquisition walks start at the incumbent, matching the released code
  (the paper describes random starts). Walks may traverse cached recipes to
  optimize EI, but return the best unseen visited candidate; if endpoints
  are cached, this prevents cycling without spending recipe calls. A
  degenerate walk uses EI over unseen incumbent neighbors; exhaustion there
  triggers a guided restart. These duplicate-handling safeguards can change
  selection relative to upstream repeat evaluations.
- Invalid-neighbor retries are bounded at 100. This fixes upstream's loop
  whose exhausted retry counter can otherwise continue indefinitely. The
  accepted neighbor's acquisition score is updated immediately, avoiding an
  upstream stale endpoint score. These are correctness safeguards, not a
  smaller search budget. Three walks each still attempt 200 valid steps.
- Private Python and NumPy RNGs are seeded without mutating caller RNG state.
  Runs are reproducible in this environment, not bitwise identical to upstream.
- There is no silent numerical fallback from the main SSK GP to an overlap
  kernel. Numerical failures stop with diagnostics and retained best-so-far.
  The overlap GP is used explicitly only for upstream-style guided restarts.
- Sequential weighted-sum execution only; no batch fantasies, embeddings,
  CUDA, checkpoint resume, or reproduction of the paper's numerical results.

## Scientific and performance limits

Full-history, 500-step exact GP fitting is intentionally more expensive than
the old capped adaptation. Feature construction scales with the number of
order-three position combinations and factorization is cubic in current
region history. There is no CPU-driven truncation or reduced default search
budget. Configurable short recipes reduce SSK order to length; orders above
three are unsupported. Those settings are outside the default length-20 study.

This remains an adapted code-based baseline: the paper/code differences and
framework/duplicate/numerical differences above must accompany comparisons.
A single small-circuit smoke test is execution evidence, not evidence of
relative optimization quality. Use matched budgets and multiple fixed seeds
for QoR studies and report the serialized effective configuration.
