# Introduction

This project implements logic-synthesis recipe optimization algorithms to generate
reproducible quality-of-result (QoR) comparison data for research. The five algorithms
share benchmarks, ABC evaluation, a 16-action pool, a normalized weighted objective,
and recipe-evaluation budget accounting.

Algorithm-specific optimization mechanisms are preserved while evaluation is
standardized. The project does not claim to reproduce papers' numerical results.

## 1. Quick Start

1. Clone the repository and its pinned benchmark submodules:

   ```bash
   git clone --recurse-submodules https://github.com/ddbTran/logic-synthesis-recipe-optimization.git
   cd logic-synthesis-recipe-optimization
   ```

2. Install system dependencies. On Debian 12 with `sudo` access:

   ```bash
   sudo apt-get update
   sudo apt-get install -y git python3.11 python3.11-venv python3-pip yosys berkeley-abc
   python3.11 --version
   command -v yosys yosys-abc
   ```

   Other systems need Python 3.11, Yosys, and ABC exposing the shared actions.
   The default ABC command is `yosys-abc` (supplied by Yosys).

3. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. Install the Python dependencies (NumPy and CPU PyTorch):

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Verify both benchmark suites and the tracked ISCAS85 conversions:

   ```bash
   git submodule update --init --recursive
   git submodule status
   test -f benchmarks/epfl/arithmetic/adder.aig
   test -f benchmarks/epfl/random_control/ctrl.aig
   test -f benchmarks/iscas85/iscas85/verilog/c17.v
   test -f benchmarks/converted/iscas85/c17.aig
   git ls-files 'benchmarks/converted/iscas85/*.aig'
   ```

   EPFL AIG files are used directly. ISCAS85 originates from Verilog and requires
   conversion to AIG; all 11 converted `.aig` inputs are already tracked in Git,
   so no reconversion is required. Optionally run `python benchmarks/convert_iscas85.py`
   to regenerate and check equivalence. See [benchmark documentation](benchmarks/README.md)
   for conversion details, validation, and EPFL/ISPRAS source attribution.

6. Run unit and regression tests:

   ```bash
   python -m unittest discover -s tests -v
   ```

7. Run the minimal real-ABC evaluator smoke suite:

   ```bash
   python tests/smoke_abc.py --evaluator-only
   ```

## 2. Layout and Status

```text
algorithms/                # Each engine has engine.py and config.yaml
  greedy/                 # Sequential Forward Greedy
  nsga2/                  # Single-objective elitist GA (weighted sum)
  drills/                 # DRiLLS
  boils/                  # BOiLS
  ast_gwo/                # AST-GWO
common/                   # ABC evaluation, actions, results, progress
benchmarks/
  epfl/                   # EPFL benchmark submodule
  iscas85/                # ISPRAS benchmark submodule (Verilog sources)
  converted/iscas85/      # Tracked converted AIG inputs
  convert_iscas85.py      # Optional conversion and equivalence checks
  README.md              # Benchmark details and attribution
tests/                    # Regression and real-tool smoke checks
results/                  # Generated, ignored experiment artifacts
run.py                    # CLI entry point
requirements.txt          # Python dependencies
```

| Algorithm | CLI identifier | Implementation |
| --- | --- | --- |
| Greedy | `greedy` | `algorithms/greedy/` |
| GA (weighted sum) | `nsga2` | `algorithms/nsga2/` |
| DRiLLS | `drills` | `algorithms/drills/` |
| BOiLS | `boils` | `algorithms/boils/` |
| AST-GWO | `ast_gwo` | `algorithms/ast_gwo/` |

The default `algorithms/nsga2/` runner is a single-objective elitist GA with
weighted-sum fitness, not canonical multi-objective NSGA-II. Its explicit Pareto
mode remains available; these examples use the scalar GA.

## 3. Running Algorithms

Run from the repository root with the virtual environment activated. These small
examples use the same ISCAS85 circuit and seed; they check execution rather than
provide comparable full-experiment budgets.

```bash
# Greedy
python run.py --algorithm greedy --circuit benchmarks/converted/iscas85/c17.aig \
  --seed 0 --recipe-length 2 --max-tool-calls 32
# GA (single-objective weighted sum)
python run.py --algorithm nsga2 --circuit benchmarks/converted/iscas85/c17.aig \
  --seed 0 --recipe-length 2 --population-size 4 --offspring-count 4 \
  --generations 1 --max-tool-calls 8 --optimization-mode weighted_sum
# DRiLLS (CPU)
python run.py --algorithm drills --circuit benchmarks/converted/iscas85/c17.aig \
  --seed 0 --recipe-length 2 --episodes 2 --max-tool-calls 4 --device cpu --deterministic
# BOiLS
python run.py --algorithm boils --circuit benchmarks/converted/iscas85/c17.aig \
  --seed 0 --recipe-length 2 --max-tool-calls 4
# AST-GWO
python run.py --algorithm ast_gwo --circuit benchmarks/converted/iscas85/c17.aig \
  --seed 0 --recipe-length 2 --population-size 3 --generations 1 --max-tool-calls 6
```

Replace `--circuit` with an EPFL `.aig` path to use that suite. Each algorithm
loads `algorithms/<identifier>/config.yaml` by default. Use `--config path/to/config.yaml`
for an alternative JSON-syntax configuration (a YAML subset); CLI options override
its values. See `python run.py --help` for supported options. For comparisons,
match circuits, seeds, recipe lengths, LUT K, objective weights, and candidate
budgets explicitly; BOiLS defaults to 200 calls, the other engines to 4000.

One complete recipe evaluation costs one ABC call regardless of length. Cache
hits cost zero. The original-circuit normalization baseline is counted separately
and excluded from the candidate budget. Budget exhaustion preserves best-so-far
measurements; an evaluation failure aborts the run with diagnostics.

Outputs default to `results/<algorithm>/<suite>/<circuit>/<parameters>/seed_<seed>/`.
Runs write `config.json`, `result.json`, `best.json`, `best.csv` when a valid
successful result exists, and `progress.jsonl`. For example, Greedy above writes
`results/greedy/iscas85/c17/len2/seed_0/`. Rerunning the same parameter path and seed
replaces its known artifacts; archive results before changing other settings.

See source attribution and adaptations in the [GA](algorithms/nsga2/PROVENANCE.md),
[DRiLLS](algorithms/drills/PROVENANCE.md), [BOiLS](algorithms/boils/README.md),
and [AST-GWO](algorithms/ast_gwo/PROVENANCE.md) documentation.
