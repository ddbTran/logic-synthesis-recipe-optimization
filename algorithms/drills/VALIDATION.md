# DRiLLS validation (2026-10-11)

Environment: Python 3.11.15, PyTorch 2.7.1+cpu, NumPy 2.2.6,
Yosys 0.9 (`1979e0b`), yosys-abc 1.01 (compiled Jan 29 2022).
Dependencies passed `pip check`. Original source comparison used the local
`../DRiLLS` checkout at `c8350ce3ac4198733b523cb52d5ba79aec0ed8aa`.

Commands were executed sequentially from the repository root:

```bash
.venv/bin/python tests/test_drills.py -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python tests/smoke_abc.py --evaluator-only
.venv/bin/python run.py --algorithm drills \
  --circuit benchmarks/epfl/arithmetic/adder.aig --device cpu \
  --episodes 2 --recipe-length 3 --max-tool-calls 6 --deterministic --seed 0
```

- Focused DRiLLS suite: 21 tests, 20 passed, one CUDA test skipped.
- Full regression suite after final code changes: 74 tests, 73 passed,
  one CUDA test skipped. Includes existing NSGA-II, weighted-sum GA, Greedy,
  shared progress/result tests and real ABC regression runs on ctrl/c17.
- ABC evaluator suite: all 12 tests passed, including shared actions, budgets,
  parsing, subprocess failure and timeout handling.
- CPU CLI training: completed both episodes and two optimizer updates;
  five candidate calls, one cache hit, one separately counted baseline call.
  The real-tool unit test also verified actor and critic parameters both change.
- CPU model/optimizer checkpoint round-trip and subsequent identical update
  passed. Deterministic CPU seeding/training passed.
- CUDA-unavailable failure handling and auto-selection branches passed.
  Real CUDA training, CUDA tensor placement, and cross-device checkpoint
  execution remain unverified because the retained PyTorch build is CPU-only.
  GPU support tests are present for a future compatible CUDA environment.

No full benchmark sweep was run. No TensorFlow numerical or published-QoR
equivalence is claimed. Model files omit experiment/RNG state and cannot resume
an interrupted experiment. CPU/GPU RNGs and framework/tool versions can change
results; see [PROVENANCE.md](PROVENANCE.md). Raw local validation output and
training results are ignored under `results/`; virtual environments and model
files are also ignored.
