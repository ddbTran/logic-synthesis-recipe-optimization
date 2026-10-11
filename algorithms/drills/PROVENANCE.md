# DRiLLS source and adaptation

Source: [scale-lab/DRiLLS](https://github.com/scale-lab/DRiLLS), revision
`c8350ce3ac4198733b523cb52d5ba79aec0ed8aa`. Compared
[`drills/model.py`](https://github.com/scale-lab/DRiLLS/blob/c8350ce3ac4198733b523cb52d5ba79aec0ed8aa/drills/model.py),
[`drills/features.py`](https://github.com/scale-lab/DRiLLS/blob/c8350ce3ac4198733b523cb52d5ba79aec0ed8aa/drills/features.py),
[`drills/fpga_session.py`](https://github.com/scale-lab/DRiLLS/blob/c8350ce3ac4198733b523cb52d5ba79aec0ed8aa/drills/fpga_session.py)
and `params.yml`. Model and feature logic are adapted under the retained
[BSD-3-Clause license](LICENSE), copyright 2019 SCALE Lab, Brown University.
Attribution: A. Hosny, S. Hashemi, M. Shalan, S. Reda, *DRiLLS: Deep
Reinforcement Learning for Logic Synthesis*, ASP-DAC 2020, pp. 581–586.

## Source behavior retained

Actor: two ReLU layers of 20 units, softmax output. Critic: 10 ReLU units and
one linear output. Xavier weights and zero biases; Adam at 0.01, gamma 0.99.
Categorical sampling remains stochastic (NumPy on CPU, PyTorch on CUDA). One combined optimizer step
follows each completed episode. Discounted returns use zero terminal bootstrap
and population-standard-deviation normalization.

The source subtracts a `[T,1]` critic from `[T]` returns, yielding `[T,T]`
advantages. Both losses are sums; the actor uses negative log action probability
times advantage, and the critic uses squared advantage. No stop-gradient is
present. This port now deliberately corrects those two behaviors: squeezing the
critic output produces one `[T]` advantage per timestep, and the actor loss uses
a detached advantage. Only the critic loss updates critic parameters. Losses
remain summed, with one combined Adam step per complete episode. Results use
variant `drills-elementwise-a2c-weighted-improvement-v2`; learning curves and loss
scales differ from the previous literal-source port. The source's unused
policy-gradient expression adds no loss here.

Observations, in order: input pins, output pins, Yosys cells, ABC edges,
ABC levels, latches, AND/cell, OR/cell, NOT/cell. Raw statistics come from a
pre-mapping Verilog export. Welford normalization resets per episode and clamps
population variance to `[0.01, 1e9]` before dividing by its square root.

## Benchmark adaptations and limits

- PyTorch replaces TensorFlow 1.x. Initialization and Adam framework arithmetic
  can differ numerically; TensorFlow checkpoints cannot be loaded. Optional
  widths/rates retain source defaults. No entropy bonus, gradient clipping,
  extra loss, reward clipping or additional training update is added.
- Zero-variance returns normalize to zeros (the source divides by zero).
  Zero-cell circuits have zero gate fractions. Nonfinite losses/gradients abort;
  a nonfinite optimizer result is rolled back and the run fails.
- The required shared 16-action pool replaces the original seven actions.
  A configured length means exactly that many actions. The original FPGA reset
  increments its iteration counter, leaving one fewer action than `iterations`.
- The required normalized weighted-fitness difference replaces the source's
  discrete timing-constraint reward table. QoR uses LUT K and the fixed original
  mapped baseline; best selection uses fitness. No ASIC mapping is implemented.
- Each candidate replays the full ordered recipe from the original circuit in
  one shared evaluator call. A pre-mapping export hook keeps raw observations
  separate from mapped QoR. Yosys and ABC feature reads run sequentially, with
  independent auxiliary process counters, instead of upstream multiprocessing.
- Baseline costs one separately reported evaluation. The global candidate budget
  spans episodes. Successful complete-recipe cache hits cost zero; failures are
  not cached. A launched failure/timeout remains charged. Interrupted partial
  trajectories are recorded without training, preserving the source's
  episode-only update boundary; no truncated-trajectory bootstrap is invented.
- CPU/CUDA selection affects neural computation only. Observation preprocessing
  and returns preprocessing remain CPU-side. CUDA action sampling uses
  `torch.multinomial` on device, so it uses a different RNG than original NumPy.
  Seeds do not
  promise bitwise agreement across devices, frameworks or tool versions.
  CUDA Adam uses `capturable=True` to keep optimizer step tensors and arithmetic
  on the selected GPU; CPU Adam uses its ordinary CPU path.
- Shared progress checkpoints are measured JSONL snapshots, not restartable
  training checkpoints. `Agent.save/load` supports weights and optimizer state
  with explicit device mapping; RNG, environment, budget and normalizer are not
  restored, so experiment training resume remains unsupported.
  Results record architecture, features, software/device/tool identity and hashes.

Small unit and real-tool tests are in `tests/test_drills.py`. This is a port with
the declared benchmark adaptations, not a claim to reproduce published QoR,
learning curves, source framework numerics, or full benchmark experiments.
