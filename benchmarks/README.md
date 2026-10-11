# Benchmark inputs and ABC smoke validation

Validated locally on 2026-10-11. Prefer binary AIGER `.aig` for experiments.
Existing submodules are initialized; do not reclone or edit them. On a fresh
checkout, use `git submodule update --init --recursive` to restore the pinned inputs.

| Submodule | Source | Pinned commit |
| --- | --- | --- |
| `epfl/` | [lsils/benchmarks](https://github.com/lsils/benchmarks) | `82d8cc6910419298e713a46644ed59fd3df53038` |
| `iscas85/` | [jpsety/verilog_benchmark_circuits](https://github.com/jpsety/verilog_benchmark_circuits) | `b4c6b6203b95b5314d47365f4a8196c08145519b` |

## Inventory and provenance

EPFL originals have matching `.aig`, `.blif`, `.v` and `.vhd` files:

- `arithmetic/`: adder, bar, div, hyp, log2, max, multiplier, sin, sqrt, square.
- `random_control/`: arbiter, cavlc, ctrl, dec, i2c, int2float, mem_ctrl,
  priority, router, voter.

Reuse these 20 original `.aig` files directly. Exclude `best_results/size/` and
`best_results/depth/`: these contain 40 already optimized/mapped LUT-6 BLIF
variants. Total EPFL formats: 20 AIGER, 60 BLIF, 20 Verilog, 20 VHDL; no `.aag`
or `.bench`. The README describes additional MtM circuits, but they are not in
this checkout and were not downloaded.

The `iscas85/` repository has 30 `.v` files and no `.aig`, `.aag`, `.blif` or
`.bench` files. ISCAS85 candidates are c17, c432, c499, c880, c1355, c1908,
c2670, c3540, c5315, c6288 and c7552. The other 19 are EPFL-derived circuits,
not additional ISCAS85 cases; exclude them from an ISCAS85-only selection.

**Provenance limitation:** the repository README states that its circuits were
synthesized with Cadence Genus, and all 30 Verilog files have Genus headers.
They are generic-gate representations, not verified untouched original ISCAS85
netlists or a technology-cell library mapping. `c17.v` contains six NAND gates,
but this alone does not establish an unoptimized source history. No unsynthesized
ISCAS85 original is available here. Use c17 for environment validation only until
this representation is accepted for the paper or an original source is supplied.

## c17 conversion

ABC's direct `read_verilog benchmarks/iscas85/c17.v` fails at line 13:
`Cannot parse a standard gate (expected opening parenthesis).`
The named primitive instance `nand NAND2_1 (...)` is unsupported by this ABC
reader. ABC returns exit status zero despite failing; check diagnostics and
completion markers. General direct Verilog support for this suite is unvalidated.

Only c17 was converted, outside the submodule, with installed
`Yosys 0.9 (git sha1 1979e0b)`. From the project root:

```bash
mkdir -p benchmarks/converted/iscas85
yosys -Q -T -p 'read_verilog benchmarks/iscas85/c17.v; hierarchy -check -top c17; techmap; aigmap; write_aiger benchmarks/converted/iscas85/c17.aig'
```

This lowers generic gates to AIGER; it runs no `synth`, `opt`, `abc`, recipe search,
or technology-library mapping. The checked-in AIGER is regenerated in a temporary
directory by the smoke test and must match byte-for-byte on this Yosys build.
Conversion of other ISCAS85 files and suite-wide equivalence remain unvalidated.
Source and generated SHA-256 hashes:

```text
c17.v:   a59804dc4fc55fdb6fb282d092c579b9ccbe94dd3fa6e687464cbf9572c2b60f
c17.aig: cfd320daf1a5ca0bad551144f83308717193ae87e01dc02ac155e5c40f4c7b79
ctrl.aig: 9da22b9ba9917569878d750926623d3920a4cd985a0980cc65763c5d3ac0e78b
```

## ABC environment and checks

- Executable: `/usr/bin/yosys-abc`, resolving to `/usr/bin/berkeley-abc`.
- Build: `UC Berkeley, ABC 1.01 (compiled Jan 29 2022 01:26:07)`.
- Executable SHA-256: `9f8539df83c751324a16ee98878b0e7e200a7ed4916d5270da3d9b2390c5ffc7`.
- AIGER parsing succeeds for EPFL `random_control/ctrl.aig` and converted c17;
  explicit `read_aiger` was also checked on ctrl.
- Measurement follows the reference examples: `read; strash; [recipe]; if -K 6;
  print_stats`. `nd` is LUT count and `lev` is LUT-network depth, not ASIC delay.
  Unmapped `and`/`lev` are recorded separately. No ASIC library is used.
- `common/abc.py` is exercised without changes, using `-s -f`, raw output,
  completion markers, timeouts and action accounting. Neither reference example
  script is modified or executed.

Baseline smoke measurements, with no pool actions applied:

| Input | Inputs / outputs / latches | AIG nodes / levels | LUT-6 count / levels |
| --- | --- | --- | --- |
| Original EPFL `epfl/random_control/ctrl.aig` | 7 / 26 / 0 | 174 / 10 | 29 / 2 |
| Converted Genus representation `converted/iscas85/c17.aig` | 5 / 2 / 0 | 6 / 3 | 2 / 1 |

All **16/16 actions passed individually on each circuit**, loaded fresh for every
check. No unsupported options were found, including `rewrite -l`, `refactor -l`,
`resub -l`, `balance -l`, their zero-cost combinations, and both GIA bundles.
On ctrl, actions 8–11 yielded 30 LUTs / 2 levels; the other actions yielded
29 / 2. Every action on c17 yielded 2 / 1. These are compatibility measurements,
not algorithm comparison results or a guarantee for every circuit.

Reproduce the lightweight checks from the project root:

```bash
python tests/smoke_abc.py
# Optional alternative binary:
python tests/smoke_abc.py --abc-binary /path/to/abc
```

Each invocation checks conversion and performs 36 ABC evaluations: four empty
recipe baselines and 32 single-action checks, totaling 32 logical pool calls.
Each GIA bundle counts as one. `MAX_TOOL_CALLS = 4000` and the action pool remain
unchanged. CSV measurements and raw logs go to ignored `results/abc_smoke/`;
failures produce a nonzero exit. No optimization engines or searches are run.
Submodule commits, statuses and all dataset file hashes were unchanged after
validation; the example scripts were also unchanged.
