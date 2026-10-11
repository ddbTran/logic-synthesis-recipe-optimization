# Benchmark validation

Revalidated locally on 2026-10-11. Both submodules are registered, initialized,
clean, and pinned by Git links; their origin URLs match `.gitmodules`.

| Directory | Source | Pinned commit |
| --- | --- | --- |
| `epfl/` | [lsils/benchmarks](https://github.com/lsils/benchmarks) | `82d8cc6910419298e713a46644ed59fd3df53038` |
| `iscas85/` | [ISPRAS hdl-benchmarks](https://github.com/ispras/hdl-benchmarks) | `10c3fb51d0810ce862659211862660dcc270a6af` |

Do not edit or reclone these sources. On a fresh checkout, restore the pinned
submodules with `git submodule update --init --recursive`.

## ISCAS85 inventory and provenance

All 11 requested circuits exist in both structural Verilog and BENCH. Paths for
each table row are `iscas85/iscas85/verilog/<top>.v` and
`iscas85/iscas85/bench/<top>.bench`, relative to this directory. The selected
conversion source is Verilog; its top module equals the circuit name.

All 11 elaborate without errors, contain only generic combinational gates, and
have no technology-specific cells or latches. Elaboration produces only `$and`,
`$not`, `$or`, and `$xor` cells; the AIGER latch counts are zero. No technology
mapping or synthesis banner was observed. Their structural optimization history
is not established from inspection alone.

ISPRAS's root README attributes ISCAS85 to USC SPORT Lab and cites Hansen,
Yalcin and Hayes (1999). It also states that public benchmarks received bug and
formatting fixes. Thus the pinned files are identifiable public representations,
not certified untouched 1985 originals. Equivalence below validates conversion
of these exact files, not historical provenance or agreement with every BENCH
variant. The previous source and its generated c17 are no longer used.

| Circuit / top | Source and AIG inputs / outputs | ABC AIG nodes / levels | Source–AIG equivalence |
| --- | --- | --- | --- |
| c17 | 5 / 2 | 6 / 3 | PASS |
| c432 | 36 / 7 | 209 / 42 | PASS |
| c499 | 41 / 32 | 400 / 20 | PASS |
| c880 | 60 / 26 | 327 / 24 | PASS |
| c1355 | 41 / 32 | 504 / 26 | PASS |
| c1908 | 33 / 25 | 414 / 32 | PASS |
| c2670 | 233 / 140 | 717 / 21 | PASS |
| c3540 | 50 / 22 | 1038 / 41 | PASS |
| c5315 | 178 / 123 | 1773 / 38 | PASS |
| c6288 | 32 / 32 | 2337 / 120 | PASS |
| c7552 | 207 / 108 | 2074 / 29 | PASS |

## Conversion and functional verification

All 11 binary AIGER files are tracked in `converted/iscas85/`. No upstream file
was modified. Reproduce conversion, ABC parsing, counts and equivalence from the
project root:

```bash
python benchmarks/convert_iscas85.py
```

Exact per-circuit commands (substitute the circuit name for `<top>` and use a
temporary location for `<source.blif>`):

```text
yosys -Q -T -p 'read_verilog benchmarks/iscas85/iscas85/verilog/<top>.v; hierarchy -check -top <top>; check -assert; techmap; aigmap; write_aiger -symbols benchmarks/converted/iscas85/<top>.aig'
yosys -Q -T -p 'read_verilog benchmarks/iscas85/iscas85/verilog/<top>.v; hierarchy -check -top <top>; check -assert; simplemap; write_blif <source.blif>'
yosys-abc -s -c 'cec <source.blif> benchmarks/converted/iscas85/<top>.aig; echo CEC_COMPLETE'
```

`techmap`/`aigmap` perform gate lowering; no `synth`, `opt`, ABC optimization
recipe, or technology-library mapping is used to generate inputs. Port symbols
are retained. The script additionally exports elaboration JSON for source counts,
checks the binary AIGER header and matching I/O counts, and reads every AIG with
`common/abc.py` using an empty recipe.

The reference BLIF is independently elaborated from Verilog using `simplemap`
truth-table lowering, without the AIG conversion path. ABC `cec` matches inputs
and outputs by name by default. All **11/11 checks actually ran** and reported
`Networks are equivalent`. Nine completed after structural hashing; c432/c499
used the default FRAIG/SAT CEC flow. The script requires a positive equivalence
result and a completion marker; matching I/O alone is not a proof.
Proof preprocessing does not alter the stored benchmark inputs.

An initial direct Yosys equivalence approach proved seven circuits, but its AIGER
reader asserted for c2670/c5315/c7552, and c6288 SAT exceeded 60 seconds. The final
ABC CEC checks succeeded for all four as well; no circuit remains unverified
against its selected Verilog source. Checks use combinational binary semantics,
not analog timing or four-state simulation equivalence.

CSV records (including source/AIG SHA-256 hashes) and raw logs are written under
ignored `results/iscas85_validation/`. Failures are recorded per circuit and
produce a nonzero script exit; circuits are not silently skipped.

## EPFL and ABC/LUT-6 checks

EPFL has 20 original `.aig` circuits: ten in `arithmetic/` and ten in
`random_control/`, with matching BLIF/Verilog/VHDL files. These are reused directly.
Exclude the 40 optimized/mapped competition BLIF files under `best_results/`.
No EPFL files were converted or duplicated. Representative arithmetic/control
inputs both parse correctly and contain zero latches:

| Input | Inputs / outputs | AIG nodes / levels | LUT-6 count / levels |
| --- | --- | --- | --- |
| EPFL `arithmetic/adder.aig` | 256 / 129 | 1020 / 255 | 254 / 51 |
| EPFL `random_control/ctrl.aig` | 7 / 26 | 174 / 10 | 29 / 2 |
| ISPRAS converted `c17.aig` | 5 / 2 | 6 / 3 | 2 / 1 |

Tools: `Yosys 0.9 (git sha1 1979e0b)` and `/usr/bin/yosys-abc` (resolves to
`/usr/bin/berkeley-abc`), `ABC 1.01 (compiled Jan 29 2022 01:26:07)`.
ABC executable SHA-256:
`9f8539df83c751324a16ee98878b0e7e200a7ed4916d5270da3d9b2390c5ffc7`.

The existing smoke test is updated for the new source and reused:

```bash
python tests/smoke_abc.py
```

It regenerates only c17 temporarily and checks exact bytes, measures three
baselines, and tests all 16 shared actions individually on ctrl and c17.
All **32 action checks passed**, including every `-l` variant and both GIA
bundles. Mapping remains `if -K 6; print_stats`; raw `nd`/`lev` are independently
compared with parsed LUT count/levels. Ctrl actions 8–11 yielded 30 LUTs / 2
levels; other actions yielded 29 / 2. All c17 actions yielded 2 / 1.

Each smoke invocation performs 38 ABC evaluations (six empty-recipe baselines
and 32 single-action evaluations), totaling 32 logical action calls. Raw logs
and CSV measurements are under ignored `results/abc_smoke/`. AIG metrics remain
separate from LUT metrics; ASIC mapping is not validated. Shared actions,
`MAX_TOOL_CALLS = 4000`, evaluator, algorithms and example scripts are unchanged.
No optimization search ran. Source commits, clean statuses and all tracked file
hashes were checked unchanged after validation.
