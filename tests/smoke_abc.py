"""Small real-tool checks only: two inputs, baselines, and each shared action."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.abc import ABCError, ActionBudget, evaluate_recipe
from common.actions import MAX_TOOL_CALLS, TRANSFORMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc-binary", default="yosys-abc")
    args = parser.parse_args()
    output = ROOT / "results" / "abc_smoke"
    output.mkdir(parents=True, exist_ok=True)
    failures = 0

    # Recreate only c17 in a temporary directory to check the tracked conversion.
    with tempfile.TemporaryDirectory(prefix="c17-conversion-") as directory:
        destination = Path(directory) / "c17.aig"
        command = (
            "read_verilog benchmarks/iscas85/c17.v; hierarchy -check -top c17; "
            f"techmap; aigmap; write_aiger {destination}"
        )
        try:
            process = subprocess.run(
                ["yosys", "-Q", "-T", "-p", command], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=30, check=True,
            )
            (output / "conversion.log").write_text(process.stdout)
            if destination.read_bytes() != (ROOT / "benchmarks/converted/iscas85/c17.aig").read_bytes():
                raise RuntimeError("c17 conversion differs; inspect Yosys version and source")
            print("PASS c17 conversion matches tracked AIGER")
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            (output / "conversion.log").write_text(str(exc) + "\n" + str(getattr(exc, "stdout", "") or ""))
            print(f"FAIL conversion: {exc}", file=sys.stderr)
            failures += 1

    fields = [
        "circuit", "check", "action_index", "action", "metric", "lut_k",
        "aig_nodes", "aig_levels", "lut_count", "lut_levels", "tool_calls",
        "runtime_seconds", "status", "error",
    ]
    circuits = [
        "benchmarks/epfl/random_control/ctrl.aig",
        "benchmarks/converted/iscas85/c17.aig",
    ]
    with (output / "checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for circuit in circuits:
            checks = [("baseline_aig", [], "aig"), ("baseline_lut", [], "lut")]
            checks += [(f"action_{i:02}", [i], "lut") for i in range(len(TRANSFORMS))]
            for name, recipe, metric in checks:
                row = {
                    "circuit": circuit, "check": name, "metric": metric,
                    "lut_k": 6 if metric == "lut" else "",
                    "action_index": recipe[0] if recipe else "",
                    "action": TRANSFORMS[recipe[0]] if recipe else "",
                }
                # Each compatibility check is an independent evaluation, not search.
                budget = ActionBudget(MAX_TOOL_CALLS)
                try:
                    result = evaluate_recipe(
                        ROOT / circuit, recipe, budget=budget, metric=metric,
                        abc_binary=args.abc_binary, timeout=30,
                        **({"lut_k": 6} if metric == "lut" else {}),
                    )
                    if result["tool_calls"] != len(recipe) or budget.tool_calls != len(recipe):
                        raise RuntimeError("Action accounting mismatch")
                    row.update(result["metrics"], tool_calls=result["tool_calls"],
                               runtime_seconds=result["runtime_seconds"], status="PASS")
                    log = result["abc_output"]
                    print(f"PASS {Path(circuit).stem} {name}: {result['metrics']}")
                except (ABCError, OSError, ValueError, RuntimeError) as exc:
                    failures += 1
                    row.update(status="FAIL", error=str(exc))
                    log = getattr(exc, "output", str(exc))
                    print(f"FAIL {circuit} {name}: {exc}", file=sys.stderr)
                (output / f"{Path(circuit).stem}_{name}.log").write_text(log)
                writer.writerow(row)
    print(f"{failures} failures; numerical checks and raw logs: {output}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
