"""Real-tool checks: three input baselines; shared actions on ctrl and c17."""

import argparse
import csv
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.abc import ABCError, ActionBudget, BudgetExceeded, evaluate_recipe, parse_qor
from common.actions import MAX_TOOL_CALLS, TRANSFORMS


class EvaluatorTests(unittest.TestCase):
    abc_binary = "yosys-abc"
    circuits = (ROOT / "benchmarks/epfl/random_control/ctrl.aig",
                ROOT / "benchmarks/converted/iscas85/c17.aig")

    def evaluate(self, recipe, *, circuit=None, budget=None, **kwargs):
        return evaluate_recipe(
            circuit if circuit is not None else self.circuits[0], recipe,
            budget=budget if budget is not None else ActionBudget(40),
            metric="lut", lut_k=6, abc_binary=self.abc_binary, **kwargs,
        )

    def test_real_recipes_and_determinism(self):
        for circuit, expected in zip(self.circuits, [(29, 2), (2, 1)]):
            with self.subTest(circuit=circuit):
                budget = ActionBudget(40)
                baseline = self.evaluate([], circuit=circuit, budget=budget, timeout=None)
                self.assertEqual(tuple(baseline["metrics"].values()), expected)
                self.assertEqual(baseline["tool_calls"], 0)
                results = []
                for recipe in ([0], [0, 4, 12], [0, 0], [14, 15], list(range(16)) + [0, 4, 8, 12]):
                    result = self.evaluate(recipe, circuit=circuit, budget=budget)
                    self.assertEqual(result["recipe"], recipe)
                    self.assertEqual(result["tool_calls"], len(recipe))
                    self.assertEqual(result["transformation_count"], len(recipe))
                    self.assertEqual(result["abc_process_invocations"], 1)
                    self.assertEqual(set(result["metrics"]), {"lut_count", "lut_levels"})
                    raw = re.findall(r"\bnd\s*=\s*(\d+).*?\blev\s*=\s*(\d+)", result["abc_output"])
                    self.assertEqual(tuple(map(int, raw[-1])), tuple(result["metrics"].values()))
                    results.append(result)
                repeated = self.evaluate([0, 4, 12], circuit=circuit, budget=budget)
                self.assertEqual(repeated["metrics"], results[1]["metrics"])
                # A baseline after optimized evaluations must reload the original.
                after = self.evaluate([], circuit=circuit, budget=budget)
                self.assertEqual(after["metrics"], baseline["metrics"])
                self.assertEqual(budget.tool_calls, 31)
                self.assertEqual(after["total_tool_calls"], 31)

    def test_real_execution_order_and_original_input(self):
        run = subprocess.Popen
        recipe = [8, 0, 14, 0]

        def inspect_and_run(args, **kwargs):
            directory = Path(kwargs["cwd"])
            self.assertEqual((directory / "input.aig").resolve(), self.circuits[0].resolve())
            commands = (directory / args[-1]).read_text().splitlines()
            actions = [line.removesuffix(";") for line in commands
                       if line.removesuffix(";") in TRANSFORMS]
            self.assertEqual(actions, [TRANSFORMS[i] for i in recipe])
            self.assertEqual(commands[:2], ["read input.aig;", "strash;"])
            self.assertEqual(commands[-3:-1], ["if -K 6;", "print_stats;"])
            return run(args, **kwargs)

        with patch("common.abc.subprocess.Popen", side_effect=inspect_and_run):
            self.assertEqual(self.evaluate(recipe)["tool_calls"], len(recipe))

    def test_real_shared_budget_and_exhaustion(self):
        budget = ActionBudget(3)
        self.assertEqual(self.evaluate([0], budget=budget)["total_tool_calls"], 1)
        result = self.evaluate([14, 15], budget=budget)
        self.assertEqual((result["tool_calls"], result["total_tool_calls"]), (2, 3))
        with patch("common.abc.subprocess.Popen") as launch:
            with self.assertRaises(BudgetExceeded):
                self.evaluate([12], budget=budget)
            launch.assert_not_called()
        self.assertEqual(self.evaluate([], budget=budget)["total_tool_calls"], 3)
        with patch("common.abc.subprocess.Popen") as launch:
            with self.assertRaises(BudgetExceeded):
                self.evaluate([0] * 20, budget=ActionBudget(19))
            launch.assert_not_called()

    def test_invalid_indices_and_timeouts(self):
        budget = ActionBudget(3)
        with patch("common.abc.subprocess.Popen") as run:
            for index in (-1, 16, True, 1.5, "0"):
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, "indices 0-15"):
                    self.evaluate([index], budget=budget)
            for timeout in (0, -1, float("inf"), float("nan")):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    self.evaluate([], budget=budget, timeout=timeout)
            run.assert_not_called()
        self.assertEqual(budget.tool_calls, 0)

    def test_real_final_call_at_default_budget_boundary(self):
        # Seed the counter to test the limit without running a full experiment.
        budget = ActionBudget(MAX_TOOL_CALLS, MAX_TOOL_CALLS - 20)
        result = self.evaluate([0] * 20, circuit=self.circuits[1], budget=budget)
        self.assertEqual(result["total_tool_calls"], 4000)
        with patch("common.abc.subprocess.Popen") as launch:
            with self.assertRaises(BudgetExceeded):
                self.evaluate([0], budget=budget)
            launch.assert_not_called()
        self.assertEqual(budget.tool_calls, 4000)

    def test_missing_input_and_executable(self):
        budget = ActionBudget(3)
        with tempfile.TemporaryDirectory() as directory, patch("common.abc.subprocess.Popen") as run:
            with self.assertRaises(FileNotFoundError):
                self.evaluate([], circuit=Path(directory) / "missing.aig", budget=budget)
            with patch("common.abc.shutil.which", return_value=None):
                with self.assertRaisesRegex(ABCError, "executable unavailable"):
                    self.evaluate([], budget=budget)
            run.assert_not_called()
        self.assertEqual(budget.tool_calls, 0)

    def test_command_errors_preserve_diagnostics_and_uncertain_count(self):
        for code in (0, 1):
            with self.subTest(returncode=code):
                budget = ActionBudget(3)
                output = "ACTION_test_0\n** cmd error: failed\n"
                process = Mock(returncode=code)
                process.communicate.return_value = (output, None)
                with patch("common.abc.uuid.uuid4", return_value=SimpleNamespace(hex="test")), \
                     patch("common.abc.subprocess.Popen", return_value=process):
                    with self.assertRaises(ABCError) as caught:
                        self.evaluate([0, 14], budget=budget)
                self.assertEqual(caught.exception.completed_actions, 1)
                self.assertEqual(caught.exception.output, output)
                self.assertIsNone(budget.tool_calls)

    def test_timeouts_preserve_known_counts(self):
        cases = [([0, 14], b"ACTION_test_0\n", None, 1),
                 ([0, 14], b"ACTION_test_0\nACTION_test_1\n", 2, 2),
                 ([], b"", 0, 0)]
        for recipe, output, expected_count, completed in cases:
            with self.subTest(recipe=recipe, output=output):
                budget = ActionBudget(3)
                error = subprocess.TimeoutExpired("abc", 0.01, output=output)
                process = Mock()
                process.communicate.side_effect = [error, (output.decode(), None)]
                with patch("common.abc.uuid.uuid4", return_value=SimpleNamespace(hex="test")), \
                     patch("common.abc.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(ABCError, "timed out") as caught:
                        self.evaluate(recipe, budget=budget, timeout=0.01)
                self.assertEqual(budget.tool_calls, expected_count)
                self.assertEqual(caught.exception.completed_actions, completed)
                self.assertEqual(caught.exception.output, output.decode())
                process.kill.assert_called_once()

    def test_launch_failure_does_not_charge_calls(self):
        budget = ActionBudget(3)
        with patch("common.abc.subprocess.Popen", side_effect=OSError("launch failure")):
            with self.assertRaisesRegex(ABCError, "Cannot execute ABC"):
                self.evaluate([0], budget=budget)
        self.assertEqual(budget.tool_calls, 0)

    def test_unexpected_launch_failure_stops_with_uncertain_count(self):
        budget = ActionBudget(3)
        with patch("common.abc.subprocess.Popen", side_effect=RuntimeError("uncertain launch")):
            with self.assertRaisesRegex(ABCError, "launch state uncertain"):
                self.evaluate([0], budget=budget)
        self.assertIsNone(budget.tool_calls)
        with self.assertRaises(BudgetExceeded):
            budget.check()

    def test_unexpected_wait_failure_stops_with_uncertain_count(self):
        budget = ActionBudget(3)
        process = Mock()
        process.communicate.side_effect = OSError("read failure")
        with patch("common.abc.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(ABCError, "Cannot collect"):
                self.evaluate([0], budget=budget)
        self.assertIsNone(budget.tool_calls)
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    def test_missing_metrics_never_returns_qor(self):
        output = "ACTION_test_0\nFINISHED_test\nno statistics\n"
        budget = ActionBudget(3)
        process = Mock(returncode=0)
        process.communicate.return_value = (output, None)
        with patch("common.abc.uuid.uuid4", return_value=SimpleNamespace(hex="test")), \
             patch("common.abc.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(ABCError, "Missing lut"):
                self.evaluate([0], budget=budget)
        self.assertEqual(budget.tool_calls, 1)
        with self.assertRaises(ABCError):
            parse_qor("and = 10 lev = 2", "lut")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc-binary", default="yosys-abc")
    parser.add_argument("--evaluator-only", action="store_true",
                        help="Focused recipe/budget/error checks; no benchmark conversion")
    args = parser.parse_args()
    if args.evaluator_only:
        EvaluatorTests.abc_binary = args.abc_binary
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvaluatorTests)
        return int(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    output = ROOT / "results" / "abc_smoke"
    output.mkdir(parents=True, exist_ok=True)
    failures = 0

    # Recreate only c17 in a temporary directory to check the tracked conversion.
    with tempfile.TemporaryDirectory(prefix="c17-conversion-") as directory:
        destination = Path(directory) / "c17.aig"
        command = (
            "read_verilog benchmarks/iscas85/iscas85/verilog/c17.v; hierarchy -check -top c17; "
            f"techmap; aigmap; write_aiger -symbols {destination}"
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
        "benchmarks/epfl/arithmetic/adder.aig",
    ]
    with (output / "checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for circuit in circuits:
            checks = [("baseline_aig", [], "aig"), ("baseline_lut", [], "lut")]
            if circuit in circuits[:2]:
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
                    label = "and" if metric == "aig" else "nd"
                    raw = re.findall(rf"\b{label}\s*=\s*(\d+).*?\blev\s*=\s*(\d+)", result["abc_output"])
                    if not raw or tuple(map(int, raw[-1])) != tuple(result["metrics"].values()):
                        raise RuntimeError("Raw ABC statistics differ from parsed QoR")
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
