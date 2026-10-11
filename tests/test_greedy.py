"""Sequential Forward Greedy contracts and small real-ABC integration checks."""

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.greedy import engine as e
from common.experiments import describe_experiment, experiment_path, prepare_run
from common.progress import ProgressLogger
import run as runner


class GreedyTests(unittest.TestCase):
    circuit = ROOT / "benchmarks/converted/iscas85/c17.aig"

    def config(self, **kwargs):
        return replace(e.Config(recipe_length=3), **kwargs)

    def evaluator(self, score=None):
        def evaluate(circuit, recipe, *, budget, **kwargs):
            budget.check()
            budget.tool_calls += 1
            budget.abc_process_invocations += 1
            metrics = (100, 10) if not recipe else (
                score(tuple(recipe)) if score else (100 + abs(recipe[-1] - 4), 10))
            return {"metrics": dict(zip(("lut_count", "lut_levels"), metrics))}
        return evaluate

    def execute(self, config=None, score=None):
        events = []
        with patch.object(e.abc, "evaluate_recipe", side_effect=self.evaluator(score)) as evaluate:
            result = e.run(self.circuit, config or self.config(), progress=events.append)
        return result, events, evaluate

    def test_select_minimum_repeats_and_extends_prefix(self):
        target = (4, 4, 2)
        result, events, evaluate = self.execute(score=lambda r: (100 + abs(r[-1] - target[len(r)-1]), 10))
        self.assertEqual(result["constructed_recipe"], list(target))
        self.assertEqual(result["status"], "completed")
        calls = evaluate.call_args_list[1:]
        for depth in range(1, 4):
            recipes = [tuple(call.args[1]) for call in calls[(depth-1)*16:depth*16]]
            self.assertEqual(recipes, [target[:depth-1] + (i,) for i in range(16)])
        self.assertEqual([event["depth"] for event in events if event["event"] == "step"], [1, 2, 3])

    def test_deterministic_ties_and_identical_runs(self):
        first, _, _ = self.execute(score=lambda r: (100, 10))
        second, _, _ = self.execute(score=lambda r: (100, 10))
        self.assertEqual(first["constructed_recipe"], [0, 0, 0])
        self.assertEqual(first["best"], second["best"])
        self.assertEqual(first["constructed_recipe"], second["constructed_recipe"])
        self.assertEqual(first["best"]["recipe"], [0])
        self.assertEqual(first["best_complete"]["recipe"], [0, 0, 0])

    def test_every_candidate_uses_original_circuit_and_shared_mapping(self):
        result, _, evaluate = self.execute()
        for call in evaluate.call_args_list:
            self.assertEqual(call.args[0], self.circuit)
            self.assertEqual(call.kwargs["metric"], "lut")
            self.assertEqual(call.kwargs["lut_k"], 6)
        budgets = [call.kwargs["budget"] for call in evaluate.call_args_list]
        self.assertTrue(all(b is budgets[1] for b in budgets[1:]))
        self.assertIsNot(budgets[0], budgets[1])
        self.assertEqual(result["baseline_tool_calls"], 1)
        self.assertEqual(result["abc_process_invocations"], 49)

    def test_normalized_weighted_fitness_changes_selection(self):
        score = lambda r: (20, 10) if r[-1] == 2 else (100, 2)
        area, _, _ = self.execute(self.config(recipe_length=1, weight_area=0.8, weight_delay=0.2), score)
        delay, _, _ = self.execute(self.config(recipe_length=1, weight_area=0.2, weight_delay=0.8), score)
        self.assertEqual(area["best"]["recipe"], [2])
        self.assertAlmostEqual(area["weighted_fitness"], 0.36)
        self.assertEqual(delay["best"]["recipe"], [0])
        self.assertAlmostEqual(delay["weighted_fitness"], 0.36)

    def test_worse_prefix_does_not_stop_and_best_seen_is_separate(self):
        result, _, _ = self.execute(score=lambda r: (len(r)*10 + abs(r[-1]-4), len(r)))
        self.assertEqual(result["completed_depth"], 3)
        self.assertEqual(result["constructed_recipe"], [4, 4, 4])
        self.assertEqual(result["best"]["recipe"], [4])
        self.assertEqual(result["best_so_far"], result["best"])
        self.assertEqual(result["best_complete"], result["constructed_solution"])
        self.assertEqual(result["tool_calls"], 48)
        self.assertEqual(result["evaluations"], 48)

    def test_one_call_per_recipe_and_cache_hits_at_full_budget(self):
        config = self.config(max_tool_calls=3)
        budget = e.abc.ActionBudget(3)
        evaluator = e.RecipeEvaluator(self.circuit, config, {"lut_count": 100, "lut_levels": 10}, budget)
        with patch.object(e.abc, "evaluate_recipe", side_effect=self.evaluator()) as launch:
            for recipe in ((0,), (0, 4, 12), (0,)*20):
                evaluator.evaluate(recipe)
            before = budget.tool_calls
            evaluator.evaluate([0, 4, 12])
            self.assertEqual(budget.tool_calls, before)
            self.assertEqual(launch.call_count, 3)
            with self.assertRaises(e.abc.BudgetExceeded):
                evaluator.evaluate([12, 4, 0])
        self.assertEqual(evaluator.cache_hits, 1)
        self.assertEqual(evaluator.evaluations, 3)
        self.assertEqual(evaluator.candidate_attempts, 5)
        self.assertEqual(len(evaluator.cache), 3)

    def test_checkpoint_thresholds_are_cumulative_evaluations(self):
        result, events, _ = self.execute(self.config(checkpoint_interval=10))
        checkpoints = [event for event in events if event["event"] == "checkpoint"]
        self.assertEqual([c["tool_calls"] for c in checkpoints], [10, 20, 30, 40, 48])
        self.assertEqual([c["checkpoint_thresholds"] for c in checkpoints], [[10], [20], [30], [40], []])
        self.assertEqual(checkpoints[-1]["evaluations"], result["evaluations"])
        self.assertEqual(events[-1]["status"], "completed")
        self.assertTrue(all("generation" not in event for event in events))
        self.assertTrue(set(event["event"] for event in events) <=
                        {"configuration", "baseline", "improvement", "checkpoint", "step", "summary"})
        for field in ("best", "best_fitness", "best_lut_count", "best_lut_levels", "evaluations",
                      "tool_calls", "cache_hits", "candidate_attempts", "elapsed_seconds", "status"):
            self.assertIn(field, checkpoints[-1])

    def test_partial_depth_budget_and_complete_candidate_tracking(self):
        for budget, completed, partial in ((0, 0, 0), (5, 0, 5), (16, 1, 0), (19, 1, 3)):
            with self.subTest(budget=budget):
                result, events, evaluate = self.execute(self.config(recipe_length=2, max_tool_calls=budget))
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(result["tool_calls"], budget)
                self.assertEqual(result["candidate_attempts"], budget+1)
                self.assertEqual(evaluate.call_count, budget+1)  # includes baseline, no rejected launch
                self.assertEqual(result["completed_depth"], completed)
                self.assertEqual(result["depth_candidates_evaluated"], partial)
                self.assertEqual(len(result["constructed_recipe"]), completed)
                self.assertEqual(events[-1]["status"], "budget_exhausted")
                if budget == 19:
                    self.assertEqual(result["best_complete"]["recipe"], [4, 2])
                    self.assertEqual(result["constructed_recipe"], [4])
                if budget == 0:
                    self.assertIsNone(result["best"])
        result, _, _ = self.execute(self.config(recipe_length=2, max_tool_calls=32))
        self.assertEqual(result["status"], "completed")

    def test_best_candidate_from_partial_depth_is_retained(self):
        result, _, _ = self.execute(self.config(recipe_length=2, max_tool_calls=19),
                                    lambda r: (100-len(r)*10+r[-1], 10))
        self.assertEqual(result["best"]["recipe"], [0, 0])
        self.assertEqual(result["constructed_recipe"], [0])
        self.assertEqual(result["best_complete"], result["best"])

    def test_invalid_config_and_file_format(self):
        for key, value in (("recipe_length", 0), ("recipe_length", True), ("max_tool_calls", -1),
                           ("checkpoint_interval", 0), ("seed", 1.5), ("lut_k", 1),
                           ("optimization_mode", "pareto"), ("timeout", 0), ("timeout", float("nan")),
                           ("weight_area", -0.1), ("weight_area", float("inf")),
                           ("weight_area", 0.7), ("abc_binary", "")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.config(**{key: value})
        self.assertEqual(e.load_config(), e.Config())
        self.assertIsNone(self.config(timeout=None).timeout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for data, error in (([], ValueError), ({"population_size": 4}, TypeError)):
                path.write_text(json.dumps(data))
                with self.assertRaises(error):
                    e.load_config(path)

    def test_invalid_baseline_and_candidate_qor(self):
        for baseline in (True, False):
            for bad in (None, {}, {"lut_count": -1, "lut_levels": 2},
                        {"lut_count": float("nan"), "lut_levels": 2},
                        {"lut_count": True, "lut_levels": 2}):
                normal = self.evaluator()
                def evaluate(circuit, recipe, **kwargs):
                    result = normal(circuit, recipe, **kwargs)
                    return {"metrics": bad} if (not recipe) == baseline else result
                with self.subTest(baseline=baseline, bad=bad), \
                     patch.object(e.abc, "evaluate_recipe", side_effect=evaluate):
                    result = e.run(self.circuit, self.config())
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["best"])
                self.assertEqual(result["evaluations"], 0)
        with self.assertRaises(ValueError):
            e.valid_metrics({"lut_count": 0, "lut_levels": 1}, baseline=True)

    def test_abc_failure_accounting_and_no_success_artifact(self):
        for uncertain in (False, True):
            normal = self.evaluator()
            def evaluate(circuit, recipe, *, budget, **kwargs):
                if recipe and recipe[-1] == 1:
                    budget.tool_calls = None if uncertain else budget.tool_calls + 1
                    raise e.abc.ABCError("failed", output="diagnostic")
                return normal(circuit, recipe, budget=budget, **kwargs)
            with patch.object(e.abc, "evaluate_recipe", side_effect=evaluate):
                result = e.run(self.circuit, self.config())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["tool_calls"], None if uncertain else 2)
            self.assertEqual(result["evaluations"], 1)
            self.assertIsNone(result["best"])
            self.assertIsNotNone(result["best_so_far"])
            self.assertEqual(result["failure"]["abc_output"], "diagnostic")
            with tempfile.TemporaryDirectory() as root:
                path = e.write_results(result, root)
                self.assertFalse((path / "best.csv").exists())
                self.assertIsNone(json.loads((path / "best.json").read_text()))
        config = self.config()
        evaluator = e.RecipeEvaluator(self.circuit, config, {"lut_count": 100, "lut_levels": 10}, e.abc.ActionBudget(2))
        with patch.object(e.abc, "evaluate_recipe", side_effect=e.abc.ABCError("failure")):
            with self.assertRaises(e.abc.ABCError):
                evaluator.evaluate([0])
        self.assertEqual(evaluator.cache, {})
        self.assertEqual(evaluator.evaluations, 0)

    def test_outputs_paths_and_rerun_cleanup(self):
        result, _, _ = self.execute(self.config(seed=7))
        experiment = describe_experiment("greedy", self.circuit, result["config"])
        self.assertEqual(experiment_path(experiment).as_posix(), "results/greedy/iscas85/c17/len3/seed_7")
        with tempfile.TemporaryDirectory() as root:
            path = prepare_run(experiment, root)
            with ProgressLogger(path / "progress.jsonl") as progress:
                progress({"event": "summary", "status": "completed"})
            result["experiment"] = experiment
            self.assertEqual(e.write_results(result, directory=path), path)
            self.assertEqual({p.name for p in path.iterdir()},
                             {"config.json", "progress.jsonl", "result.json", "best.json", "best.csv"})
            self.assertEqual(json.loads((path / "best.json").read_text()), result["best"])
            self.assertEqual(json.loads((path / "result.json").read_text()), result)
            with (path / "best.csv").open() as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(float(row["weighted_fitness"]), result["weighted_fitness"])
            self.assertEqual(json.loads(row["best_recipe"]), result["best"]["recipe"])
            (path / "notes.txt").write_text("keep")
            failed = {**result, "status": "failed", "best": None}
            e.write_results(failed, root)
            self.assertFalse((path / "best.csv").exists())
            self.assertFalse((path / "pareto.json").exists())
            self.assertEqual((path / "notes.txt").read_text(), "keep")

    def test_interruption_propagates_after_stale_artifacts_are_removed(self):
        from contextlib import redirect_stdout
        import io
        argv = ["run.py", "--algorithm", "greedy", "--circuit", str(self.circuit)]
        prepare = prepare_run
        with tempfile.TemporaryDirectory() as root:
            experiment = describe_experiment("greedy", self.circuit, asdict(e.Config()))
            path = prepare(experiment, root)
            (path / "best.csv").write_text("old success")
            with patch.object(sys, "argv", argv), \
                 patch("common.experiments.prepare_run", side_effect=lambda x: prepare(x, root)), \
                 patch.object(e.abc, "evaluate_recipe", side_effect=KeyboardInterrupt), \
                 redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                runner.main()
            self.assertFalse((path / "best.csv").exists())
            self.assertFalse((path / "result.json").exists())
            self.assertTrue((path / "config.json").exists())

    @unittest.skipUnless(shutil.which("yosys-abc"), "requires yosys-abc")
    def test_real_cli_overrides_metadata_logging_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "custom.yaml"
            config_path.write_text(json.dumps({"recipe_length": 1, "seed": 99}))
            command = [sys.executable, str(ROOT / "run.py"), "--algorithm", "greedy",
                       "--circuit", str(self.circuit), "--config", str(config_path),
                       "--recipe-length", "2", "--seed", "7", "--max-tool-calls", "32",
                       "--checkpoint-interval", "10", "--weight-area", "0.8", "--weight-delay", "0.2",
                       "--lut-k", "6", "--abc-binary", "yosys-abc", "--timeout", "none",
                       "--optimization-mode", "weighted_sum"]
            previous = None
            for _ in range(2):
                process = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                path = Path(directory) / "results/greedy/iscas85/c17/len2/seed_7"
                result = json.loads((path / "result.json").read_text())
                if previous:
                    self.assertEqual(previous["best"], result["best"])
                previous = result
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["tool_calls"], 32)
                self.assertEqual(result["baseline_tool_calls"], 1)
                self.assertEqual(result["config"]["weight_area"], 0.8)
                self.assertIsNone(result["config"]["timeout"])
                metadata = json.loads((path / "config.json").read_text())
                self.assertEqual(metadata, result["experiment"])
                self.assertEqual(metadata["config"], result["config"])
                self.assertEqual(metadata["abc_executable"]["resolved"], result["provenance"]["abc_executable"])
                events = [json.loads(line) for line in (path / "progress.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["best"], result["best"])
                self.assertTrue(all(event["algorithm"] == "greedy" for event in events))
                self.assertTrue(all(event["circuit"] == metadata["circuit_identity"] for event in events))
                self.assertIn("depth 0/2", process.stdout)
                self.assertIn("depth 2/2", process.stdout)
                self.assertEqual(len(process.stdout.splitlines()), 5)
                self.assertNotIn("generation", process.stdout)
                self.assertFalse((path / "pareto.json").exists())
            for extra in (["--optimization-mode", "pareto"], ["--population-size", "4"]):
                process = subprocess.run(command + extra, cwd=directory, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
