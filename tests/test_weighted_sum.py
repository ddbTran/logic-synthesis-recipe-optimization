"""Scalar selection, shared baseline budget and small real-ABC CLI checks."""

import csv
from contextlib import redirect_stderr
from dataclasses import asdict
import json
import io
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.nsga2 import engine as e
from common import abc
import run as runner


class WeightedSumTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(optimization_mode="weighted_sum", population_size=2,
                      offspring_count=2, generations=1, max_tool_calls=5)
        values.update(overrides)
        return e.Config(**values)

    def fake_evaluator(self, circuit, recipe, *, budget, **kwargs):
        budget.check()
        budget.tool_calls += 1
        metrics = (10, 20) if not recipe else (1 + sum(recipe), 1 + max(recipe))
        return {"metrics": dict(zip(("lut_count", "lut_levels"), metrics))}

    def test_normalized_fitness_and_weight_dependent_ranking(self):
        baseline = {"lut_count": 10, "lut_levels": 20}
        self.assertAlmostEqual(e.weighted_fitness((5, 30), baseline, self.config()), 1)
        self.assertAlmostEqual(e.weighted_fitness((5, 30), baseline,
                               self.config(weight_area=0.8, weight_delay=0.2)), 0.7)
        for area, expected in ((1, 0), (0, 1)):
            config = self.config(weight_area=area, weight_delay=1-area)
            members = [e.Individual((i,), qor, weighted_fitness=e.weighted_fitness(qor, baseline, config))
                       for i, qor in enumerate(((2, 20), (10, 2)))]
            # Deliberately oppose nondominated-rank/crowding metadata.
            members[expected].rank = 9
            self.assertIs(e.tournament(members, random.Random(0), key=e.scalar_key), members[expected])
            self.assertEqual(e.select_survivors(members, 1, key=e.scalar_key), [members[expected]])
        ties = [e.Individual((i,), (9-i, i+1), weighted_fitness=1) for i in (2, 1)]
        self.assertIs(e.tournament(ties, random.Random(0), key=e.scalar_key), ties[1])
        self.assertEqual(e.select_survivors(ties, 1, key=e.scalar_key), [ties[1]])

    def test_weighted_run_drives_parents_survival_and_best(self):
        def evaluate(circuit, recipe, *, budget, **kwargs):
            budget.check()
            budget.tool_calls += 1
            qor = (20, 20) if not recipe else (20, 1) if recipe[0] == 0 else (1, 20)
            return {"metrics": dict(zip(("lut_count", "lut_levels"), qor))}

        for area, gene in ((1, 1), (0, 0)):
            with patch.object(e.random.Random, "randrange", side_effect=[0]*20 + [1]*20), \
                 patch.object(abc, "evaluate_recipe", side_effect=evaluate), \
                 patch.object(e, "nondominated_sort", side_effect=AssertionError("scalar mode ranked a Pareto front")), \
                 patch.object(e, "vary", wraps=e.vary) as variation, \
                 patch.object(e, "select_survivors", wraps=e.select_survivors) as survival:
                result = e.run("unused.aig", self.config(weight_area=area, weight_delay=1-area,
                               crossover_rate=0, mutation_rate=0, max_tool_calls=3))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["best"]["recipe"], [gene]*20)
            self.assertAlmostEqual(result["weighted_fitness"], 0.05)
            self.assertEqual(variation.call_args.args[:2], ((gene,)*20, (gene,)*20))
            self.assertEqual(survival.call_count, 2)
            self.assertTrue(all(c.kwargs["key"] is e.scalar_key for c in survival.call_args_list))
            self.assertEqual(result["pareto"], [])
            self.assertEqual((result["tool_calls"], result["cache_hits"]), (3, 2))

    def test_baseline_and_candidate_share_budget_and_options(self):
        with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator) as tool:
            result = e.run("unused.aig", self.config(max_tool_calls=4, lut_k=5,
                           abc_binary="custom-abc", timeout=12))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["baseline_qor"], {"lut_count": 10, "lut_levels": 20})
        self.assertEqual(result["baseline_evaluations"], 1)
        self.assertEqual(result["successful_evaluations"], 3)
        self.assertEqual(result["candidate_attempts"], 4)
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(tool.call_args_list[0].args[1], [])
        budget = tool.call_args_list[0].kwargs["budget"]
        for call in tool.call_args_list:
            self.assertIs(call.kwargs["budget"], budget)
            self.assertEqual((call.kwargs["lut_k"], call.kwargs["abc_binary"], call.kwargs["timeout"]),
                             (5, "custom-abc", 12))
        for limit in (0, 1, 2):
            with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator) as tool:
                result = e.run("unused.aig", self.config(max_tool_calls=limit))
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(result["tool_calls"], limit)
            self.assertEqual(tool.call_count, limit)
            self.assertEqual(result["successful_evaluations"], max(0, limit-1))
            if limit <= 1:
                self.assertIsNone(result["best"])

    def test_cache_is_free_after_baseline(self):
        events = []
        with patch.object(e.random.Random, "randrange", return_value=0), \
             patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator) as tool:
            result = e.run("unused.aig", self.config(max_tool_calls=2, generations=3,
                           crossover_rate=0, mutation_rate=0), progress=events.append)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(tool.call_count, 2)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["cache_hits"], 7)
        self.assertEqual(result["candidate_attempts"], 8)
        self.assertEqual(events[-1]["tool_calls"], 2)
        self.assertEqual(events[-1]["cache_hits"], 7)

    def test_invalid_weights_and_modes_precede_abc(self):
        invalid = [(float("nan"), 0.5), (float("inf"), 0.5), (-0.1, 1.1),
                   (0.5, -1), (0, 0), (0.4, 0.4), ("0.5", 0.5), (True, 0)]
        with patch.object(abc, "evaluate_recipe") as tool:
            for mode in ("pareto", "weighted_sum"):
                for area, delay in invalid:
                    with self.subTest(mode=mode, area=area, delay=delay), self.assertRaises(ValueError):
                        e.run("unused.aig", self.config(optimization_mode=mode,
                              weight_area=area, weight_delay=delay))
            with self.assertRaises(ValueError):
                self.config(optimization_mode="unknown")
            tool.assert_not_called()
        self.config(weight_area=0.5, weight_delay=0.5000000001)
        with patch.object(sys, "argv", ["run.py", "--algorithm", "nsga2", "--circuit", "unused.aig",
                                       "--weight-area", "nan"]), \
             patch.object(abc, "evaluate_recipe") as tool, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            tool.assert_not_called()

    def test_nonpositive_baseline_fails_before_candidates(self):
        for qor in ((0, 1), (1, 0)):
            def evaluate(circuit, recipe, *, budget, **kwargs):
                budget.tool_calls += 1
                return {"metrics": dict(zip(("lut_count", "lut_levels"), qor))}
            with patch.object(abc, "evaluate_recipe", side_effect=evaluate) as tool:
                result = e.run("unused.aig", self.config())
            self.assertEqual(result["status"], "failed")
            self.assertIn("must be positive", result["failure"]["message"])
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["candidate_attempts"], 0)
            self.assertIsNone(result["weighted_fitness"])
            tool.assert_called_once()

    def test_baseline_and_candidate_timeouts_count_launched_processes(self):
        evaluator = abc.evaluate_recipe
        circuit = ROOT / "benchmarks/converted/iscas85/c17.aig"
        for fail_baseline in (True, False):
            process = Mock()
            process.communicate.side_effect = [subprocess.TimeoutExpired("abc", 0.01), ("", None)]
            def evaluate(path, recipe, **kwargs):
                if not recipe and not fail_baseline:
                    return self.fake_evaluator(path, recipe, **kwargs)
                return evaluator(path, recipe, **kwargs)
            events = []
            with patch.object(abc, "evaluate_recipe", side_effect=evaluate), \
                 patch.object(abc.subprocess, "Popen", return_value=process):
                result = e.run(circuit, self.config(timeout=0.01), progress=events.append)
            self.assertEqual(result["status"], "failed")
            self.assertIn("timed out", result["failure"]["message"])
            self.assertEqual(result["tool_calls"], 1 if fail_baseline else 2)
            self.assertEqual(events[-1]["tool_calls"], result["tool_calls"])
            self.assertIsNone(result["best"])

    def test_fixed_seed_reproducibility(self):
        results = []
        for _ in range(2):
            with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator):
                results.append(e.run("unused.aig", self.config(generations=3, max_tool_calls=20)))
        for key in ("best", "weighted_fitness", "baseline_qor", "tool_calls", "cache_hits", "candidate_attempts"):
            self.assertEqual(results[0][key], results[1][key])

    def test_cli_config_mode_and_weight_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(json.dumps(asdict(self.config(optimization_mode="pareto", generations=0,
                                                         max_tool_calls=3, weight_area=0.9, weight_delay=0.1))))
            for mode, extra, calls in (("pareto", [], 2),
                                       ("weighted_sum", ["--optimization-mode", "weighted_sum"], 3)):
                process = subprocess.run([
                    sys.executable, str(ROOT / "run.py"), "--algorithm", "nsga2",
                    "--circuit", str(ROOT / "benchmarks/converted/iscas85/c17.aig"),
                    "--config", str(path), "--weight-area", "0.7", "--weight-delay", "0.3", *extra],
                    cwd=directory, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                results = [json.loads(p.read_text()) for p in Path(directory).glob("results/nsga2/**/result.json")]
                result, = (r for r in results if r["optimization_mode"] == mode)
                self.assertEqual((result["weight_area"], result["weight_delay"]), (0.7, 0.3))
                self.assertEqual(result["tool_calls"], calls)
                self.assertEqual(result["status"], "completed")

    def test_real_cli_both_modes_ctrl_c17_and_artifacts(self):
        for circuit, baseline in (("benchmarks/epfl/random_control/ctrl.aig", (29, 2)),
                                  ("benchmarks/converted/iscas85/c17.aig", (2, 1))):
            for mode in ("pareto", "weighted_sum"):
                with self.subTest(circuit=circuit, mode=mode), tempfile.TemporaryDirectory() as directory:
                    process = subprocess.run([
                        sys.executable, str(ROOT / "run.py"), "--algorithm", "nsga2",
                        "--circuit", str(ROOT / circuit),
                        *(["--optimization-mode", mode] if mode == "pareto" else []),
                        "--weight-area", "0.8", "--weight-delay", "0.2", "--seed", "7",
                        "--population-size", "2", "--offspring-count", "2", "--generations", "1",
                        "--max-tool-calls", "4", "--mutation-rate", "1"], cwd=directory,
                        capture_output=True, text=True)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    result_path, = Path(directory).glob("results/nsga2/**/result.json")
                    result = json.loads(result_path.read_text())
                    metadata = json.loads((result_path.parent / "config.json").read_text())
                    self.assertEqual(metadata, result["experiment"])
                    self.assertEqual(metadata["config"], result["config"])
                    self.assertEqual(metadata["config"]["seed"], 7)
                    self.assertEqual(metadata["abc_executable"]["resolved"], result["provenance"]["abc_executable"])
                    self.assertEqual(metadata["abc_executable"]["sha256"], result["provenance"]["abc_sha256"])
                    self.assertEqual(result["optimization_mode"], mode)
                    self.assertEqual(result["config"], asdict(e.Config(optimization_mode=mode,
                        weight_area=0.8, weight_delay=0.2, seed=7, population_size=2,
                        offspring_count=2, generations=1, max_tool_calls=4, mutation_rate=1)))
                    self.assertEqual(result["tool_calls"], 4)
                    self.assertEqual(result["candidate_attempts"], 4)
                    with (result_path.parent / "best.csv").open() as stream:
                        row, = csv.DictReader(stream)
                    self.assertEqual(row["optimization_mode"], mode)
                    self.assertEqual(int(row["tool_calls"]), result["tool_calls"])
                    self.assertEqual(json.loads(row["best_recipe"]), result["best"]["recipe"])
                    events = [json.loads(line) for line in
                              (result_path.parent / "progress.jsonl").read_text().splitlines()]
                    self.assertTrue(all(event["optimization_mode"] == mode for event in events))
                    self.assertTrue(all(event["algorithm"] == "nsga2" for event in events))
                    self.assertTrue(all(event["circuit"] == metadata["circuit_identity"] for event in events))
                    self.assertTrue(all(event["results_directory"] == str(result_path.parent.relative_to(directory))
                                        for event in events))
                    self.assertTrue(all(event["effective_config"] == result["config"] for event in events))
                    self.assertEqual(events[-1]["best"], result["best"])
                    self.assertEqual(events[-1]["tool_calls"], 4)
                    self.assertIn(f"mode={mode}", process.stdout)
                    self.assertNotIn("nsga2 baseline", process.stdout)
                    self.assertNotIn("nsga2 improvement", process.stdout)
                    self.assertNotIn("nsga2 checkpoint", process.stdout)
                    self.assertNotIn("nsga2 summary", process.stdout)
                    self.assertNotIn("LUTs=n/a", process.stdout)
                    if mode == "weighted_sum":
                        self.assertEqual(result["status"], "budget_exhausted")
                        self.assertEqual(tuple(result["baseline_qor"].values()), baseline)
                        self.assertEqual((result["baseline_evaluations"], result["successful_evaluations"]), (1, 3))
                        expected = e.weighted_fitness((result["best"]["lut_count"], result["best"]["lut_levels"]),
                                                     result["baseline_qor"], self.config(weight_area=0.8, weight_delay=0.2))
                        self.assertAlmostEqual(result["weighted_fitness"], expected)
                        self.assertAlmostEqual(float(row["weighted_fitness"]), expected)
                        self.assertEqual(events[-1]["weighted_fitness"], expected)
                        self.assertEqual(json.loads((result_path.parent / "best.json").read_text()), result["best"])
                        self.assertFalse((result_path.parent / "pareto.json").exists())
                        self.assertTrue(all("pareto_size" not in event for event in events))
                        self.assertNotIn("pareto=", process.stdout)
                    else:
                        self.assertEqual(result["status"], "completed")
                        self.assertIsNone(result["baseline_qor"])
                        self.assertEqual(result["successful_evaluations"], 4)
                        self.assertTrue((result_path.parent / "pareto.json").exists())
                    print(f"SMOKE {Path(circuit).stem} {mode}: calls={result['tool_calls']}, "
                          f"LUTs={result['best']['lut_count']}, levels={result['best']['lut_levels']}, "
                          f"fitness={result['weighted_fitness']}, status={result['status']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
