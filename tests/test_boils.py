"""BOiLS numerical, search and framework integration contracts."""

from collections import defaultdict
from dataclasses import replace
from itertools import combinations
import csv
import json
from pathlib import Path
import shutil
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.boils import engine as e
from algorithms.boils.optimizer import Optimizer, TrustRegion, hamming
from algorithms.boils.surrogate import StringSubsequenceKernel, ExactSSKGP, rank_standardise


class BoilsTests(unittest.TestCase):
    circuit = ROOT / "benchmarks/converted/iscas85/c17.aig"

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **kwargs):
        return replace(e.Config(max_tool_calls=9, recipe_length=3, n_initial=3,
                                gp_training_steps=3, candidate_pool_size=8,
                                acquisition_steps=1, checkpoint_interval=3), **kwargs)

    def evaluator(self):
        def evaluate(circuit, recipe, *, budget, **kwargs):
            budget.check()
            budget.tool_calls += 1
            budget.abc_process_invocations += 1
            return {"metrics": {"lut_count": 100 if not recipe else 20+sum(recipe),
                                "lut_levels": 10 if not recipe else 2+recipe[0] % 5}}
        return evaluate

    def execute(self, config=None):
        events = []
        with patch.object(e.abc, "evaluate_recipe", side_effect=self.evaluator()) as launch:
            result = e.run(self.circuit, config or self.config(), progress=events.append)
        return result, events, launch

    def test_ssk_matches_independent_subsequence_enumeration(self):
        strings = [[0, 0, 1, 2], [0, 1, 0, 2], [3, 3, 3, 3]]
        for order in (1, 2, 3):
            def feature(s):
                result = defaultdict(float)
                for indices in combinations(range(len(s)), order):
                    key = tuple(s[i] for i in indices)
                    result[key] += 0.8**order * 0.5**(indices[-1]-indices[0]+1-order)
                return result
            features = [feature(s) for s in strings]
            expected = torch.zeros((3, 3), dtype=torch.float64)
            for i, a in enumerate(features):
                for j, b in enumerate(features):
                    expected[i, j] = sum(v*b.get(k, 0) for k, v in a.items()) / (
                        sum(v*v for v in a.values()) * sum(v*v for v in b.values()))**0.5
            kernel = StringSubsequenceKernel(4, alphabet_size=4, order=order)
            actual = kernel(strings)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(kernel(strings[:1], strings[1:]), expected[:1, 1:])
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(actual).min().detach()), -1e-10)
            if order > 1:
                actual.sum().backward()
                self.assertTrue(torch.isfinite(kernel.raw_gap.grad))
        with self.assertRaises(ValueError):
            kernel([[4, 0, 0, 0]])

    def test_gp_fits_predicts_and_acquires(self):
        x = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        gp = ExactSSKGP(3).fit(x, [0.1, 0.5, 0.9], steps=20)
        mean, variance = gp.predict(x)
        self.assertLess(float(mean[0]), float(mean[1]))
        self.assertLess(float(mean[1]), float(mean[2]))
        self.assertLess(float(variance.max()), 0.1)
        unknown_mean, unknown_variance = gp.predict([(3, 3, 3)])
        self.assertGreater(float(unknown_variance[0]), float(variance.max()))
        self.assertTrue(torch.isfinite(unknown_mean).all())
        for kind in ("ei", "lcb"):
            score = gp.acquisition(x+[(3, 3, 3)], x[0], kind)
            self.assertTrue(torch.isfinite(score).all())
        self.assertGreater(float(gp.acquisition([(3, 3, 3)], x[0])[0]),
                           float(gp.acquisition([x[0]], x[0])[0]))
        torch.testing.assert_close(rank_standardise([1, 1, 3]),
                                  torch.tensor([0., 0., 0.6744897501960817], dtype=torch.float64))

    def test_acquisition_selects_maximum_and_respects_region(self):
        c = self.config(acquisition_steps=0, tr_initial_radius=1)
        optimizer = Optimizer(c)
        seen = {}
        for r, v in [((0, 0, 0), 0.1), ((1, 1, 1), 0.5), ((2, 2, 2), 0.9)]:
            seen[r] = v
            optimizer.observe(r, v, seen)
        class Acquisition:
            def acquisition(self, recipes, incumbent):
                self_recipes.extend(recipes)
                return torch.tensor([sum(r) for r in recipes], dtype=torch.float64)
        self_recipes = []
        with patch.object(optimizer, "fit", return_value=Acquisition()):
            selected = optimizer.suggest(seen)
        self.assertEqual(sum(selected), max(map(sum, self_recipes)))
        self.assertEqual(hamming(selected, (0, 0, 0)), 1)
        self.assertNotIn(selected, seen)
        for _ in range(20):
            self.assertEqual(hamming(optimizer.random_recipe((0, 0, 0)), (0, 0, 0)), 1)

    def test_initial_sampling_and_bounded_training_keep_incumbent(self):
        config = self.config(max_training_points=3)
        optimizer = Optimizer(config)
        rng = random.Random(config.seed)
        self.assertEqual(optimizer.suggest({}), tuple(rng.randrange(16) for _ in range(3)))
        observations = [((i, i, i), float(i)) for i in range(6)]
        with patch("algorithms.boils.optimizer.ExactSSKGP") as model:
            optimizer.fit(observations)
        recipes, values, steps = model.return_value.fit.call_args.args
        self.assertEqual(recipes, [(0, 0, 0), (4, 4, 4), (5, 5, 5)])
        self.assertEqual(values, [0., 4., 5.])
        self.assertEqual(steps, config.gp_training_steps)

    def test_surrogate_failure_is_reported_without_kernel_fallback(self):
        with patch.object(e.abc, "evaluate_recipe", side_effect=self.evaluator()), \
             patch.object(Optimizer, "fit", side_effect=RuntimeError("Cholesky failed")):
            result = e.run(self.circuit, self.config(n_initial=2))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["tool_calls"], 2)
        self.assertIsNotNone(result["best_so_far"])
        self.assertEqual(result["failure"]["message"], "Cholesky failed")
        self.assertEqual(len(result["observations"]), 2)

    def test_region_expands_shrinks_and_restarts(self):
        region = TrustRegion(2, 6, success_tolerance=2, failure_tolerance=2)
        region.update(0.9, 1)
        region.update(0.8, 0.9)
        self.assertEqual(region.radius, 3)
        region.update(1, 0.8)
        region.update(1, 0.8)
        self.assertEqual(region.radius, 2)
        region.radius = 1
        self.assertFalse(region.update(1, 0.8))
        self.assertTrue(region.update(1, 0.8))
        optimizer = Optimizer(self.config(n_initial=1, tr_initial_radius=1, tr_failure_tolerance=1))
        seen = {(0, 0, 0): 1}
        optimizer.observe((0, 0, 0), 1, seen)
        seen[(1, 0, 0)] = 2
        optimizer.observe((1, 0, 0), 2, seen)
        self.assertEqual(optimizer.restarts, 1)
        self.assertEqual(optimizer.restart_bests, [((0, 0, 0), 1)])
        self.assertEqual(optimizer.local, [])
        self.assertNotIn(optimizer.suggest(seen), seen)
        self.assertEqual(optimizer.surrogate_fits, 1)
        self.assertEqual(len(optimizer.history), 2)

    def test_determinism_normalization_best_and_budget(self):
        first, events, launch = self.execute()
        second, _, other_launch = self.execute()
        self.assertEqual(first["observations"], second["observations"])
        self.assertEqual(first["best"], second["best"])
        different, _, _ = self.execute(self.config(seed=4))
        self.assertNotEqual(first["observations"], different["observations"])
        self.assertEqual(first["tool_calls"], 9)
        self.assertEqual(first["baseline_tool_calls"], 1)
        self.assertEqual(first["abc_process_invocations"], 10)
        self.assertEqual(first["status"], "budget_exhausted")
        self.assertGreater(first["surrogate_fits"], 0)
        self.assertEqual(first["cache_hits"], 0)
        self.assertEqual(len({tuple(p["recipe"]) for p in first["observations"]}), 9)
        best = min(first["observations"], key=lambda p: (p["weighted_fitness"], p["recipe"]))
        self.assertEqual(first["best"]["recipe"], best["recipe"])
        self.assertAlmostEqual(first["weighted_fitness"],
                               0.5*first["best"]["lut_count"]/100 + 0.5*first["best"]["lut_levels"]/10)
        for call in launch.call_args_list[1:]:
            self.assertEqual(len(call.args[1]), 3)
            self.assertEqual(call.args[0], self.circuit)
            self.assertEqual(call.kwargs["metric"], "lut")
            self.assertEqual(call.kwargs["lut_k"], 6)
        self.assertEqual([x["tool_calls"] for x in events if x["event"] == "checkpoint"], [3, 6, 9])
        improvements = [x["best_fitness"] for x in events if x["event"] == "improvement"]
        self.assertEqual(improvements, sorted(improvements, reverse=True))

    def test_zero_budget_cache_and_space_exhaustion(self):
        result, _, launch = self.execute(self.config(max_tool_calls=0))
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(launch.call_count, 1)
        self.assertIsNone(result["best"])
        result, _, _ = self.execute(self.config(recipe_length=1, max_tool_calls=20, n_initial=2))
        self.assertEqual(result["tool_calls"], 16)
        self.assertEqual(result["status"], "search_space_exhausted")
        config = self.config(max_tool_calls=1)
        evaluator = e.RecipeEvaluator(self.circuit, config, {"lut_count": 100, "lut_levels": 10}, e.abc.ActionBudget(1))
        with patch.object(e.abc, "evaluate_recipe", side_effect=self.evaluator()) as launch:
            evaluator.evaluate((0, 0, 0))
            evaluator.evaluate([0, 0, 0])
            self.assertEqual(launch.call_count, 1)
            with self.assertRaises(e.abc.BudgetExceeded):
                evaluator.evaluate((1, 0, 0))
        self.assertEqual(evaluator.cache_hits, 1)
        self.assertEqual(evaluator.evaluations, 1)

    def test_failures_preserve_best_so_far_and_accounting(self):
        for uncertain in (False, True):
            normal = self.evaluator()
            calls = 0
            def evaluate(circuit, recipe, *, budget, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    budget.tool_calls = None if uncertain else budget.tool_calls+1
                    budget.abc_process_invocations += 1
                    raise e.abc.ABCError("ABC timed out", output="diagnostic", completed_actions=1)
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
                self.assertIsNotNone(json.loads((path / "result.json").read_text())["best_so_far"])
        with patch.object(e.abc, "evaluate_recipe", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            e.run(self.circuit, self.config())
        self.assertEqual(torch.get_num_threads(), 1)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            e.load_config()
        self.assertEqual(e.load_config(max_tool_calls=9), e.Config(max_tool_calls=9))
        for key, value in (("n_initial", 0), ("ssk_order", 4), ("gp_training_steps", 0),
                           ("max_training_points", 1), ("candidate_pool_size", True),
                           ("tr_multiplier", 1), ("tr_multiplier", float("nan")),
                           ("optimization_mode", "pareto")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.config(**{key: value})
        with self.assertRaises(TypeError):
            e.load_config(max_tool_calls=9, population_size=10)

    @unittest.skipUnless(shutil.which("yosys-abc"), "requires real ABC")
    def test_real_cli_overrides_results_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"custom.yaml"
            path.write_text(json.dumps({"recipe_length": 8, "seed": 99, "max_tool_calls": 100}))
            command = [sys.executable, str(ROOT/"run.py"), "--algorithm", "boils", "--circuit", str(self.circuit),
                       "--config", str(path), "--recipe-length", "3", "--seed", "7", "--max-tool-calls", "6",
                       "--n-initial", "2", "--gp-training-steps", "3", "--candidate-pool-size", "8",
                       "--acquisition-steps", "1", "--checkpoint-interval", "2", "--timeout", "none",
                       "--weight-area", "0.8", "--weight-delay", "0.2"]
            previous = None
            for _ in range(2):
                process = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stderr)
                result_path = Path(directory)/"results/boils/iscas85/c17/len3/seed_7"
                result = json.loads((result_path/"result.json").read_text())
                self.assertEqual(result["tool_calls"], 6)
                self.assertGreater(result["surrogate_fits"], 0)
                self.assertEqual(result["config"]["weight_area"], 0.8)
                self.assertIsNone(result["config"]["timeout"])
                self.assertEqual(json.loads((result_path/"config.json").read_text()), result["experiment"])
                self.assertEqual(json.loads((result_path/"best.json").read_text()), result["best"])
                with (result_path/"best.csv").open() as stream:
                    row, = csv.DictReader(stream)
                self.assertEqual(float(row["weighted_fitness"]), result["weighted_fitness"])
                events = [json.loads(line) for line in (result_path/"progress.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["best"], result["best"])
                self.assertEqual(events[-1]["status"], "budget_exhausted")
                if previous is not None:
                    self.assertEqual(previous["observations"], result["observations"])
                previous = result
            process = subprocess.run(command+["--optimization-mode", "pareto"], cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
