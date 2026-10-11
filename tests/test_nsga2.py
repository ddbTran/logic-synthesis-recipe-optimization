"""Focused NSGA-II checks; no conversion or equivalence work."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.nsga2 import engine as e
from common import abc
from common.actions import TRANSFORMS


class NSGA2Tests(unittest.TestCase):
    def member(self, values, index=0):
        return e.Individual((index,), values)

    def fake_evaluator(self, circuit, recipe, *, budget, **kwargs):
        budget.check(len(recipe))
        budget.tool_calls += len(recipe)
        return {"metrics": {"lut_count": sum(recipe), "lut_levels": 100 - sum(recipe)}}

    def config(self, **overrides):
        values = dict(recipe_length=3, population_size=4, offspring_count=3,
                      generations=3, max_tool_calls=100, seed=42)
        values.update(overrides)
        return e.Config(**values)

    def test_dominance_and_fronts(self):
        self.assertTrue(e.dominates((1, 2), (1, 3)))
        self.assertFalse(e.dominates((1, 2), (1, 2)))
        self.assertFalse(e.dominates((1, 3), (2, 2)))
        pop = [self.member(v, i) for i, v in enumerate([(1, 3), (2, 2), (3, 3), (4, 4), (1, 3)])]
        fronts = e.nondominated_sort(pop)
        self.assertEqual([[p.recipe[0] for p in f] for f in fronts], [[0, 1, 4], [2], [3]])
        self.assertEqual(e.nondominated_sort([]), [])

    def test_crowding_and_elitist_survival(self):
        pop = [self.member(v, i) for i, v in enumerate([(1, 4), (2, 3), (3, 2), (4, 1)])]
        e.assign_crowding(pop)
        self.assertEqual(pop[0].crowding, float("inf"))
        self.assertAlmostEqual(pop[1].crowding, 4 / 3)
        chosen = e.select_survivors(pop + [self.member((9, 9), 9)], 2)
        self.assertEqual([p.recipe for p in chosen], [(0,), (3,)])
        equal = [self.member((1, 1), i) for i in range(4)]
        e.assign_crowding(equal)
        self.assertEqual([p.crowding for p in equal], [float("inf"), 0, 0, float("inf")])

    def test_variation_and_tournament(self):
        a, b = (0, 0, 0), (15, 15, 15)
        self.assertEqual(e.vary(a, b, random.Random(1), self.config(crossover_rate=0, mutation_rate=0)), [a, b])
        children = e.vary(a, b, random.Random(2), self.config(crossover_rate=1, mutation_rate=0))
        for i in range(3):
            self.assertEqual(sorted(c[i] for c in children), [0, 15])
        children = e.vary(a, b, random.Random(3), self.config(mutation_rate=1))
        self.assertTrue(all(len(c) == 3 and all(0 <= g < len(TRANSFORMS) for g in c) for c in children))
        self.assertNotEqual(children, [a, b])
        first, second = self.member((1, 1)), self.member((2, 2))
        second.rank = 1
        self.assertIs(e.tournament([first, second], random.Random(1)), first)
        self.assertIs(e.TRANSFORMS, TRANSFORMS)

    def test_repeatability_population_and_odd_offspring(self):
        with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator):
            first = e.run("unused.aig", self.config())
            second = e.run("unused.aig", self.config())
        for key in ("best", "pareto", "tool_calls", "cache_hits", "completed_generations"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first["completed_generations"], 3)
        self.assertEqual(first["successful_evaluations"] + first["cache_hits"], 4 + 3 * 3)
        pairs = [(p["lut_count"], p["lut_levels"]) for p in first["pareto"]]
        self.assertFalse(any(e.dominates(a, b) for a in pairs for b in pairs))
        self.assertEqual(first["best"], min(first["pareto"], key=lambda p: (p["lut_count"], p["lut_levels"], p["recipe"])))

    def test_shared_budget_whole_recipe_and_exhaustion(self):
        seen = []
        def evaluate(*args, **kwargs):
            seen.append(kwargs["budget"])
            return self.fake_evaluator(*args, **kwargs)
        with patch.object(abc, "evaluate_recipe", side_effect=evaluate) as tool:
            result = e.run("unused.aig", self.config(max_tool_calls=8))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["tool_calls"], 6)
        self.assertEqual(tool.call_count, 2)
        self.assertTrue(all(b is seen[0] for b in seen))
        self.assertIsNotNone(result["best"])
        with patch.object(abc, "evaluate_recipe") as tool:
            result = e.run("unused.aig", self.config(max_tool_calls=2))
        tool.assert_not_called()
        self.assertIsNone(result["best"])
        self.assertEqual(result["tool_calls"], 0)

    def test_repeated_recipe_real_cache_hit(self):
        with patch.object(e.random.Random, "randrange", return_value=0), \
             patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator) as tool:
            result = e.run("unused.aig", self.config(crossover_rate=0, mutation_rate=0, max_tool_calls=3))
        self.assertEqual(tool.call_count, 1)
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["best"]["recipe"], [0, 0, 0])
        self.assertEqual(result["cache_hits"], 12)
        self.assertEqual(result["status"], "completed")

    def test_failures_abort_and_preserve_diagnostics(self):
        for uncertain in (False, True):
            count = 0
            def evaluate(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 1:
                    return self.fake_evaluator(*args, **kwargs)
                if uncertain:
                    kwargs["budget"].tool_calls = None
                raise abc.ABCError("timeout or malformed metrics", "raw failure", 1)
            with patch.object(abc, "evaluate_recipe", side_effect=evaluate):
                result = e.run("unused.aig", self.config())
            self.assertEqual(count, 2)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["best"])
            self.assertEqual(result["failure"]["abc_output"], "raw failure")
            self.assertEqual(result["tool_calls"], None if uncertain else 3)
            with tempfile.TemporaryDirectory() as directory:
                output = e.write_results(result, directory)
                self.assertFalse((output / "best.csv").exists())
                self.assertTrue(json.loads((output / "pareto.json").read_text()))

    def test_output_schema(self):
        with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator):
            result = e.run("unused.aig", self.config())
        with tempfile.TemporaryDirectory() as directory:
            output = e.write_results(result, directory)
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertEqual(json.loads((output / "pareto.json").read_text()), result["pareto"])
            with (output / "best.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(json.loads(row["best_recipe"]), result["best"]["recipe"])
            self.assertEqual(json.loads(row["config"]), result["config"])
            self.assertEqual(json.loads(row["provenance"]), result["provenance"])
            self.assertEqual(int(row["tool_calls"]), result["tool_calls"])
            self.assertGreaterEqual(float(row["runtime_seconds"]), 0)

    def test_configuration(self):
        self.assertEqual(asdict(e.load_config()), asdict(e.Config()))
        for values in ({"population_size": 1}, {"recipe_length": 0}, {"generations": -1},
                       {"offspring_count": 0}, {"timeout": float("nan")},
                       {"mutation_rate": 2}, {"objective": "weighted_sum"}, {"max_tool_calls": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(**values)

    def test_cli_failure_and_no_affordable_candidate(self):
        runner = Path(e.__file__).resolve().parents[2] / "run.py"
        circuit = runner.parent / "benchmarks/converted/iscas85/c17.aig"
        for extra, status in ((["--abc-binary", "/nonexistent/abc"], "failed"),
                              (["--max-tool-calls", "0"], "budget_exhausted")):
            with tempfile.TemporaryDirectory() as directory:
                process = subprocess.run(
                    [sys.executable, str(runner), "--algorithm", "nsga2", "--circuit", str(circuit),
                     "--recipe-length", "2", "--population-size", "2", "--generations", "0", *extra],
                    cwd=directory, capture_output=True, text=True, check=False)
                self.assertEqual(process.returncode, 1, process.stderr)
                result_path, = Path(directory).glob("results/nsga2-*/result.json")
                result = json.loads(result_path.read_text())
                self.assertEqual(result["status"], status)
                self.assertEqual(result["tool_calls"], 0)
                self.assertIsNone(result["best"])
                self.assertFalse((result_path.parent / "best.csv").exists())

    def test_other_engines_remain_placeholders(self):
        result = subprocess.run([sys.executable, "run.py", "--algorithm", "drills", "--circuit", "missing.aig"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not implemented", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
