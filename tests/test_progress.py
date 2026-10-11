"""Measured checkpoints, successful candidate counts, and sequence budgets."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.nsga2 import engine as e
from common.progress import CheckpointScheduler


class ProgressTests(unittest.TestCase):
    def config(self, **overrides):
        return replace(e.Config(recipe_length=3, population_size=2, offspring_count=2,
                                generations=0, checkpoint_interval=2, max_tool_calls=6),
                       **overrides)

    def evaluate(self, circuit, recipe, *, budget, **kwargs):
        budget.check()
        budget.tool_calls += 1
        budget.abc_process_invocations += 1
        # The second candidate improves area but worsens levels: scalar fields
        # must come from the fitness winner, never independent component minima.
        metrics = (10, 10) if not recipe else (2, 8) if recipe[0] == 0 else (1, 10)
        return {"metrics": dict(zip(("lut_count", "lut_levels"), metrics))}

    def run_engine(self, config, evaluator=None):
        events = []
        with patch.object(e.random.Random, "randrange", side_effect=[0]*3 + [1]*3), \
             patch.object(e.abc, "evaluate_recipe", side_effect=evaluator or self.evaluate):
            result = e.run("unused.aig", config, progress=events.append)
        return result, events, [p for p in events if p["event"] == "checkpoint"]

    def test_schema_thresholds_and_weighted_best(self):
        result, events, checkpoints = self.run_engine(self.config())
        self.assertEqual([p["checkpoint_thresholds"] for p in checkpoints], [[2], []])
        self.assertEqual([p["tool_calls"] for p in checkpoints], [2, 3])
        self.assertEqual([p["evaluations"] for p in checkpoints], [1, 2])
        self.assertEqual([p["best_fitness"] for p in checkpoints], [0.5, 0.5])
        self.assertEqual([(p["best_lut_count"], p["best_lut_levels"]) for p in checkpoints],
                         [(2, 8), (2, 8)])
        for checkpoint in checkpoints:
            self.assertIs(type(checkpoint["best_fitness"]), float)
            self.assertIs(type(checkpoint["best_lut_count"]), int)
            self.assertIs(type(checkpoint["best_lut_levels"]), int)
        for event in events:
            self.assertIs(type(event["evaluations"]), int)
            self.assertIs(type(event["tool_calls"]), int)
            self.assertIs(type(event["elapsed_seconds"]), float)
            self.assertGreaterEqual(event["elapsed_seconds"], 0)
            self.assertEqual(event["baseline_evaluations"], int(event["event"] != "configuration"))
        self.assertEqual(result["evaluations"], result["successful_evaluations"])
        self.assertEqual(result["abc_process_invocations"], 3)
        self.assertEqual(result["best_fitness"], checkpoints[-1]["best_fitness"])
        # The scheduled event occurs before candidate generation 1 finishes.
        self.assertLess(events.index(checkpoints[0]), next(
            i for i, event in enumerate(events) if event["event"] == "generation"))

    def test_final_checkpoint_below_interval_and_budget_exhaustion(self):
        result, _, checkpoints = self.run_engine(self.config(checkpoint_interval=200, max_tool_calls=2))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["candidate_attempts"], 2)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["checkpoint_thresholds"], [])
        self.assertEqual(checkpoints[0]["status"], "budget_exhausted")
        result, _, checkpoints = self.run_engine(self.config(max_tool_calls=0))
        self.assertEqual(result["baseline_evaluations"], 0)
        self.assertEqual(result["evaluations"], 0)
        self.assertEqual(checkpoints[0]["tool_calls"], 0)
        self.assertIsNone(checkpoints[0]["best_fitness"])

    def test_baseline_crossing_counts_one_sequence(self):
        result, _, checkpoints = self.run_engine(self.config(checkpoint_interval=1))
        self.assertEqual([p["tool_calls"] for p in checkpoints], [1, 2, 3])
        self.assertEqual([p["checkpoint_thresholds"] for p in checkpoints], [[1], [2], [3]])
        self.assertEqual(checkpoints[0]["evaluations"], 0)
        self.assertEqual(checkpoints[0]["baseline_evaluations"], 1)
        self.assertIsNone(checkpoints[0]["best_fitness"])
        self.assertEqual(result["tool_calls"], result["evaluations"] + 1)

    def test_pareto_retains_archive_without_scalar_selection(self):
        result, _, checkpoints = self.run_engine(self.config(optimization_mode="pareto"))
        self.assertEqual(result["baseline_evaluations"], 0)
        self.assertEqual(result["abc_process_invocations"], 2)
        for checkpoint in checkpoints:
            for name in ("best_fitness", "best_lut_count", "best_lut_levels"):
                self.assertIsNone(checkpoint[name])
            self.assertEqual(len(checkpoint["pareto"]), checkpoint["pareto_size"])
        self.assertEqual(checkpoints[-1]["pareto"], result["pareto"])
        self.assertIsNotNone(result["best"])  # Legacy CSV representative retained.

    def test_cache_hits_and_final_state(self):
        events = []
        with patch.object(e.random.Random, "randrange", return_value=0), \
             patch.object(e.abc, "evaluate_recipe", side_effect=self.evaluate):
            result = e.run("unused.aig", self.config(generations=2, crossover_rate=0,
                           mutation_rate=0, max_tool_calls=3), progress=events.append)
        checkpoints = [p for p in events if p["event"] == "checkpoint"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["cache_hits"], 5)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual([p["checkpoint_thresholds"] for p in checkpoints], [[2], []])
        self.assertEqual(checkpoints[-1]["cache_hits"], 5)

    def test_failure_does_not_count_success_or_invent_qor(self):
        for uncertain in (False, True):
            calls = 0
            def evaluate(circuit, recipe, *, budget, **kwargs):
                nonlocal calls
                calls += 1
                if calls < 3:
                    return self.evaluate(circuit, recipe, budget=budget, **kwargs)
                budget.check()
                budget.tool_calls = None if uncertain else budget.tool_calls + 1
                raise e.abc.ABCError("failed after actions", completed_actions=0 if uncertain else 3)
            result, events, checkpoints = self.run_engine(self.config(), evaluate)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["evaluations"], 1)
            self.assertEqual(result["tool_calls"], None if uncertain else 3)
            self.assertEqual(checkpoints[-1]["evaluations"], 1)
            self.assertEqual(checkpoints[-1]["best_fitness"], 0.5)
            self.assertEqual(events[-1]["evaluations"], result["evaluations"])
            self.assertIsNone(result["best"])  # Failed runs still write no successful CSV.

    def test_scheduler_deduplicates_and_records_all_thresholds(self):
        scheduler = CheckpointScheduler(2)
        snapshot = dict(tool_calls=7, evaluations=1, best_fitness=None)
        checkpoint = scheduler.checkpoint(snapshot)
        self.assertEqual(checkpoint["checkpoint_thresholds"], [2, 4, 6])
        self.assertIsNone(scheduler.checkpoint(snapshot))
        self.assertIsNone(scheduler.checkpoint({**snapshot, "elapsed_seconds": 100}, final=True))
        final = scheduler.checkpoint({**snapshot, "tool_calls": 8, "evaluations": 2}, final=True)
        self.assertEqual(final["checkpoint_thresholds"], [8])
        self.assertIsNone(scheduler.checkpoint(final, final=True))
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                CheckpointScheduler(value)
            with self.assertRaises(ValueError):
                self.config(checkpoint_interval=value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
