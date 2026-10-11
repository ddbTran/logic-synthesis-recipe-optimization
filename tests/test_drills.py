"""Sequential DRiLLS contracts, literal upstream A2C math and real tool smoke tests."""

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
from algorithms.drills import engine as e, model as m
from algorithms.drills.environment import Environment, improvement_reward
from algorithms.drills.features import FEATURE_NAMES, FeatureExtractor, parse_features
from common import abc
from common.actions import TRANSFORMS
from common.experiments import describe_experiment, experiment_path
from common.progress import ProgressLogger

CIRCUIT = ROOT / "benchmarks/epfl/arithmetic/adder.aig"
FEATURES = [4, 2, 10, 20, 3, 0, 0.6, 0.1, 0.3]


def config(**overrides):
    return replace(e.Config(device="cpu", episodes=2, recipe_length=3, max_tool_calls=6), **overrides)


def measurement(circuit, recipe, *, budget, **kwargs):
    budget.check()
    budget.tool_calls += 1
    budget.abc_process_invocations += 1
    return {"metrics": {"lut_count": 100 - 10 * len(recipe), "lut_levels": 10},
            "observation": [*FEATURES[:2], 10 + len(recipe), *FEATURES[3:]],
            "transformation_count": len(recipe)}


class ObservationTests(unittest.TestCase):
    def test_order_and_fractions(self):
        yosys = "Number of cells: 10\n  $and 6\n  $or 1\n  $not 3\n"
        abc_output = "design: i/o = 4/2 lat = 0 nd = 10 edge = 20 lev = 3\n"
        self.assertEqual(parse_features(yosys, abc_output), FEATURES)
        self.assertEqual(FEATURE_NAMES, ("primary_inputs", "primary_outputs", "cells", "edges",
            "logic_levels", "latches", "and_fraction", "or_fraction", "not_fraction"))
        self.assertEqual(parse_features("Number of cells: 0", abc_output)[6:], [0, 0, 0])
        with self.assertRaises(abc.ABCError):
            parse_features("missing stats", abc_output)

    def test_episode_normalization(self):
        normalizer = m.Normalizer()
        np = m.np
        np.testing.assert_array_equal(normalizer.observe_and_normalize(FEATURES), np.zeros(9))
        second = np.asarray(FEATURES) + 2
        np.testing.assert_allclose(normalizer.observe_and_normalize(second), np.ones(9), atol=1e-6)
        np.testing.assert_allclose(normalizer.mean, np.asarray(FEATURES) + 1, atol=1e-6)
        normalizer.reset()
        np.testing.assert_array_equal(normalizer.observe_and_normalize(second), np.zeros(9))

    def test_auxiliary_timeout_and_definite_launch_failure(self):
        extractor = FeatureExtractor(timeout=0.01)
        with patch("algorithms.drills.features.shutil.which", return_value="/fake/yosys"), \
             patch("algorithms.drills.features.subprocess.Popen", side_effect=OSError("cannot exec")):
            with self.assertRaises(abc.ABCError):
                extractor._run("yosys", [], ROOT, "yosys")
        self.assertEqual(extractor.yosys_process_invocations, 0)
        with patch("algorithms.drills.features.shutil.which", return_value="/fake/yosys"), \
             patch("algorithms.drills.features.subprocess.Popen") as launch:
            launch.return_value.communicate.side_effect = [subprocess.TimeoutExpired("yosys", 0.01), ("timeout output", None)]
            with self.assertRaisesRegex(abc.ABCError, "timed out"):
                extractor._run("yosys", [], ROOT, "yosys")
            launch.return_value.kill.assert_called_once()
        self.assertEqual(extractor.yosys_process_invocations, 1)
        self.assertEqual(extractor.failed_processes, 1)
        self.assertEqual(extractor.timed_out_processes, 1)


class ModelTests(unittest.TestCase):
    def test_checkpoint_mapping_and_next_update(self):
        agent = m.Agent(config())
        states = [m.np.zeros(9), m.np.ones(9)]
        trajectory = (states, [0, 1], [0.1, -0.2])
        agent.update(*trajectory)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            agent.save(path)
            restored = m.Agent(config())
            restored.load(path)
        self.assertEqual(restored.updates, 1)
        for net, other in ((agent.actor, restored.actor), (agent.critic, restored.critic)):
            for left, right in zip(net.parameters(), other.parameters()):
                self.assertTrue(m.torch.equal(left, right))
                self.assertEqual(right.device.type, "cpu")
        self.assertTrue(all(value.device.type == "cpu" for state in restored.optimizer.state.values()
                            for value in state.values() if m.torch.is_tensor(value)))
        self.assertEqual(agent.update(*trajectory), restored.update(*trajectory))
        for left, right in zip(agent.actor.parameters(), restored.actor.parameters()):
            self.assertTrue(m.torch.equal(left, right))

    def test_deterministic_seeding_and_training(self):
        def train():
            m.seed_all(7, deterministic=True)
            agent = m.Agent(config(deterministic=True))
            actions = [agent.sample(m.np.zeros(9)), agent.sample(m.np.ones(9))]
            loss = agent.update([m.np.zeros(9), m.np.ones(9)], actions, [0.1, -0.2])
            return actions, loss, [p.detach().clone() for p in agent.actor.parameters()]
        try:
            first, second = train(), train()
            self.assertEqual(first[:2], second[:2])
            self.assertTrue(all(m.torch.equal(a, b) for a, b in zip(first[2], second[2])))
        finally:
            m.seed_all(0)

    def test_literal_loss_and_gradients(self):
        torch = m.torch
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]], requires_grad=True)
        values = torch.tensor([[0.2], [0.7]], requires_grad=True)
        returns = torch.tensor([1., -1.])
        total, actor, critic, advantage = m.losses(probabilities, values, torch.tensor([0, 1]), returns)
        self.assertEqual(tuple(advantage.shape), (2, 2))
        expected_actor = sum(-torch.log(probabilities[j, [0, 1][j]]) * (returns[j] - values[i, 0])
                             for i in range(2) for j in range(2))
        self.assertAlmostEqual(actor.item(), expected_actor.item(), places=6)
        self.assertAlmostEqual(critic.item(), sum((r-v)**2 for v in (0.2, 0.7) for r in (1, -1)), places=6)
        actor.backward(retain_graph=True)
        self.assertGreater(values.grad.abs().sum().item(), 0)  # Source does not detach critic.
        total.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_returns_forward_sampling_and_optimizer_update(self):
        m.np.testing.assert_allclose(m.discounted_returns([1., 2., 3.], 0.5), [2.75, 3.5, 3])
        normalized = m.normalize_returns([1., 2., 3.])
        self.assertAlmostEqual(normalized.mean(), 0)
        self.assertAlmostEqual(normalized.std(), 1)
        m.np.testing.assert_array_equal(m.normalize_returns([0., 0.]), [0., 0.])
        m.seed_all(0)
        agent = m.Agent(config())
        states = [m.np.zeros(9), m.np.ones(9), -m.np.ones(9)]
        tensors = m.torch.as_tensor(m.np.asarray(states), dtype=m.torch.float32)
        self.assertEqual(tuple(agent.actor(tensors).shape), (3, len(TRANSFORMS)))
        self.assertEqual(tuple(agent.critic(tensors).shape), (3, 1))
        with patch.object(m.np.random, "choice", return_value=7) as choice:
            self.assertEqual(agent.sample(states[0]), 7)
            self.assertAlmostEqual(choice.call_args.kwargs["p"].sum(), 1)
        before = [[p.detach().clone() for p in net.parameters()] for net in (agent.actor, agent.critic)]
        training = agent.update(states, [0, 1, 0], [0.1, -0.2, 0.3])
        self.assertTrue(all(m.np.isfinite(list(training.values()))))
        for net, old in zip((agent.actor, agent.critic), before):
            self.assertTrue(any(not m.torch.equal(p, q) for p, q in zip(net.parameters(), old)))
            self.assertTrue(all(p.device.type == "cpu" and p.grad is not None for p in net.parameters()))
        self.assertEqual(agent.updates, 1)
        self.assertTrue(all(value.device.type == "cpu" for state in agent.optimizer.state.values()
                            for value in state.values() if m.torch.is_tensor(value)))


class EnvironmentTests(unittest.TestCase):
    def environment(self, **overrides):
        cfg = config(**overrides)
        env = Environment(CIRCUIT, cfg, abc.ActionBudget(cfg.max_tool_calls), lambda path: FEATURES)
        env.initialize(abc.ActionBudget(1))
        env.reset()
        return env

    def test_reward_baseline_order_repeats_and_best_fitness(self):
        with patch.object(abc, "evaluate_recipe", side_effect=measurement) as evaluate:
            env = self.environment()
            first, reward, _, _ = env.step(4)
            self.assertAlmostEqual(reward, 1 - first["weighted_fitness"])
            second, reward, _, _ = env.step(4)
            self.assertEqual(second["recipe"], [4, 4])
            self.assertEqual(reward, first["weighted_fitness"] - second["weighted_fitness"])
            self.assertEqual(env.best, second)
            baseline = env.baseline.copy()
            self.assertEqual(env.reset(), FEATURES)
            self.assertEqual(env.baseline, baseline)
            self.assertEqual(env.previous_fitness, 1)
            env.step(2)
            self.assertEqual(env.best, second)
            self.assertTrue(all(call.args[0] == CIRCUIT for call in evaluate.call_args_list))
            self.assertEqual([call.args[1] for call in evaluate.call_args_list], [(), (4,), (4, 4), (2,)])
        self.assertEqual(improvement_reward(0.4, 0.7), 0.4 - 0.7)

    def test_weighted_fitness_normalization(self):
        with patch.object(abc, "evaluate_recipe", side_effect=measurement):
            env = self.environment(weight_area=0.8, weight_delay=0.2)
            member, reward, _, _ = env.step(0)
            self.assertAlmostEqual(member["weighted_fitness"], 0.8*0.9 + 0.2)
            self.assertAlmostEqual(reward, 0.08)

    def test_cache_zero_cost_and_order_at_full_budget(self):
        with patch.object(abc, "evaluate_recipe", side_effect=measurement) as evaluate:
            env = self.environment(max_tool_calls=2)
            env.evaluate([0, 4, 0])
            env.evaluate([4, 0, 0])
            cached = env.evaluate([0, 4, 0])
            cached["observation"][0] = 999
            self.assertEqual(env.evaluate([0, 4, 0])["observation"][0], 4)
            self.assertEqual(env.budget.tool_calls, 2)
            self.assertEqual(evaluate.call_count, 3)  # Includes separate baseline.
            with self.assertRaises(abc.BudgetExceeded):
                env.evaluate([1])
            self.assertEqual(evaluate.call_count, 3)

    def test_failure_and_timeout_accounting(self):
        with patch.object(abc, "evaluate_recipe", side_effect=measurement):
            env = self.environment()
        def fail(circuit, recipe, *, budget, **kwargs):
            budget.tool_calls += 1
            budget.abc_process_invocations += 1
            raise abc.ABCError("ABC timed out", completed_actions=1)
        with patch.object(abc, "evaluate_recipe", side_effect=fail):
            with self.assertRaises(abc.ABCError):
                env.step(0)
        self.assertEqual((env.failed_evaluations, env.timed_out_evaluations, env.budget.tool_calls), (1, 1, 1))
        self.assertEqual(env.confirmed_failed_transformations, 1)
        self.assertEqual(env.recipe, ())
        self.assertFalse(env.cache)


class EpisodeTests(unittest.TestCase):
    def execute(self, cfg, actions):
        events = []
        with patch.object(abc, "evaluate_recipe", side_effect=measurement) as evaluate, \
             patch.object(m.Agent, "sample", side_effect=actions):
            result = e.run(CIRCUIT, cfg, progress=events.append)
        return result, events, evaluate

    def test_global_budget_incomplete_trajectory_and_checkpoints(self):
        result, events, evaluate = self.execute(config(max_tool_calls=4, checkpoint_interval=2), [0, 1, 2, 3, 4])
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual((result["training_updates"], result["completed_episodes"], result["incomplete_episodes"]), (1, 1, 1))
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(evaluate.call_count, 5)
        self.assertEqual(result["best"]["recipe"], [0, 1, 2])
        self.assertFalse(result["episode_history"][-1]["trained"])
        self.assertEqual(result["episode_history"][-1]["recipe"], [3])
        checkpoints = [event for event in events if event["event"] == "checkpoint"]
        self.assertEqual([event["tool_calls"] for event in checkpoints], [2, 4, 4])
        self.assertEqual([event["checkpoint_thresholds"] for event in checkpoints], [[2], [4], []])
        self.assertEqual(events[-1]["status"], "budget_exhausted")

    def test_complete_cached_episode_and_reset(self):
        result, _, evaluate = self.execute(config(episodes=3, recipe_length=2, max_tool_calls=2), [0, 0, 0, 0, 1])
        self.assertEqual(result["training_updates"], 2)
        self.assertEqual(result["cache_hits"], 2)
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(result["episode_history"][0]["rewards"], result["episode_history"][1]["rewards"])
        self.assertEqual(result["episode_history"][2]["length"], 0)

    def test_zero_budget_and_baseline_failure(self):
        result, _, evaluate = self.execute(config(max_tool_calls=0), [0])
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(result["training_updates"], 0)
        with patch.object(abc, "evaluate_recipe", side_effect=abc.ABCError("missing ABC")):
            failed = e.run(CIRCUIT, config())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["baseline_tool_calls"], 0)
        self.assertEqual(failed["failed_baseline_evaluations"], 1)

    def test_candidate_failure_aborts_without_partial_update(self):
        def fail(circuit, recipe, **kwargs):
            measured = measurement(circuit, recipe, **kwargs)
            if len(recipe) == 2:
                raise abc.ABCError("ABC timed out", completed_actions=1)
            return measured
        with patch.object(abc, "evaluate_recipe", side_effect=fail) as evaluate, \
             patch.object(m.Agent, "sample", return_value=0):
            result = e.run(CIRCUIT, config())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["training_updates"], 0)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(result["timed_out_evaluations"], 1)
        self.assertEqual(result["best_so_far"]["recipe"], [0])
        self.assertFalse(result["episode_history"][0]["trained"])


class ResultTests(unittest.TestCase):
    def test_deterministic_paths_and_all_config_settings(self):
        cfg = config()
        describe = lambda values: describe_experiment("drills", CIRCUIT, values)
        path = experiment_path(describe(asdict(cfg)))
        self.assertEqual(path.parts[:4], ("results", "drills", "epfl", "adder"))
        self.assertEqual(path.parent.name, "ep2_len3")
        self.assertEqual(path.name, "seed_0")
        self.assertEqual(path, experiment_path(describe(dict(reversed(list(asdict(cfg).items()))))))
        for field, value in dict(device="auto", gamma=0.9, max_tool_calls=5, timeout=None,
                                 actor_hidden=(30, 20), abc_binary="other", weight_area=0.6).items():
            self.assertEqual(path, experiment_path(describe({**asdict(cfg), field: value})))
        sibling = experiment_path(describe({**asdict(cfg), "seed": 1}))
        self.assertEqual(sibling.parent, path.parent)

    def test_json_csv_progress_and_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            with ProgressLogger(directory / "progress.jsonl") as progress, \
                 patch.object(abc, "evaluate_recipe", side_effect=measurement), \
                 patch.object(m.Agent, "sample", side_effect=[0, 1, 2, 3, 4, 5]):
                result = e.run(CIRCUIT, config(), progress=progress)
            e.write_results(result, directory=directory)
            restored = json.loads((directory / "result.json").read_text())
            self.assertEqual(restored["training_updates"], 2)
            self.assertEqual(restored["provenance"]["resolved_device"], "cpu")
            self.assertFalse(restored["provenance"]["resume_supported"])
            with (directory / "best.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(json.loads(row["best_recipe"]), result["best"]["recipe"])
            events = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["status"], "completed")
            self.assertEqual(sum(event["event"] == "episode" for event in events), 2)


@unittest.skipUnless(shutil.which("yosys") and shutil.which("yosys-abc"), "ABC/Yosys required")
class ToolTests(unittest.TestCase):
    def test_cpu_end_to_end_updates_both_networks(self):
        changed = []
        original_update = m.Agent.update
        def update(agent, *args):
            before = [[p.detach().clone() for p in net.parameters()] for net in (agent.actor, agent.critic)]
            result = original_update(agent, *args)
            changed.append([any(not m.torch.equal(p, q) for p, q in zip(net.parameters(), old))
                            for net, old in zip((agent.actor, agent.critic), before)])
            return result
        with patch.object(m.Agent, "update", update):
            result = e.run(CIRCUIT, config(episodes=1, recipe_length=3, max_tool_calls=3))
        self.assertEqual(result["status"], "completed", result["failure"])
        self.assertEqual(changed, [[True, True]])
        self.assertEqual(result["training_updates"], 1)
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["auxiliary_yosys_process_invocations"], 4)
        self.assertEqual(result["auxiliary_abc_process_invocations"], 4)
        self.assertEqual(result["total_abc_process_invocations"], 8)
        self.assertGreater(result["baseline_observation"][2], result["baseline_qor"]["lut_count"])
        # Mapping K changes QoR while observations of the same raw state stay equal.
        other = e.run(CIRCUIT, config(episodes=1, recipe_length=1, max_tool_calls=1, lut_k=4))
        self.assertEqual(other["status"], "completed", other["failure"])
        self.assertEqual(other["baseline_observation"], result["baseline_observation"])
        self.assertNotEqual(other["baseline_qor"], result["baseline_qor"])


class DeviceTests(unittest.TestCase):
    def test_auto_cpu_and_explicit_unavailable(self):
        with patch.object(m.torch.cuda, "is_available", return_value=False):
            self.assertEqual(str(m.resolve_device("auto")), "cpu")
            result = e.run(CIRCUIT, config(device="cuda"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("CUDA requested but unavailable", result["failure"]["message"])
        self.assertEqual(result["baseline_tool_calls"], 0)
        self.assertEqual(result["tool_calls"], 0)
        with patch.object(m.torch.cuda, "is_available", return_value=True):
            self.assertEqual(str(m.resolve_device("auto")), "cuda")

    @unittest.skipUnless(m.torch.cuda.is_available(), "CUDA hardware/runtime unavailable")
    def test_real_cuda_training(self):
        agent = m.Agent(config(device="cuda"))
        seen = []
        for net in (agent.actor, agent.critic):
            net.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].device.type))
        with patch.object(m.np.random, "choice", side_effect=AssertionError("CUDA sampling moved to CPU")):
            self.assertIn(agent.sample(m.np.zeros(9)), range(len(TRANSFORMS)))
        before = [[p.detach().clone() for p in net.parameters()] for net in (agent.actor, agent.critic)]
        agent.update([m.np.zeros(9), m.np.ones(9)], [0, 1], [0.1, -0.2])
        self.assertEqual(agent.updates, 1)
        self.assertTrue(seen and all(device == "cuda" for device in seen))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.pt"
            agent.save(path)
            cpu = m.Agent(config(device="cpu"))
            cpu.load(path)
            cpu.update([m.np.zeros(9), m.np.ones(9)], [0, 1], [0.1, -0.2])
            cpu.save(path)
            agent.load(path)
            agent.update([m.np.zeros(9), m.np.ones(9)], [0, 1], [0.1, -0.2])
        for net, old in zip((agent.actor, agent.critic), before):
            self.assertTrue(any(not m.torch.equal(p, q) for p, q in zip(net.parameters(), old)))
        self.assertTrue(all(p.is_cuda and p.grad.is_cuda for net in (agent.actor, agent.critic) for p in net.parameters()))
        self.assertTrue(all(value.is_cuda for state in agent.optimizer.state.values()
                            for value in state.values() if m.torch.is_tensor(value)))
        result = e.run(CIRCUIT, config(device="cuda", episodes=1, recipe_length=3, max_tool_calls=3))
        self.assertEqual(result["status"], "completed", result["failure"])
        self.assertEqual(result["training_updates"], 1)
        self.assertEqual(result["resolved_device"], "cuda")
        self.assertTrue(result["cuda_available"])

    def test_other_engines_do_not_import_torch(self):
        code = "import algorithms.nsga2.engine, algorithms.greedy.engine, run, sys; assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
