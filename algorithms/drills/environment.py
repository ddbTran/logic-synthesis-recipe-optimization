"""Fixed-baseline episodic replay environment with a run-local raw-state cache."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

from algorithms.greedy.engine import valid_metrics
from algorithms.nsga2.engine import weighted_fitness
from common import abc
from common.actions import TRANSFORMS
from common.experiments import fingerprint
from .features import validate_features


def improvement_reward(previous_fitness, current_fitness):
    if not all(math.isfinite(x) for x in (previous_fitness, current_fitness)):
        raise ValueError("Reward requires finite weighted fitness values")
    return previous_fitness - current_fitness


class Environment:
    def __init__(self, circuit, config, budget, extractor):
        self.circuit, self.config, self.budget, self.extractor = circuit, config, budget, extractor
        self.baseline = self.baseline_features = None
        self.cache = {}
        # Context makes cached QoR AND raw features specific to input, tools and config.
        self.context = (str(Path(circuit).resolve()), fingerprint(circuit),
                        json.dumps(asdict(config), sort_keys=True),
                        fingerprint(shutil.which(config.abc_binary)),
                        fingerprint(shutil.which(config.yosys_binary)))
        self.evaluations = self.candidate_attempts = self.cache_hits = 0
        self.failed_evaluations = self.timed_out_evaluations = 0
        self.transformation_executions = self.confirmed_failed_transformations = 0
        self.best = self.best_complete = None
        self.recipe = ()
        self.previous_fitness = None

    def measure(self, recipe, budget):
        return abc.evaluate_recipe(self.circuit, recipe, budget=budget, metric="lut",
            lut_k=self.config.lut_k, timeout=self.config.timeout, abc_binary=self.config.abc_binary,
            observation_extractor=self.extractor)

    def initialize(self, baseline_budget):
        result = self.measure((), baseline_budget)
        self.baseline = valid_metrics(result.get("metrics"), baseline=True)
        self.baseline_features = validate_features(result.get("observation"))

    def reset(self):
        if self.baseline is None:
            raise ValueError("Environment requires a valid fixed baseline")
        self.recipe = ()
        self.previous_fitness = weighted_fitness(
            (self.baseline["lut_count"], self.baseline["lut_levels"]), self.baseline, self.config)
        return list(self.baseline_features)

    def evaluate(self, recipe):
        recipe = tuple(recipe)
        if any(type(a) is not int or not 0 <= a < len(TRANSFORMS) for a in recipe):
            raise ValueError("Invalid shared action index")
        self.candidate_attempts += 1
        key = self.context, recipe
        if key in self.cache:
            self.cache_hits += 1
            member = self.cache[key]
        else:
            self.budget.check()
            try:
                result = self.measure(recipe, self.budget)
                self.transformation_executions += result["transformation_count"]
                if self.budget.tool_calls is None:
                    raise abc.ABCError("Evaluator returned with uncertain budget")
                metrics = valid_metrics(result.get("metrics"))
                features = validate_features(result.get("observation"))
                fitness = weighted_fitness((metrics["lut_count"], metrics["lut_levels"]),
                                           self.baseline, self.config)
                if not math.isfinite(fitness):
                    raise ValueError("Non-finite weighted fitness")
                member = {"recipe": list(recipe), **metrics, "weighted_fitness": fitness,
                          "observation": features}
            except (abc.ABCError, OSError, ValueError, OverflowError) as exc:
                self.failed_evaluations += 1
                self.timed_out_evaluations += int("timed out" in str(exc).lower())
                self.confirmed_failed_transformations += getattr(exc, "completed_actions", 0)
                raise
            self.cache[key] = member
            self.evaluations += 1
        # Use an immutable view in the cache; callers receive fresh lists.
        return {**member, "recipe": list(member["recipe"]), "observation": list(member["observation"])}

    def step(self, action):
        recipe = self.recipe + (action,)
        member = self.evaluate(recipe)
        reward = improvement_reward(self.previous_fitness, member["weighted_fitness"])
        self.recipe, self.previous_fitness = recipe, member["weighted_fitness"]
        key = lambda m: (m["weighted_fitness"], tuple(m["recipe"]))
        improved = self.best is None or key(member) < key(self.best)
        if improved:
            self.best = member
        done = len(recipe) == self.config.recipe_length
        if done and (self.best_complete is None or key(member) < key(self.best_complete)):
            self.best_complete = member
        return member, reward, done, improved
