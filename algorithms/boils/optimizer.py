# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Adapted from HEBO/BOiLS CASMOPOLITAN; see LICENSE for BSD-3-Clause terms.
"""Sequential acquisition search, adaptive Hamming regions and guided restarts."""

from dataclasses import dataclass
from itertools import product
import random

from common.actions import TRANSFORMS
from .surrogate import ExactSSKGP


@dataclass
class TrustRegion:
    radius: int
    maximum: int
    success_tolerance: int = 3
    failure_tolerance: int = 40
    multiplier: float = 1.5
    successes: int = 0
    failures: int = 0

    def update(self, value, previous_best):
        # Require a strict improvement also when the objective is zero.
        if value < previous_best and value <= previous_best - 1e-3 * abs(previous_best):
            self.successes += 1
            self.failures = 0
        else:
            self.failures += 1
            self.successes = 0
        if self.successes >= self.success_tolerance:
            self.radius = min(self.maximum, max(self.radius+1, int(self.radius*self.multiplier)))
            self.successes = 0
        elif self.failures >= self.failure_tolerance:
            self.radius = int(self.radius/self.multiplier)
            self.failures = 0
        return self.radius < 1


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


class Optimizer:
    def __init__(self, config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.history = []
        self.restart_bests = []
        self.restarts = self.surrogate_fits = 0
        self.initial_center = None
        self._reset()

    def _reset(self):
        c = self.config
        self.local = []
        self.region = TrustRegion(min(c.recipe_length, c.tr_initial_radius), c.recipe_length,
                                  c.tr_success_tolerance, c.tr_failure_tolerance, c.tr_multiplier)

    def random_recipe(self, center=None):
        if center is None:
            return tuple(self.rng.randrange(len(TRANSFORMS)) for _ in range(self.config.recipe_length))
        result = list(center)
        count = self.rng.randint(1, min(len(result), self.region.radius))
        for i in self.rng.sample(range(len(result)), count):
            # Sample a different action so Hamming distance is exactly count.
            action = self.rng.randrange(len(TRANSFORMS)-1)
            result[i] = action + (action >= result[i])
        return tuple(result)

    def pool(self, seen, center=None):
        candidates = set()
        for _ in range(self.config.candidate_pool_size * 4):
            recipe = self.random_recipe(center)
            if recipe not in seen:
                candidates.add(recipe)
            if len(candidates) >= self.config.candidate_pool_size:
                break
        return sorted(candidates)

    def fresh_global(self, seen):
        for _ in range(self.config.candidate_pool_size * 4):
            recipe = self.random_recipe()
            if recipe not in seen:
                return recipe
        # Finite deterministic fallback, including tiny fully enumerated spaces.
        return next((r for r in product(range(len(TRANSFORMS)), repeat=self.config.recipe_length)
                     if r not in seen), None)

    def fit(self, observations):
        # Bound CPU cost, retaining the incumbent and newest local observations.
        best = min(observations, key=lambda p: (p[1], p[0]))
        retained = observations[-(self.config.max_training_points-1):]
        retained = [best] + [p for p in retained if p[0] != best[0]]
        gp = ExactSSKGP(self.config.recipe_length, self.config.ssk_order)
        gp.fit([p[0] for p in retained], [p[1] for p in retained], self.config.gp_training_steps)
        self.surrogate_fits += 1
        return gp

    def restart(self, seen):
        if self.local:
            self.restart_bests.append(min(self.local, key=lambda p: (p[1], p[0])))
        self.restarts += 1
        self._reset()
        pool = self.pool(seen)
        self.initial_center = None
        if pool and self.restart_bests:
            gp = self.fit(self.restart_bests)
            incumbent = min(self.restart_bests, key=lambda p: p[1])[0]
            scores = gp.acquisition(pool, incumbent, kind="lcb")
            self.initial_center = pool[int(scores.argmax())]

    def suggest(self, seen):
        if len(seen) >= len(TRANSFORMS) ** self.config.recipe_length:
            return None
        if len(self.local) < self.config.n_initial:
            if self.initial_center is not None:
                if self.initial_center not in seen:
                    return self.initial_center
                pool = self.pool(seen, self.initial_center)
                if pool:
                    return self.rng.choice(pool)
            return self.fresh_global(seen)
        center = min(self.local, key=lambda p: (p[1], p[0]))[0]
        pool = self.pool(seen, center)
        # Include all one-edit neighbours to avoid spurious local exhaustion.
        for i in range(len(center)):
            for action in range(len(TRANSFORMS)):
                r = center[:i] + (action,) + center[i+1:]
                if r not in seen:
                    pool.append(r)
        pool = sorted(set(pool))
        if not pool:
            self.restart(seen)
            return self.suggest(seen)
        gp = self.fit(self.local)
        scores = gp.acquisition(pool, center)
        # Three starts, bounded steepest-ascent acquisition search in the TR.
        starts = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), pool[i]))[:3]
        winner, value = pool[starts[0]], float(scores[starts[0]])
        for index in starts:
            current, score = pool[index], float(scores[index])
            for _ in range(self.config.acquisition_steps):
                neighbors = set()
                for i in range(len(current)):
                    for action in range(len(TRANSFORMS)):
                        r = current[:i] + (action,) + current[i+1:]
                        if r not in seen and 0 < hamming(r, center) <= self.region.radius:
                            neighbors.add(r)
                neighbors = sorted(neighbors)
                if not neighbors:
                    break
                values = gp.acquisition(neighbors, center)
                j = int(values.argmax())
                if float(values[j]) <= score:
                    break
                current, score = neighbors[j], float(values[j])
            if score > value:
                winner, value = current, score
        return winner

    def observe(self, recipe, value, seen):
        collapse = False
        if len(self.local) >= self.config.n_initial:
            collapse = self.region.update(value, min(v for _, v in self.local))
        observation = (tuple(recipe), value)
        self.local.append(observation)
        self.history.append(observation)
        if collapse:
            self.restart(seen)
