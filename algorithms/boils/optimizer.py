# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Adapted from HEBO/BOiLS CASMOPOLITAN; see LICENSE for BSD-3-Clause terms.
"""Sequential acquisition search, adaptive Hamming regions and guided restarts."""

from dataclasses import dataclass
from itertools import product
import random

import numpy as np

from common.actions import TRANSFORMS
from .surrogate import ExactSSKGP, TransformedOverlapKernel


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
        if value <= previous_best - 1e-3 * abs(previous_best):
            self.successes += 1
            self.failures = 0
        else:
            self.failures += 1
            self.successes = 0
        if self.successes >= self.success_tolerance:
            self.radius = min(self.maximum, int(self.radius*self.multiplier))
            self.successes = 0
        elif self.failures >= self.failure_tolerance:
            self.radius = int(self.radius/self.multiplier)
            self.failures = 0
        # Upstream restarts at length_min_discrete == 1.
        return self.radius <= 1


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def latin_hypercube_recipes(count, length, rng):
    """Upstream LHS in one-hot coordinates, then categorical argmax warping."""
    dimensions = length * len(TRANSFORMS)
    centers = (1 + 2*np.arange(count)) / (2*count)
    samples = np.empty((count, dimensions))
    for column in range(dimensions):
        samples[:, column] = centers[rng.permutation(count)]
    samples += rng.uniform(-1, 1, samples.shape) / (2*count)
    return [tuple(map(int, row)) for row in samples.reshape(count, length, len(TRANSFORMS)).argmax(-1)]


class Optimizer:
    def __init__(self, config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.lhs_rng = np.random.RandomState(config.seed)
        self.initial_radius = (config.tr_initial_radius if config.tr_initial_radius is not None
                               else max(1, int(0.666*config.recipe_length)))
        self.n_candidates = (config.candidate_pool_size if config.candidate_pool_size is not None
                             else min(100*config.recipe_length, 5000))
        self.history = []
        self.restart_bests = []
        self.restarts = self.surrogate_fits = 0
        self.initial_center = None
        self._reset()
        self.pending = latin_hypercube_recipes(config.n_initial, config.recipe_length, self.lhs_rng)

    def _reset(self):
        c = self.config
        self.local = []
        self.region = TrustRegion(self.initial_radius, c.recipe_length,
                                  c.tr_success_tolerance, c.tr_failure_tolerance, c.tr_multiplier)
        self.pending = []

    def random_recipe(self, center=None):
        if center is None:
            return tuple(self.rng.randrange(len(TRANSFORMS)) for _ in range(self.config.recipe_length))
        result = list(center)
        # Upstream perturbs radius positions; each may retain its current action.
        for i in self.rng.sample(range(len(result)), min(len(result), self.region.radius)):
            result[i] = self.rng.randrange(len(TRANSFORMS))
        return tuple(result)

    def neighbor(self, recipe):
        i = self.rng.randrange(len(recipe))
        action = self.rng.choice([a for a in range(len(TRANSFORMS)) if a != recipe[i]])
        return recipe[:i] + (action,) + recipe[i+1:]

    def fresh_global(self, seen):
        for _ in range(self.n_candidates * 4):
            recipe = self.random_recipe()
            if recipe not in seen:
                return recipe
        return next((r for r in product(range(len(TRANSFORMS)), repeat=self.config.recipe_length)
                     if r not in seen), None)

    def fit(self, observations, *, restart=False):
        # All data collected since the last restart; no arbitrary training cap.
        gp = ExactSSKGP(self.config.recipe_length, self.config.ssk_order,
                        kernel=TransformedOverlapKernel() if restart else None)
        gp.fit([p[0] for p in observations], [p[1] for p in observations],
               self.config.restart_gp_training_steps if restart else self.config.gp_training_steps,
               rank_targets=not restart, standardise=self.config.standardise and not restart)
        self.surrogate_fits += 1
        return gp

    def restart(self, seen):
        if self.local:
            self.restart_bests.append(min(self.local, key=lambda p: p[1]))
        self.restarts += 1
        self._reset()
        self.initial_center = None
        pool = latin_hypercube_recipes(self.n_candidates, self.config.recipe_length, self.lhs_rng)
        pool = list(dict.fromkeys(r for r in pool if r not in seen))
        if pool and self.restart_bests:
            gp = self.fit(self.restart_bests, restart=True)
            incumbent = min(self.restart_bests, key=lambda p: p[1])[0]
            scores = gp.acquisition(pool, incumbent, kind="lcb")
            self.initial_center = pool[int(scores.argmax())]
            self.pending = [self.initial_center] + [self.random_recipe(self.initial_center)
                                                    for _ in range(self.config.n_initial-1)]

    def suggest(self, seen):
        if len(seen) >= len(TRANSFORMS) ** self.config.recipe_length:
            return None
        while self.pending:
            recipe = self.pending.pop(0)
            if recipe not in seen:
                return recipe
        if len(self.local) < self.config.n_initial:
            if self.initial_center is not None:
                for _ in range(self.n_candidates):
                    recipe = self.random_recipe(self.initial_center)
                    if recipe not in seen:
                        return recipe
            return self.fresh_global(seen)
        center = min(self.local, key=lambda p: p[1])[0]
        gp = self.fit(self.local)
        # Match upstream: three walks from the incumbent, one random valid
        # neighbor per step, accepting only strictly larger augmented EI.
        best_recipe, best_score = None, -float("inf")
        for _ in range(3):
            current = center
            current_score = float(gp.acquisition([current], center)[0])
            for _ in range(self.config.acquisition_steps):
                candidate = None
                for _ in range(100):
                    proposal = self.neighbor(current)
                    if 0 < hamming(proposal, center) <= self.region.radius:
                        candidate = proposal
                        break
                if candidate is None:
                    break
                score = float(gp.acquisition([candidate], center)[0])
                # Keep the best unseen visited point if an endpoint is cached.
                if candidate not in seen and score > best_score:
                    best_recipe, best_score = candidate, score
                if score > current_score:
                    current, current_score = candidate, score
        if best_recipe is not None:
            return best_recipe
        # Degenerate/cached walks: acquisition over unseen one-edit neighbours.
        neighbors = []
        for i in range(len(center)):
            for action in range(len(TRANSFORMS)):
                recipe = center[:i] + (action,) + center[i+1:]
                if recipe not in seen:
                    neighbors.append(recipe)
        if neighbors:
            scores = gp.acquisition(neighbors, center)
            return neighbors[int(scores.argmax())]
        self.restart(seen)
        return self.suggest(seen)

    def observe(self, recipe, value, seen):
        collapse = False
        if len(self.local) >= self.config.n_initial:
            collapse = self.region.update(value, min(v for _, v in self.local))
        observation = (tuple(recipe), value)
        self.local.append(observation)
        self.history.append(observation)
        # Upstream tests the threshold after every observation, even during init.
        if collapse or self.region.radius <= 1:
            self.restart(seen)
