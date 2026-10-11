"""Numerical core adapted from ast-gwo-aicas2026; see PROVENANCE.md."""
from typing import Optional
import numpy as np


class Leaders:
    """Tracks the three best individuals (alpha, beta, delta)."""

    def __init__(self) -> None:
        self.bits = [None, None, None]   # np.ndarray | None
        self.fit  = [np.inf, np.inf, np.inf]
        self.impr = [0.0, 0.0, 0.0]

    @property
    def alpha(self) -> Optional[np.ndarray]:
        return self.bits[0]

    @property
    def alpha_fit(self) -> float:
        return self.fit[0]

    @property
    def alpha_improve(self) -> float:
        return self.impr[0]

    def update(
        self,
        pop: np.ndarray,
        pop_fit: np.ndarray,
        pop_improve: np.ndarray,
    ) -> None:
        top_idx = np.argsort(pop_fit)[:3]

        if self.bits[0] is None:
            for rank, i in enumerate(top_idx):
                self.bits[rank] = pop[i].copy()
                self.fit[rank]  = pop_fit[i]
                self.impr[rank] = pop_improve[i]
            return

        pool_bits = np.vstack([np.array(self.bits), pop[top_idx]])
        pool_fit  = np.array(self.fit  + [pop_fit[i]     for i in top_idx])
        pool_impr = np.array(self.impr + [pop_improve[i] for i in top_idx])

        order = np.argsort(pool_fit)
        for rank in range(3):
            o = order[rank]
            self.bits[rank] = pool_bits[o].copy()
            self.fit[rank]  = pool_fit[o]
            self.impr[rank] = pool_impr[o]


class Optimizer:
    def __init__(self, config):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)
        self.pop = self._rng.integers(
            0, 2, size=(config.population_size, 4 * config.recipe_length), dtype=np.uint8)
        self._leaders = Leaders()
        self.a = 2.0
        self.k = config.k_min

    def _update_coefficients(self, t: int) -> None:
        """Linearly decay a; advance adaptive sigmoid slope k."""
        self.a = 2.0 - 2.0 * t / self.cfg.generations
        self.k = self.cfg.k_min + (t / self.cfg.generations) * (self.cfg.k_max - self.cfg.k_min)

    # =====================================================
    # Position update — preserves upstream AstGWO._step()
    # =====================================================
    def _step(self) -> None:
        """Move the whole population one step toward the leaders."""

        x_alpha = self._leaders.bits[0]
        x_beta  = self._leaders.bits[1]
        x_delta = self._leaders.bits[2]

        r1_alpha = self._rng.random(self.pop.shape)
        r1_beta  = self._rng.random(self.pop.shape)
        r1_delta = self._rng.random(self.pop.shape)
        r2_alpha = self._rng.random(self.pop.shape)
        r2_beta  = self._rng.random(self.pop.shape)
        r2_delta = self._rng.random(self.pop.shape)

        A_alpha = 2 * self.a * r1_alpha - self.a
        A_beta  = 2 * self.a * r1_beta  - self.a
        A_delta = 2 * self.a * r1_delta - self.a
        C_alpha = 2 * r2_alpha
        C_beta  = 2 * r2_beta
        C_delta = 2 * r2_delta

        # Distance vectors  Eq. (3.1.4) - (3.1.6)
        D_alpha = np.abs(C_alpha * x_alpha - self.pop)
        D_beta  = np.abs(C_beta  * x_beta  - self.pop)
        D_delta = np.abs(C_delta * x_delta - self.pop)

        # Candidate positions  Eq. (3.1.7) - (3.1.9)
        X1 = x_alpha - A_alpha * D_alpha
        X2 = x_beta  - A_beta  * D_beta
        X3 = x_delta - A_delta * D_delta

        # Mean position  Eq. (3.1.10)
        X_mean = (X1 + X2 + X3) / 3.0

        # Adaptive sigmoid transfer  Eq. (3.4.1)
        S = 1.0 / (1.0 + np.exp(-self.k * (X_mean - 0.5)))

        # Binary update  Eq. (3.4.3)
        r = self._rng.random(self.pop.shape)
        self.pop = (r < S).astype(np.uint8)


def bits_to_recipe(bits):
    """Decode big-endian four-bit opcodes into shared action indices."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or not len(bits) or len(bits) % 4 or not np.isin(bits, [0, 1]).all():
        raise ValueError("Expected a nonempty binary vector with length divisible by four")
    return tuple(int(value) for value in bits.reshape(-1, 4) @ np.array([8, 4, 2, 1]))
