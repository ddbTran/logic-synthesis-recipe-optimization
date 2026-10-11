# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Adapted from HEBO/BOiLS; BSD-3-Clause terms and disclaimer in LICENSE.
"""CPU exact GP with normalized fixed-order string-subsequence features.

The explicit feature map equals seq_kernel_fast.py's DP with a strictly
upper-triangular decay matrix. It avoids materializing all pairs of strings.
"""

from itertools import combinations
import math

import torch

DTYPE = torch.float64


class StringSubsequenceKernel(torch.nn.Module):
    def __init__(self, sequence_length, alphabet_size=16, order=3, gap_decay=0.5,
                 match_decay=0.8):
        super().__init__()
        if not 1 <= order <= min(3, sequence_length):
            raise ValueError("SSK order must be in [1, min(3, sequence_length)]")
        if not 0 < gap_decay < 1 or not 0 < match_decay <= 1:
            raise ValueError("invalid SSK decay")
        self.sequence_length, self.alphabet_size, self.order = sequence_length, alphabet_size, order
        self.match_decay = match_decay
        self.raw_gap = torch.nn.Parameter(torch.tensor(math.log(gap_decay / (1-gap_decay)), dtype=DTYPE))
        indices = torch.tensor(list(combinations(range(sequence_length), order)), dtype=torch.long)
        self.register_buffer("indices", indices)
        self.register_buffer("gaps", (indices[:, -1] - indices[:, 0] + 1 - order).to(DTYPE))
        self.register_buffer("powers", alphabet_size ** torch.arange(order-1, -1, -1))

    def features(self, x):
        x = torch.as_tensor(x, dtype=torch.long, device="cpu")
        if x.ndim != 2 or x.shape[1] != self.sequence_length or (x < 0).any() or (x >= self.alphabet_size).any():
            raise ValueError("SSK input must be a matrix of fixed-length action indices")
        codes = (x[:, self.indices] * self.powers).sum(-1)
        weights = self.raw_gap.sigmoid().pow(self.gaps) * self.match_decay ** self.order
        features = torch.zeros((len(x), self.alphabet_size ** self.order), dtype=DTYPE)
        features = features.scatter_add(1, codes, weights.expand(len(x), -1))
        return features / features.norm(dim=1, keepdim=True).clamp_min(1e-15)

    def forward(self, x, y=None):
        left = self.features(x)
        right = left if y is None else self.features(y)
        return left @ right.T


def rank_standardise(y):
    """Upstream's maximum rank for ties, then Gaussian quantiles."""
    y = torch.as_tensor(y, dtype=DTYPE)
    ranks = (y[:, None] >= y[None, :]).sum(1).to(DTYPE)
    return math.sqrt(2) * torch.erfinv(2 * ranks / (len(y) + 1) - 1)


class ExactSSKGP(torch.nn.Module):
    def __init__(self, sequence_length, order=3):
        super().__init__()
        self.kernel = StringSubsequenceKernel(sequence_length, order=min(order, sequence_length))
        self.mean = torch.nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.raw_scale = torch.nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.raw_noise = torch.nn.Parameter(torch.tensor(-3., dtype=DTYPE))

    @property
    def scale(self):
        return 0.5 + 4.5 * self.raw_scale.sigmoid()

    @property
    def noise(self):
        return 1e-6 + (0.1 - 1e-6) * self.raw_noise.sigmoid()

    def fit(self, x, y, steps=50):
        self.x = torch.as_tensor(x, dtype=torch.long)
        self.y = rank_standardise(y)
        identity = torch.eye(len(self.x), dtype=DTYPE)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.03)
        for _ in range(steps):
            optimizer.zero_grad()
            covariance = self.scale * self.kernel(self.x) + (self.noise + 1e-8) * identity
            chol = torch.linalg.cholesky(covariance)
            residual = self.y - self.mean
            alpha = torch.cholesky_solve(residual[:, None], chol).squeeze(1)
            loss = 0.5 * residual.dot(alpha) + chol.diag().log().sum()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite SSK GP marginal likelihood")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            self.train_features = self.kernel.features(self.x)
            covariance = self.scale * (self.train_features @ self.train_features.T) + (self.noise + 1e-8) * identity
            self.chol = torch.linalg.cholesky(covariance)
            self.alpha = torch.cholesky_solve((self.y-self.mean)[:, None], self.chol).squeeze(1)
        return self

    @torch.no_grad()
    def predict(self, x):
        cross = self.scale * (self.kernel.features(x) @ self.train_features.T)
        mean = self.mean + cross @ self.alpha
        projected = torch.linalg.solve_triangular(self.chol, cross.T, upper=False)
        variance = (self.scale - projected.square().sum(0)).clamp_min(1e-12)
        return mean, variance

    @torch.no_grad()
    def acquisition(self, x, incumbent, kind="ei"):
        mean, variance = self.predict(x)
        if kind == "lcb":
            return -mean + 1.96 * (variance + self.noise).sqrt()
        target = self.predict([incumbent])[0][0]
        std = variance.sqrt()
        improvement = target - mean
        z = improvement / std
        ei = improvement * (0.5 * (1 + torch.erf(z / math.sqrt(2)))) + std * torch.exp(-0.5*z.square()) / math.sqrt(2*math.pi)
        return ei * (1 - self.noise.sqrt() / (self.noise + variance).sqrt())
