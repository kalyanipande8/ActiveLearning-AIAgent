"""
Experiment-selection agents.

BayesianOptimizationAgent: a real Gaussian-Process-surrogate active-learning
agent. It fits a GaussianProcessRegressor (scikit-learn) on all experiments
observed so far, then scores every untested candidate with an Expected
Improvement (EI) acquisition function that trades off predicted mean against
predictive uncertainty, and selects the top-scoring candidate to test next.

GreedyAgent: a baseline that fits the same kind of surrogate but always
selects the candidate with the highest *predicted mean* only -- it ignores
uncertainty entirely, so it never explores beyond what already looks good
and will get stuck exploiting decoy peaks.

No live LLM calls are made anywhere in this repository (see README).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def _make_gp(seed: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=0.2, length_scale_bounds=(1e-2, 1.0)
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=seed,
    )


class _BaseAgent:
    name = "base"

    def __init__(self, space, n_init: int = 5, seed: int = 0):
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.tested_idx: list[int] = []
        self.observed_y: list[float] = []
        self.gp = _make_gp(seed)
        self.n_init = n_init

    def _initial_candidates(self):
        return self.rng.choice(self.space.n_candidates, size=self.n_init, replace=False)

    def _fit(self):
        X = self.space.X[self.tested_idx]
        y = np.array(self.observed_y)
        self.gp.fit(X, y)

    def _predict_all(self):
        mu, sigma = self.gp.predict(self.space.X, return_std=True)
        return mu, sigma

    def _select_next(self, mu, sigma, best_so_far):
        raise NotImplementedError

    def run(self, max_experiments: int = 400, discovery_fraction: float = 0.95):
        """Sequentially select and run experiments until discovery or budget runs out."""
        untested = set(range(self.space.n_candidates))

        for idx in self._initial_candidates():
            idx = int(idx)
            self._observe(idx, untested)
            if self.space.is_discovery(idx, discovery_fraction):
                return len(self.tested_idx)

        while len(self.tested_idx) < max_experiments and untested:
            self._fit()
            mu, sigma = self._predict_all()
            best_so_far = max(self.observed_y)

            remaining = np.array(sorted(untested))
            scores = self._select_next(mu[remaining], sigma[remaining], best_so_far)
            next_idx = int(remaining[np.argmax(scores)])

            self._observe(next_idx, untested)
            if self.space.is_discovery(next_idx, discovery_fraction):
                return len(self.tested_idx)

        return None  # did not discover within budget

    def _observe(self, idx: int, untested: set):
        y = self.space.measure(idx, self.rng)
        self.tested_idx.append(idx)
        self.observed_y.append(y)
        untested.discard(idx)


class BayesianOptimizationAgent(_BaseAgent):
    """GP-surrogate agent using Expected Improvement acquisition."""

    name = "bayesian_optimization"

    def __init__(self, space, n_init: int = 5, seed: int = 0, xi: float = 0.01):
        super().__init__(space, n_init=n_init, seed=seed)
        self.xi = xi

    def _select_next(self, mu, sigma, best_so_far):
        sigma = np.maximum(sigma, 1e-9)
        improvement = mu - best_so_far - self.xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-8] = 0.0
        return ei


class GreedyAgent(_BaseAgent):
    """Baseline: always exploit the current best-predicted mean, no exploration."""

    name = "greedy"

    def _select_next(self, mu, sigma, best_so_far):
        return mu  # pure exploitation, uncertainty ignored entirely
