"""
Synthetic experimental search space for the active-learning agent.

Simulates a 1,200+ candidate chemical/condition screening library. Each
candidate is a point in a 2-D continuous parameter space (e.g. two
normalized process/formulation variables such as "reagent ratio" and
"reaction temperature"). A hidden Gaussian "activity" surface defines the
true (noiseless) response of each candidate -- the ground truth a wet-lab
assay would eventually reveal. One region is a true high-activity "hit"
that the agent must discover. Observed measurements include realistic
Gaussian measurement noise, as in a real assay.
"""

from __future__ import annotations

import numpy as np


class SearchSpace:
    """A fixed library of candidate experiments with a hidden discovery target."""

    def __init__(
        self,
        n_candidates: int = 1250,
        n_dims: int = 2,
        noise_std: float = 0.05,
        hit_center: np.ndarray | None = None,
        hit_width: float = 0.08,
        n_decoy_peaks: int = 4,
        seed: int = 0,
    ):
        if n_candidates < 1200:
            raise ValueError("search space must contain at least 1,200 candidates")

        self.n_candidates = n_candidates
        self.n_dims = n_dims
        self.noise_std = noise_std
        rng = np.random.default_rng(seed)

        # Candidate library: normalized coordinates in [0, 1]^n_dims
        self.X = rng.uniform(0.0, 1.0, size=(n_candidates, n_dims))

        # The true hidden "hit" region (the discovery target)
        self.hit_center = (
            hit_center if hit_center is not None else rng.uniform(0.3, 0.7, size=n_dims)
        )
        self.hit_width = hit_width

        # A few smaller decoy local-optima peaks so the landscape is non-trivial
        # and a naive greedy search can get stuck exploiting a decoy.
        self.decoy_centers = rng.uniform(0.0, 1.0, size=(n_decoy_peaks, n_dims))
        self.decoy_heights = rng.uniform(0.45, 0.75, size=n_decoy_peaks)
        self.decoy_widths = rng.uniform(0.05, 0.12, size=n_decoy_peaks)

        self.true_values = self._true_response(self.X)
        self.best_idx = int(np.argmax(self.true_values))
        self.best_value = float(self.true_values[self.best_idx])

        self._rng = rng

    def _true_response(self, X: np.ndarray) -> np.ndarray:
        """Noiseless ground-truth 'activity' of candidates (never seen by agents)."""
        d2_hit = np.sum((X - self.hit_center) ** 2, axis=1)
        response = 1.0 * np.exp(-d2_hit / (2 * self.hit_width ** 2))

        for center, height, width in zip(
            self.decoy_centers, self.decoy_heights, self.decoy_widths
        ):
            d2 = np.sum((X - center) ** 2, axis=1)
            response += height * np.exp(-d2 / (2 * width ** 2))

        return response

    def measure(self, idx: int, rng: np.random.Generator) -> float:
        """Run a (simulated) noisy experiment on candidate `idx`."""
        true_val = self.true_values[idx]
        return float(true_val + rng.normal(0.0, self.noise_std))

    def discovery_threshold(self, fraction: float = 0.95) -> float:
        """True-value threshold that counts as 'finding the hit', within tolerance."""
        return fraction * self.best_value

    def is_discovery(self, idx: int, fraction: float = 0.95) -> bool:
        return self.true_values[idx] >= self.discovery_threshold(fraction)
