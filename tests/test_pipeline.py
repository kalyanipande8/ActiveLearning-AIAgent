import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import BayesianOptimizationAgent, GreedyAgent
from search_space import SearchSpace


def test_search_space_has_1200plus_candidates():
    space = SearchSpace(n_candidates=1250, seed=0)
    assert space.n_candidates >= 1200
    assert space.X.shape[0] >= 1200
    assert space.true_values.shape[0] >= 1200


def test_discovery_threshold_identifies_true_best():
    space = SearchSpace(n_candidates=1250, seed=1)
    assert space.is_discovery(space.best_idx, fraction=0.95)


def test_bayesian_agent_runs_and_finds_discovery():
    space = SearchSpace(n_candidates=1250, seed=2)
    agent = BayesianOptimizationAgent(space, n_init=5, seed=2)
    n = agent.run(max_experiments=400, discovery_fraction=0.95)
    assert n is not None
    assert 0 < n <= 400


def test_greedy_agent_runs():
    space = SearchSpace(n_candidates=1250, seed=3)
    agent = GreedyAgent(space, n_init=5, seed=3)
    n = agent.run(max_experiments=400, discovery_fraction=0.95)
    # greedy may or may not find it within budget; just must not crash
    assert n is None or 0 < n <= 400


def test_bayesian_outperforms_greedy_on_average_across_seeds():
    """Sanity check: across several seeds, the Bayesian GP+EI agent should need
    fewer experiments on average to find the injected discovery than the
    no-exploration greedy baseline, and the reduction should be in a
    reasonable (non-trivial, non-absurd) band.
    """
    seeds = range(6)
    bo_counts, greedy_counts = [], []

    for seed in seeds:
        space = SearchSpace(n_candidates=1250, seed=seed)

        bo_agent = BayesianOptimizationAgent(space, n_init=5, seed=seed)
        bo_n = bo_agent.run(max_experiments=400, discovery_fraction=0.95)

        greedy_agent = GreedyAgent(space, n_init=5, seed=seed)
        greedy_n = greedy_agent.run(max_experiments=400, discovery_fraction=0.95)

        bo_counts.append(bo_n if bo_n is not None else 400)
        greedy_counts.append(greedy_n if greedy_n is not None else 400)

    bo_mean = np.mean(bo_counts)
    greedy_mean = np.mean(greedy_counts)

    assert bo_mean < greedy_mean, (
        f"Bayesian agent ({bo_mean}) should beat greedy baseline ({greedy_mean}) on average"
    )

    reduction_pct = 100.0 * (greedy_mean - bo_mean) / greedy_mean
    assert 5.0 <= reduction_pct <= 90.0, f"reduction % out of reasonable band: {reduction_pct}"
