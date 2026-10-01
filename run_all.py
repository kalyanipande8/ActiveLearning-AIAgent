"""
Run the full Bayesian-optimization vs. greedy-baseline experiment-selection
benchmark over multiple random seeds, and report results.

No live LLM API calls are made. Everything here is a self-contained,
deterministic (seeded) numerical simulation: sklearn's GaussianProcessRegressor
as the surrogate model, scipy for the Expected Improvement acquisition
function, and numpy for the synthetic candidate library.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import BayesianOptimizationAgent, GreedyAgent
from search_space import SearchSpace

N_CANDIDATES = 1250
N_TRIALS = 20
MAX_EXPERIMENTS = 400
DISCOVERY_FRACTION = 0.95


def run_trial(seed: int):
    space = SearchSpace(n_candidates=N_CANDIDATES, seed=seed)

    bo_agent = BayesianOptimizationAgent(space, n_init=5, seed=seed)
    bo_result = bo_agent.run(max_experiments=MAX_EXPERIMENTS, discovery_fraction=DISCOVERY_FRACTION)

    greedy_agent = GreedyAgent(space, n_init=5, seed=seed)
    greedy_result = greedy_agent.run(
        max_experiments=MAX_EXPERIMENTS, discovery_fraction=DISCOVERY_FRACTION
    )

    return bo_result, greedy_result


def main():
    t0 = time.time()
    bo_counts, greedy_counts = [], []
    bo_failures, greedy_failures = 0, 0

    print(f"Running {N_TRIALS} trials, {N_CANDIDATES} candidates each, budget={MAX_EXPERIMENTS}...")
    for seed in range(N_TRIALS):
        bo_n, greedy_n = run_trial(seed)

        if bo_n is None:
            bo_failures += 1
            bo_n = MAX_EXPERIMENTS
        if greedy_n is None:
            greedy_failures += 1
            greedy_n = MAX_EXPERIMENTS

        bo_counts.append(bo_n)
        greedy_counts.append(greedy_n)
        print(f"  seed={seed:2d}  bayesian={bo_n:4d}  greedy={greedy_n:4d}")

    bo_counts = np.array(bo_counts)
    greedy_counts = np.array(greedy_counts)

    bo_mean = float(bo_counts.mean())
    greedy_mean = float(greedy_counts.mean())
    reduction_pct = 100.0 * (greedy_mean - bo_mean) / greedy_mean

    results = {
        "n_candidates": N_CANDIDATES,
        "n_trials": N_TRIALS,
        "max_experiments_budget": MAX_EXPERIMENTS,
        "discovery_fraction": DISCOVERY_FRACTION,
        "bayesian_experiments_to_discovery_mean": bo_mean,
        "bayesian_experiments_to_discovery_std": float(bo_counts.std()),
        "greedy_experiments_to_discovery_mean": greedy_mean,
        "greedy_experiments_to_discovery_std": float(greedy_counts.std()),
        "pct_reduction_bayesian_vs_greedy": reduction_pct,
        "bayesian_failures_to_discover": bo_failures,
        "greedy_failures_to_discover": greedy_failures,
        "bayesian_counts": bo_counts.tolist(),
        "greedy_counts": greedy_counts.tolist(),
        "runtime_seconds": time.time() - t0,
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Bayesian-optimization agent: {bo_mean:.1f} +/- {bo_counts.std():.1f} experiments to discovery")
    print(f"Greedy baseline agent:       {greedy_mean:.1f} +/- {greedy_counts.std():.1f} experiments to discovery")
    print(f"Reduction (Bayesian vs greedy): {reduction_pct:.1f}%")
    print(f"Bayesian non-discoveries within budget: {bo_failures}/{N_TRIALS}")
    print(f"Greedy non-discoveries within budget:   {greedy_failures}/{N_TRIALS}")
    print(f"Total runtime: {results['runtime_seconds']:.1f}s")

    make_figure(bo_counts, greedy_counts, reduction_pct)
    print("\nWrote results.json and figures/experiments_to_discovery.png")


def make_figure(bo_counts, greedy_counts, reduction_pct):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    trials = np.arange(1, len(bo_counts) + 1)
    ax.plot(trials, bo_counts, "o-", color="#2E86AB", label="Bayesian optimization (GP + EI)")
    ax.plot(trials, greedy_counts, "s-", color="#D1495B", label="Greedy baseline")
    ax.set_xlabel("Trial (random seed)")
    ax.set_ylabel("Experiments to discovery")
    ax.set_title("Experiments-to-discovery per trial")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    means = [bo_counts.mean(), greedy_counts.mean()]
    stds = [bo_counts.std(), greedy_counts.std()]
    bars = ax.bar(
        ["Bayesian\noptimization", "Greedy\nbaseline"],
        means,
        yerr=stds,
        color=["#2E86AB", "#D1495B"],
        capsize=6,
    )
    ax.set_ylabel("Mean experiments to discovery")
    ax.set_title(f"Mean comparison ({reduction_pct:.1f}% reduction)")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 2, f"{m:.1f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("figures/experiments_to_discovery.png", dpi=150)


if __name__ == "__main__":
    main()
