"""
EpisodeStarts: which observations begin an episode in a NEXT_STEP vector env.
"""

import numpy as np


class EpisodeStarts:
    """
    Marks observations that begin an episode under NEXT_STEP autoreset, where a
    done flag arrives with the terminal observation and the following observation
    starts a new episode.

    Feed it every reset() and step() result of the vector env, in order. Each call
    returns a fresh (num_envs,) bool mask for the observation just received.
    """

    def __init__(self, num_envs: int):
        self._pending = np.zeros(num_envs, dtype=bool)

    def reset(self) -> np.ndarray:
        """Mask for a reset() observation: every env begins an episode."""
        self._pending = np.zeros_like(self._pending)
        return np.ones_like(self._pending)

    def step(self, terminations, truncations) -> np.ndarray:
        """Mask for a step() observation: the envs whose previous observation was done."""
        starts = self._pending
        self._pending = np.logical_or(terminations, truncations)
        return starts
