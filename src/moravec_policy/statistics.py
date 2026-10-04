"""Descriptive uncertainty for independent binary trials; not arbitrary process scores."""

import math


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid trial counts")
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)
