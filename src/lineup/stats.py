from __future__ import annotations


def percentile_interval(values, alpha: float = 0.05, min_defined: int = 20):
    """A two-sided percentile interval over bootstrap samples, ignoring undefined draws.

    Returns (low, high), or (None, None) when too few resamples produced a defined value for
    the interval to mean anything.
    """
    defined = [value for value in values if value is not None]
    if len(defined) < min_defined:
        return (None, None)
    ordered = sorted(defined)
    low = ordered[int((alpha / 2) * len(ordered))]
    high = ordered[min(len(ordered) - 1, int((1 - alpha / 2) * len(ordered)))]
    return (low, high)
