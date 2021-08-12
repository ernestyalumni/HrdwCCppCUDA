from Voltron.Algorithms.LeetCode.UniquePaths import (
    # In order of appearance or usage.
    unique_paths
    )

import pytest


def test_unique_paths():
    assert unique_paths(3, 7) == 28
    assert unique_paths(3, 2) == 3
    assert unique_paths(7, 3) == 28
    assert unique_paths(3, 3) == 6