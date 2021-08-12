from Voltron.Algorithms.DynamicProgramming.memoization import (
    # In order of appearance or usage.
    fibonacci,
    grid_traveler
    )

import pytest

def test_fibonacci():
    assert fibonacci(6) == 8
    assert fibonacci(7) == 13
    assert fibonacci(8) == 21
    assert fibonacci(50) == 12586269025

def test_grid_travler():
    assert grid_traveler(1, 1) == 1
    assert grid_traveler(2, 1) == 1
    assert grid_traveler(1, 2) == 1
    assert grid_traveler(3, 1) == 1
    assert grid_traveler(1, 3) == 1
    assert grid_traveler(2, 2) == 2
    assert grid_traveler(2, 3) == 3
    assert grid_traveler(3, 2) == 3
    assert grid_traveler(3, 3) == 6
    assert grid_traveler(7, 3) == 28
    assert grid_traveler(18, 18) == 2333606220