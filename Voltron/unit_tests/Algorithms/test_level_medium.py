"""
@file test_level_medium.py
"""

from Voltron.Algorithms.level_medium import (
    smallest_difference)

import pytest


def test_sample_case():
    array_one = [-1, 5, 10, 20, 28, 3]
    array_two = [26, 134, 135, 15, 17]

    result = smallest_difference(array_one, array_two)    

    assert result[0] == 28
    assert result[1] == 26