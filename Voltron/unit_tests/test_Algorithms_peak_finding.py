"""
@file test_Algorithms_peak_finding.py
"""
from Voltron.Algorithms import peak_finding
from Voltron.Algorithms.peak_finding import straightforward_search_1d

import pytest


@pytest.fixture
def example_1d_values():

    class Example1dValues:
        a = [6, 7]


    return Example1dValues()


def test_straightforward_search_1d_on_examples(example_1d_values):
    assert example_1d_values.a[0] == 6