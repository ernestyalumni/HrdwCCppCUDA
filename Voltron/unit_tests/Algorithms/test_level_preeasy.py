"""
https://blog.faangshui.com/p/before-leetcode
"""

from Voltron.Algorithms.level_preeasy import (
    # 7. Generate All Subsets of a Set
    generate_all_subsets_iterative,
    generate_all_subsets_recursive
    )

import pytest

@pytest.fixture
def power_set_test_cases():
    example_1 = {
        "input": [1, 2, 3],
        "output": [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    }
    example_2 = {
        "input": [0],
        "output": [[], [0]]
    }
    example_3 = {
        "input": [],
        "output": [[]]
    }
    return [example_1, example_2, example_3]

def test_generate_all_subsets_iterative(power_set_test_cases):
    test_cases = power_set_test_cases
    for test_case in test_cases:
        assert sorted(generate_all_subsets_iterative(test_case["input"])) == \
            sorted(test_case["output"])

def test_generate_all_subsets_recursive(power_set_test_cases):
    test_cases = power_set_test_cases
    for test_case in test_cases:
        assert sorted(generate_all_subsets_recursive(test_case["input"])) == \
            sorted(test_case["output"])