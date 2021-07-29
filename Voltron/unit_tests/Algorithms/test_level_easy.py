"""
@file test_level_easy.py

@details

Example Usage:

pytest Utilities/test_level_easy.py 
"""
from Voltron.Algorithms.level_easy import *

def test_get_nth_fibonacci_recursive():
    assert get_nth_fibonacci_recursive(2) == 1

    assert get_nth_fibonacci_recursive(6) == 5

    assert get_nth_fibonacci_recursive(17) == 987

    assert get_nth_fibonacci_recursive(18) == 1597


def test_get_nth_fibonacci_recursive_no_branch():
    assert get_nth_fibonacci_recursive_no_branch(2) == 1

    assert get_nth_fibonacci_recursive_no_branch(6) == 5

    assert get_nth_fibonacci_recursive_no_branch(17) == 987

    assert get_nth_fibonacci_recursive_no_branch(18) == 1597
