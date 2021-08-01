"""
@file test_level_easy.py

@details

Example Usage:

pytest Utilities/test_level_easy.py 
"""
from Voltron.Algorithms.level_easy import (
    # In order of appearance or usage.
    get_nth_fibonacci_recursive,
    fibonacci_no_branch,
    get_nth_fibonacci_recursive_no_branch,
    non_constructible_change_all_permutations,
    non_constructible_change_observe_sum)


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


def test_non_constructible_change_all_permutations():

    coins = [5, 7, 1, 1, 2, 3, 22]

    assert non_constructible_change_all_permutations(coins) == 20

    # Test case 12
    coins = [109, 2000, 8765, 19, 18, 17, 16, 8, 1, 1, 2, 4]
    assert non_constructible_change_observe_sum(coins) == 87

    # Test case 13
    coins = [1, 2, 3, 4, 5, 6, 7]

    assert non_constructible_change_all_permutations(coins) == 29


def test_non_constructible_change_observe_sum():

    coins = [5, 7, 1, 1, 2, 3, 22]

    assert non_constructible_change_observe_sum(coins) == 20


    # Test case 12
    coins = [109, 2000, 8765, 19, 18, 17, 16, 8, 1, 1, 2, 4]
    assert non_constructible_change_observe_sum(coins) == 87

    # Test case 13
    coins = [1, 2, 3, 4, 5, 6, 7]

    assert non_constructible_change_observe_sum(coins) == 29

