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
    non_constructible_change_observe_sum,
    minimum_waiting_time,
    minimum_waiting_time_optimal,
    class_photos,
    tandem_bicycle
    )


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


def test_minimum_waiting_time_sample_case():
    queries = [3, 2, 1, 2, 6]

    assert minimum_waiting_time(queries) == 17

    # Test case 12
    queries = [5, 4, 3, 2, 1]

    assert minimum_waiting_time(queries) == 20

    # Test case 12
    queries = [1, 2, 3, 4, 5]

    assert minimum_waiting_time(queries) == 20


def test_minimum_waiting_time_optimal_sample_case():
    queries = [3, 2, 1, 2, 6]

    assert minimum_waiting_time_optimal(queries) == 17

    # Test case 12
    queries = [5, 4, 3, 2, 1]

    assert minimum_waiting_time_optimal(queries) == 20

    # Test case 12
    queries = [1, 2, 3, 4, 5]

    assert minimum_waiting_time_optimal(queries) == 20


def test_class_photos_sample_case():
    test_red_shirt_heights = [5, 8, 1, 3, 4]
    test_blue_shirt_heights = [6, 9, 2, 4, 5]

    result = class_photos(test_red_shirt_heights, test_blue_shirt_heights)

def test_class_photos_test_cases():
    
    # Test case 2
    test_red_shirt_heights = [6, 9, 2, 4, 5]
    test_blue_shirt_heights = [5, 8, 1, 3, 4]

    result = class_photos(test_red_shirt_heights, test_blue_shirt_heights)

    assert result

    # Test case 3
    test_red_shirt_heights = [6, 9, 2, 4, 5, 1]
    test_blue_shirt_heights = [5, 8, 1, 3, 4, 9]

    result = class_photos(test_red_shirt_heights, test_blue_shirt_heights)

    assert not result

    # Test case 4
    test_red_shirt_heights = [6,]
    test_blue_shirt_heights = [6,]

    result = class_photos(test_red_shirt_heights, test_blue_shirt_heights)

    assert not result

    # Test case 5
    test_red_shirt_heights = [126,]
    test_blue_shirt_heights = [125,]

    result = class_photos(test_red_shirt_heights, test_blue_shirt_heights)

    assert result
