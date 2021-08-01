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
    get_nth_fibonacci_memoization,
    non_constructible_change_all_permutations,
    non_constructible_change_observe_sum,
    minimum_waiting_time,
    minimum_waiting_time_optimal,
    class_photos,
    tandem_bicycle,
    product_sum,
    binary_search,
    find_three_largest_numbers
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


def test_get_nth_fibonacci_memoization():

    assert get_nth_fibonacci_memoization(2) == 1

    assert get_nth_fibonacci_memoization(6) == 5

    assert get_nth_fibonacci_memoization(17) == 987

    assert get_nth_fibonacci_memoization(18) == 1597


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


def test_tandem_bicycle_test_cases():
    
    # Test Case 1
    red_shirt_speeds = [5, 5, 3, 9, 2]
    blue_shirt_speeds = [3, 6, 7, 2, 1]

    result = tandem_bicycle(red_shirt_speeds, blue_shirt_speeds, True)
    assert result == 32


    # Test Case 2
    red_shirt_speeds = [5, 5, 3, 9, 2]
    blue_shirt_speeds = [3, 6, 7, 2, 1]

    result = tandem_bicycle(red_shirt_speeds, blue_shirt_speeds, False)
    assert result == 25


    # Test Case 3
    red_shirt_speeds = [1, 2, 1, 9, 12, 3]
    blue_shirt_speeds = [3, 3, 4, 6, 1, 2]

    result = tandem_bicycle(red_shirt_speeds, blue_shirt_speeds, False)
    assert result == 30

    # Test Case 4
    red_shirt_speeds = [1, 2, 1, 9, 12, 3]
    blue_shirt_speeds = [3, 3, 4, 6, 1, 2]

    result = tandem_bicycle(red_shirt_speeds, blue_shirt_speeds, True)
    assert result == 37


def test_product_sum_sample_input():

    array = [5, 2, [7, -1], 3, [6, [-13, 8], 4]]

    result = product_sum(array)

    assert result == 12

def test_product_sum_test_cases():

    # Test Case 2.

    array = [1, 2, 3, 4, 5]

    result = product_sum(array)

    assert result == 15

    # Test Case 3.

    array = [1, 2, [3], 4, 5]

    result = product_sum(array)

    assert result == 18

    # Test Case 4.
    array = [
        [1, 2],
        3,
        [4, 5]]

    result = product_sum(array)


def test_binary_search_cases():

    # Test case 1
    array = [0, 1, 21, 33, 45, 45, 61, 71, 72, 73]

    result = binary_search(array, 33)

    assert result == 3

    # Test case 2

    array = [1, 5, 23, 111]

    result = binary_search(array, 111)

    assert result == 3

    # Test case 3

    result = binary_search(array, 5)

    assert result == 1

    # Test case 4

    result = binary_search(array, 35)

    assert result == -1


def test_find_three_largest_numbers():

    # Sample Input and Test Case 1

    array = [141, 1, 17, -7, -17, -27, 18, 541, 8, 7, 7]

    result = find_three_largest_numbers(array)

    assert result == [18, 141, 541]

def test_find_three_largest_numbers_test_cases():

    # Test Case 2

    array = [55, 7, 8]

    result = find_three_largest_numbers(array)

    assert result == [7, 8, 55]

    # Test Case 3

    array = [55, 43, 11, 3, -3, 10]

    result = find_three_largest_numbers(array)

    assert result == [11, 43, 55]
