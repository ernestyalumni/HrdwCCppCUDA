"""
@file test_level_medium.py
"""

from Voltron.Algorithms.level_medium import (
    # In order of usage or appearance.
    smallest_difference,
    three_number_sum,
    three_number_sum_with_sorted_array,
    move_element_to_end,
    is_monotonic,
    spiral_traverse,
    _find_peaks,
    _find_peak_length,
    longest_peak,
    array_of_products,
    brute_force_array_of_products,
    brute_force_first_duplicate_value,
    first_duplicate_value_with_ds,
    brute_merge_overlapping_intervals,
    sorted_merge_overlapping_intervals,
    remove_islands
    )

import pytest


def test_sample_case():
    array_one = [-1, 5, 10, 20, 28, 3]
    array_two = [26, 134, 135, 15, 17]

    result = smallest_difference(array_one, array_two)    

    assert result[0] == 28
    assert result[1] == 26


def test_three_number_sum():

    # Sample Input and Test Case 1
    array = [12, 3, 1, 2, -6, 5, -8, 6]
    target = 0

    results = three_number_sum(array, target)

    assert results == [[-8, 2, 6], [-8, 3, 5], [-6, 1, 5]]

    # Test Case 2

    array = [1, 2, 3]
    target = 6

    results = three_number_sum(array, target)

    assert results == [[1, 2, 3]]

    # Test Case 3

    target = 7

    results = three_number_sum(array, target)

    assert results == []

    # Test Case 4

    array = [8, 10, -2, 49, 14]
    target = 57

    results = three_number_sum(array, target)
    assert results == [[-2, 10, 49]]


def test_three_number_sum_with_sorted_array():

    # Sample Input and Test Case 1
    array = [12, 3, 1, 2, -6, 5, -8, 6]
    target = 0

    results = three_number_sum_with_sorted_array(array, target)

    assert results == [[-8, 2, 6], [-8, 3, 5], [-6, 1, 5]]

    # Test Case 2

    array = [1, 2, 3]
    target = 6

    results = three_number_sum_with_sorted_array(array, target)

    assert results == [[1, 2, 3]]

    # Test Case 3

    target = 7

    results = three_number_sum_with_sorted_array(array, target)

    assert results == []

    # Test Case 4

    array = [8, 10, -2, 49, 14]
    target = 57

    results = three_number_sum_with_sorted_array(array, target)
    assert results == [[-2, 10, 49]]


def test_smallest_difference():
    array_one = [10, 1000, 9124, 2142, 59, 24, 596, 591, 124, -123, 530]
    array_two = [-1441, -124, -25, 1014, 1500, 660, 410, 245, 530]

    result = smallest_difference(array_one, array_two)

    assert result == [530, 530]


def test_move_element_to_end():

    # Test Case 3
    array = [1, 2, 4, 5, 6]
    to_move = 3
    result = move_element_to_end(array, to_move)
    assert result == [1, 2, 4, 5, 6]

    # Test Case 6
    array = [1, 2, 4, 5, 3]
    to_move = 3
    result = move_element_to_end(array, to_move)
    assert result == [1, 2, 4, 5, 3]

    # Test Case 7
    array = [1, 2, 3, 4, 5]
    to_move = 3
    result = move_element_to_end(array, to_move)
    assert result == [1, 2, 4, 5, 3]

    # Test Case 10
    array = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 5, 5, 5, 5, 5, 5]
    to_move = 5
    result = move_element_to_end(array, to_move)
    assert result == [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 5, 5, 5, 5, 5, 5]


def test_is_monotonic():

    # Sample Input and Test Case 1
    array = [-1, -5, -10, -1100, -1100, -1101, -1102, -9001]
    assert is_monotonic(array)

    # Test Case 2
    array = []
    assert is_monotonic(array)   

    # Test Case 3
    array = [1]
    assert is_monotonic(array)

    # Test Case 4
    array = [1, 2]
    assert is_monotonic(array)

    # Test Case 5
    array = [2, 1]
    assert is_monotonic(array)

    # Test Case 12
    array = [-1, -1, -2, -3, -4, -5, -5, -5, -6, -7, -8, -7, -9, -10, -11]
    assert not is_monotonic(array)

    # Test Case 13
    array = [-1, -1, -2, -3, -4, -5, -5, -5, -6, -7, -8, -8, -9, -10, -11]
    assert is_monotonic(array)

def test_spiral_traverse():

    # Sample Input and Test Case 1
    array = [
        [1, 2, 3, 4],
        [12, 13, 14, 5],
        [11, 16, 15, 6],
        [10, 9, 8, 7]]

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    result = spiral_traverse(array)

    assert result == expected

    # Test Case 2
    array = [[1]]

    result = spiral_traverse(array)

    assert result == [1]

    # Test Case 3
    array = [
        [1, 2],
        [4, 3]]

    result = spiral_traverse(array)

    assert result == [1, 2, 3, 4]

    # Test Case 4
    array = [
        [1, 2, 3],
        [8, 9, 4],
        [7, 6, 5]]

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    result = spiral_traverse(array)

    assert result == expected

    # Test Case 5
    array = [
        [19, 32, 33, 34, 25, 8],
        [16, 15, 14, 13, 12, 11],
        [18, 31, 36, 35, 26, 9],
        [1, 2, 3, 4, 5, 6],
        [20, 21, 22, 23, 24, 7],
        [17, 30, 29, 28, 27, 10]]

    result = spiral_traverse(array)

    expected = [19, 32, 33, 34, 25, 8, 11, 9, 6, 7, 10, 27, 28, 29, 30, 17, 20,
        1, 18, 16, 15, 14, 13, 12, 26, 5, 24, 23, 22, 21, 2, 31, 36, 35, 4, 3]

    assert result == expected

    # Test Case 8
    array = [
        [1, 2, 3, 4],
        [10, 11, 12, 5],
        [9, 8, 7, 6]]

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    result = spiral_traverse(array)

    assert result == expected

    # Test case 9
    array = [
        [1, 2, 3],
        [12, 13, 4],
        [11, 14, 5],
        [10, 15, 6],
        [9, 8, 7]]

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    result = spiral_traverse(array)

    assert result == expected


def test_longest_peak():

    # Test case 1
    array = [1, 2, 3, 3, 4, 0, 10, 6, 5, -1, -3, 2, 3]

    expected = 6

    result = longest_peak(array)

    assert result == expected

    # Test case 2
    array = []
    result = longest_peak(array)
    assert result == 0

    # Test case 7
    array = [5, 4, 3, 2, 1, 2, 10, 12]
    assert longest_peak(array) == 0

    # Test case 8
    array = [1, 2, 3, 4, 5, 6, 10, 100, 1000]
    assert longest_peak(array) == 0

    # Test case 9
    array = [1, 2, 3, 3, 2, 1]
    assert longest_peak(array) == 0    


def test_array_of_products():

    # Test case 1
    array = [5, 1, 4, 2]

    assert array_of_products(array) == [8, 40, 10, 20]

    # Test Case 2
    array = [1, 8, 6, 2, 4]

    assert array_of_products(array) == [384, 48, 64, 192, 96]

    # Test Case 3
    array = [-5, 2, -4, 14, -6]

    assert array_of_products(array) == [672, -1680, 840, -240, 560]


def test_brute_force_array_of_products():

    # Test case 1
    array = [5, 1, 4, 2]

    assert brute_force_array_of_products(array) == [8, 40, 10, 20]

    # Test Case 2
    array = [1, 8, 6, 2, 4]

    assert brute_force_array_of_products(array) == [384, 48, 64, 192, 96]

    # Test Case 3
    array = [-5, 2, -4, 14, -6]

    assert brute_force_array_of_products(array) == [672, -1680, 840, -240, 560]


def test_first_duplicate_value_with_ds():

    # Test case 3
    array = [1, 1, 2, 3, 3, 2, 2]

    assert first_duplicate_value_with_ds(array) == 1

    # Test case 4
    array = [3, 1, 3, 1, 1, 4, 4]

    assert first_duplicate_value_with_ds(array) == 3

    # Test case 5
    array = []

    assert first_duplicate_value_with_ds(array) == -1


def test_remove_islands():

    # Sample Input and Test Case 1
    matrix = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1]]
    result = remove_islands(matrix)

    expected = [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1]]

    assert result == expected

    # Test Case 2
    matrix = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]]
    result = remove_islands(matrix)
    expected = [
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1]]

    assert result == expected
