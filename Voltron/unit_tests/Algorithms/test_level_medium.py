"""
@file test_level_medium.py
"""

from Voltron.Algorithms.level_medium import (
    # In order of usage or appearance.
    smallest_difference,
    three_number_sum,
    three_number_sum_with_sorted_array
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