from Voltron.Algorithms.binary_search import (
    binary_search,
    binary_search_iterative,
    binary_search_recursive,
    calculate_midpoint,
    quick_calculate_midpoint_index)

import pytest


def test_calculate_midpoint_works_for_odd_number_of_elements():

    x = calculate_midpoint(0, 4)
    assert x == 2

    x = calculate_midpoint(0, 6)
    assert x == 3

    x = calculate_midpoint(1, 5)
    assert x == 3

    x = calculate_midpoint(1, 7)
    assert x == 4

    x = calculate_midpoint(2, 6)
    assert x == 4

    x = calculate_midpoint(2, 8)
    assert x == 5

def test_calculate_midpoint_gets_left_index_for_even_number_of_elements():

    x = calculate_midpoint(0, 3)
    assert x == 1

    x = calculate_midpoint(0, 5)
    assert x == 2

    x = calculate_midpoint(1, 6)
    assert x == 3

    x = calculate_midpoint(1, 8)
    assert x == 4

    x = calculate_midpoint(2, 7)
    assert x == 4

    x = calculate_midpoint(2, 9)
    assert x == 5

def test_quick_calculate_midpoint_index_for_odd_number_of_elements():

    x = quick_calculate_midpoint_index(0, 4)
    assert x == 2

    x = quick_calculate_midpoint_index(0, 6)
    assert x == 3

    x = quick_calculate_midpoint_index(1, 5)
    assert x == 3

    x = quick_calculate_midpoint_index(1, 7)
    assert x == 4

    x = quick_calculate_midpoint_index(2, 6)
    assert x == 4

    x = quick_calculate_midpoint_index(2, 8)
    assert x == 5

def test_quick_calculate_midpoint_index_for_even_number_of_elements():

    x = quick_calculate_midpoint_index(0, 3)
    assert x == 1

    x = quick_calculate_midpoint_index(0, 5)
    assert x == 2

    x = quick_calculate_midpoint_index(1, 6)
    assert x == 3

    x = quick_calculate_midpoint_index(1, 8)
    assert x == 4

    x = quick_calculate_midpoint_index(2, 7)
    assert x == 4

    x = quick_calculate_midpoint_index(2, 9)
    assert x == 5


def test_binary_search_recursive_works():

    element = 18

    array = [1, 2, 5, 7, 13, 15, 16, 18, 24, 28, 29]

    result = binary_search_recursive(array, element, 0, len(array) - 1)

    assert result == 7

    # Another example.

    element = 20

    array = [4, 14, 16, 17, 19, 21, 24, 28, 30, 35, 36, 38, 39, 40, 41, 43]

    result = binary_search_recursive(array, element, 0, len(array) - 1)

    assert result == None


def test_binary_search_recursive_works_on_all_elements():
    # cf. https://www.geeksforgeeks.org/binary-search/

    array = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]

    assert binary_search_recursive(array, 2, 0, len(array) - 1) == 0
    assert binary_search_recursive(array, 5, 0, len(array) - 1) == 1
    assert binary_search_recursive(array, 8, 0, len(array) - 1) == 2
    assert binary_search_recursive(array, 12, 0, len(array) - 1) == 3
    assert binary_search_recursive(array, 16, 0, len(array) - 1) == 4
    assert binary_search_recursive(array, 23, 0, len(array) - 1) == 5
    assert binary_search_recursive(array, 38, 0, len(array) - 1) == 6
    assert binary_search_recursive(array, 56, 0, len(array) - 1) == 7
    assert binary_search_recursive(array, 72, 0, len(array) - 1) == 8
    assert binary_search_recursive(array, 91, 0, len(array) - 1) == 9


def test_binary_search_iterative_works():

    element = 18

    array = [1, 2, 5, 7, 13, 15, 16, 18, 24, 28, 29]

    result = binary_search_iterative(array, element)

    assert result == 7

    # Another example.

    element = 20

    array = [4, 14, 16, 17, 19, 21, 24, 28, 30, 35, 36, 38, 39, 40, 41, 43]

    result = binary_search_iterative(array, element)

    assert result == None


def test_binary_search_iterative_works_on_all_elements():

    array = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]

    assert binary_search_iterative(array, 2) == 0
    assert binary_search_iterative(array, 5) == 1
    assert binary_search_iterative(array, 8) == 2
    assert binary_search_iterative(array, 12) == 3
    assert binary_search_iterative(array, 16) == 4
    assert binary_search_iterative(array, 23) == 5
    assert binary_search_iterative(array, 38) == 6
    assert binary_search_iterative(array, 56) == 7
    assert binary_search_iterative(array, 72) == 8
    assert binary_search_iterative(array, 91) == 9
