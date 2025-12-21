"""
@file test_level_medium.py
"""

from Voltron.Algorithms.level_medium import (
    # In order of usage or appearance.
    # 3. Longest Substring Without Repeating Characters
    LongestSubstringWithoutRepeatingCharacters,
    # 11. Container With Most Water
    ContainerWithMostWater,
    # 15. 3Sum
    three_number_sum,
    three_number_sum_with_sorted_array,
    # 73. Set Matrix Zeroes
    SetMatrixZeroes,
    # 238. Product of Array Except Self
    ProductOfArrayExceptSelf,
    # 424. Longest Repeating Substring With Replacement
    LongestRepeatingCharacterReplacement,
    smallest_difference,
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
    sorted_merge_overlapping_intervals,
    remove_islands
    )

from Voltron.Algorithms.LeetCode import UTF8Validation

import pytest

def test_longest_substring_without_repeating_characters_work():
    """
    3. Longest Substring Without Repeating Characters
    https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
    """
    s = "abcabcbb"
    expected = 3
    assert \
        LongestSubstringWithoutRepeatingCharacters.length_of_longest_substring(s) == \
            expected

    s = "bbbbb"
    expected = 1
    assert \
        LongestSubstringWithoutRepeatingCharacters.length_of_longest_substring(s) == \
            expected

    s = "pwwkew"
    expected = 3
    assert \
        LongestSubstringWithoutRepeatingCharacters.length_of_longest_substring(s) == \
            expected


def test_container_with_most_water_max_area():
    """
    11. Container With Most Water
    """
    height = [1,8,6,2,5,4,8,3,7]    
    expected = 49
    assert ContainerWithMostWater.max_area(height) == expected

    height = [1,1]
    expected = 1
    assert ContainerWithMostWater.max_area(height) == expected


def test_three_number_sum():
    """
    15. 3Sum
    https://leetcode.com/problems/3sum/

    Three Integer Sum
    https://neetcode.io/problems/three-integer-sum
    """

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
    """
    15. 3Sum
    https://leetcode.com/problems/3sum/

    Three Integer Sum
    https://neetcode.io/problems/three-integer-sum
    """

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

"""
https://leetcode.com/problems/set-matrix-zeroes/description/
73. Set Matrix Zeroes
Given an m x n integer matrix matrix, if an element is 0, set its entire row and
column to 0's.
You must do it in place.
"""

def create_set_matrix_zeroes_test_cases():
    example_1 = [[1,1,1],[1,0,1],[1,1,1]]
    output_1 = [[1,0,1],[0,0,0],[1,0,1]]

    example_2 = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    output_2 = [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

    # After test case 200, test case 201
    example_3 = [[-1],[2],[3]]
    output_3 = [[-1],[2],[3]]

    # After test case 143, test case 144
    example_4 = [[1,2,3,4],[5,0,7,8],[0,10,11,12],[13,14,15,0]]
    output_4 = [[0,0,3,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    return [
        example_1,
        output_1,
        example_2,
        output_2,
        example_3,
        output_3,
        example_4,
        output_4]

def test_set_matrix_zeroes_naively():
    example_1, output_1, example_2, output_2, example_3, output_3, _, _ = \
        create_set_matrix_zeroes_test_cases()

    set_zeroes = SetMatrixZeroes.set_zeroes_naive

    assert set_zeroes(example_1) == output_1
    assert set_zeroes(example_2) == output_2
    assert set_zeroes(example_3) == output_3

def test_set_matrix_zeroes_with_negative_1():
    example_1, output_1, example_2, output_2, example_3, output_3, _, _ = \
        create_set_matrix_zeroes_test_cases()

    set_zeroes = SetMatrixZeroes.set_zeroes_with_negative_1

    assert set_zeroes(example_1) == output_1
    assert set_zeroes(example_2) == output_2
    assert set_zeroes(example_3) != output_3

def test_set_matrix_zeroes_with_edges():
    example_1, output_1, example_2, output_2, example_3, output_3, example_4, output_4 = \
        create_set_matrix_zeroes_test_cases()

    set_zeroes = SetMatrixZeroes.set_zeroes_with_edges

    assert set_zeroes(example_1) == output_1
    assert set_zeroes(example_2) == output_2
    assert set_zeroes(example_3) == output_3
    # If we do not check if any 0th row or 0th column has any zeroes, then in
    # the first pass, given x_11=0, x_20=0, x_33=0, x_01, x_10, and x_20, x_00,
    # and x_30, x_03 are marked with 0, respectively.
    # For rows, rows x_1j, x_2j, x_3j are marked with 0's.
    # For columns, columns x_j1, x_j3 are marked with 0's.
    assert set_zeroes(example_4) == output_4

def test_product_except_self():

    # 238. Product of Array Except Self

    # https://neetcode.io/problems/products-of-array-discluding-self

    nums = [1,2,4,6]
    expected = [48,24,12,8]
    assert ProductOfArrayExceptSelf.product_except_self(nums) == expected

    nums = [-1,0,1,2,3]    
    expected = [0,-6,0,0,0]
    assert ProductOfArrayExceptSelf.product_except_self(nums) == expected

    # https://leetcode.com/problems/product-of-array-except-self/description/

    nums = [1,2,3,4]    
    expected = [24,12,8,6]
    assert ProductOfArrayExceptSelf.product_except_self(nums) == expected

    nums = [-1,1,0,-3,3]
    expected = [0,0,9,0,0]
    assert ProductOfArrayExceptSelf.product_except_self(nums) == expected


def test_longest_repeating_character_replacement():
    """
    424. Longest Repeating Substring With Replacement

    https://youtu.be/gqXU1UyA8pk?si=BgYHCLVVMt7P3na7
    """
    # https://neetcode.io/problems/longest-repeating-substring-with-replacement

    s = "XYYX"; k = 2
    expected = 4

    assert LongestRepeatingCharacterReplacement.character_replacement(s, k) == \
        expected

    s = "AAABABB"; k = 1
    expected = 5

    assert LongestRepeatingCharacterReplacement.character_replacement(s, k) == \
        expected

    # https://leetcode.com/problems/longest-repeating-character-replacement/description/

    s = "ABAB"; k = 2
    expected = 4

    assert LongestRepeatingCharacterReplacement.character_replacement(s, k) == \
        expected

    s = "AABABBA"; k = 1
    expected = 4

    assert LongestRepeatingCharacterReplacement.character_replacement(s, k) == \
        expected

    # Passed test cases: 9 / 23 
    s="BAAA"
    k=0

    expected = 3

    assert LongestRepeatingCharacterReplacement.character_replacement(s, k) == \
        expected


def test_sample_case_on_smallest_difference():
    array_one = [-1, 5, 10, 20, 28, 3]
    array_two = [26, 134, 135, 15, 17]

    result = smallest_difference(array_one, array_two)    

    assert result[0] == 28
    assert result[1] == 26


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

    
def test_valid_utf8():
    """
    393. Given an integer array data representing the data, return whether it is
    a valid UTF-8 encoding (i.e. translates to a sequence of valid UTF-8 encoded
    characters)
    """
    example1 = [197, 139, 1]
    example2 = [235, 140, 4]
    example3 = [39,89,227,83,132,95,10,0]

    valid_utf8 = UTF8Validation.valid_utf8
    assert valid_utf8(example1)
    assert not valid_utf8(example2)

    # Expect to be False
    assert not valid_utf8(example3)

def test_utf8_validation_check_most_significant_2_bits_of_byte():
    check_2_bits = UTF8Validation.check_most_significant_2_bits_of_byte

    assert not check_2_bits(197)
    assert check_2_bits(130)
    assert not check_2_bits(1)
    assert not check_2_bits(235)
    assert check_2_bits(140)
    assert not check_2_bits(4)

    assert check_2_bits(197, 1, 1)
    assert check_2_bits(1, 0, 0)
    assert check_2_bits(235, 1, 1)
    assert check_2_bits(4, 0, 0)

def test_utf8_validation_int_to_bitfield():
    int_to_bitfield = UTF8Validation.int_to_bitfield
    result = int_to_bitfield(123)
    assert result == [1, 1, 1, 1, 0, 1, 1]
    result = int_to_bitfield(255)
    assert result == [1, 1, 1, 1, 1, 1, 1, 1]
    result = int_to_bitfield(1234567)
    assert (result ==
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1])

    examples = [240,162,138,147,145]
    examples_2 = [240,162,138,147,17]
    examples_3 = [39,89,227,83,132,95,10,0]