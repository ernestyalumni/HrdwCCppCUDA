"""
@file test_level_easy.py

@details

Example Usage:

pytest Utilities/test_level_easy.py 
"""
from Voltron.Algorithms.level_easy import (
    # In order of appearance or usage.
    # 1. Two Sum
    TwoSum,
    # 104. Maximum Depth of Binary Tree
    MaximumDepthOfBinaryTree,
    # 121. Best Time to Buy and Sell Stock
    BestTimeToBuyAndSellStock,
    # 217. Contains Duplicate
    ContainsDuplicate,
    # 242. Valid Anagram
    ValidAnagram,
    two_number_sum,
    validate_subsequence,
    validate_subsequence_with_for_loop,
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
    find_three_largest_numbers,
    is_palindrome,
    caesar_cipher_encryptor,
    run_length_encoding,
    generate_document,
    first_non_repeating_character_with_ordered_dict
    )

from Voltron.Algorithms.Sorting.bubble_sort import (
    bubble_sort_naive,
    bubble_sort_optimized,
    bubble_sort_no_extra_swaps
    )
from Voltron.Algorithms.Sorting.insertion_sort import (
    insertion_sort,
    insertion_sort_optimized
    )
from Voltron.Algorithms.Sorting.selection_sort import selection_sort


from collections import OrderedDict

import pytest

@pytest.fixture
def bubble_sort_test_cases_fixture():

    test_cases = OrderedDict()

    test_cases["Test case 1"] = [8, 5, 2, 9, 5, 6, 3]
    test_cases["Test case 2"] = [1]
    test_cases["Test case 3"] = [1, 2]
    test_cases["Test case 4"] = [2, 1]
    test_cases["Test case 5"] = [1, 3, 2]
    test_cases["Test case 6"] = [3, 1, 2]
    test_cases["Test case 7"] = [1, 2, 3]
    test_cases["Test case 8"] = [
        -4, 5, 10, 8, -10, -6, -4, -2, -5, 3, 5, -4, -5, -1, 1, 6, -7, -6, -7,
        8]
    test_cases["Test case 9"] = [
        -7, 2, 3, 8, -10, 4, -6, -10, -2, -7, 10, 5, 2, 9, -9, -5, 3, 8]

    return test_cases

def test_two_sum():
    """
    1. Two Sum
    """
    # https://leetcode.com/problems/two-sum/description/

    nums = [2,7,11,15]; target = 9

    assert set(TwoSum.two_sum(nums, target)) == set([0,1])

    nums = [3,2,4]; target = 6

    assert set(TwoSum.two_sum(nums, target)) == set([1,2])

    nums = [3,3]; target = 6

    assert set(TwoSum.two_sum(nums, target)) == set([0,1])

    # https://neetcode.io/problems/two-integer-sum

    nums = [3,4,5,6]; target = 7

    assert set(TwoSum.two_sum(nums, target)) == set([0,1])

    nums = [4,5,6]; target = 10

    assert set(TwoSum.two_sum(nums, target)) == set([0,2])

    nums = [5,5]; target = 10

    assert set(TwoSum.two_sum(nums, target)) == set([0,1])

# 104. Maximum Depth of Binary Tree

def create_maximum_depth_of_binary_tree_test_cases():
    TreeNode = MaximumDepthOfBinaryTree.TreeNode
    # Example 1
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    expected = 3

    # Example 2
    root_2 = TreeNode(1)
    root_2.right = TreeNode(2)
    expected_2 = 2

    return [root, expected], [root_2, expected_2]

def test_maximum_depth_of_binary_tree_iterative():
    test_cases = create_maximum_depth_of_binary_tree_test_cases()
    for root, expected in test_cases:
        assert MaximumDepthOfBinaryTree.max_depth_iterative(root) == expected

def test_maximum_depth_of_binary_tree_recursive():
    test_cases = create_maximum_depth_of_binary_tree_test_cases()
    for root, expected in test_cases:
        assert MaximumDepthOfBinaryTree.max_depth_recursive(root) == expected

def test_best_time_to_buy_and_sell_stock_max_profit():
    """
    121. Best Time to Buy and Sell Stock
    """
    prices = [7,1,5,3,6,4]
    expected = 5
    assert BestTimeToBuyAndSellStock.max_profit(prices) == expected

    prices = [7,6,4,3,1]
    expected = 0
    assert BestTimeToBuyAndSellStock.max_profit(prices) == expected

    # https://neetcode.io/problems/buy-and-sell-crypto    

    prices = [10,1,5,6,7,1]
    expected = 6
    assert BestTimeToBuyAndSellStock.max_profit(prices) == expected

    prices = [10,8,7,5,2]
    expected = 0
    assert BestTimeToBuyAndSellStock.max_profit(prices) == expected


def test_contains_duplicates():
    """
    217. Contains Duplicate

    Given an integer array nums, return true if any value appears at least
    twice in the array, and return false if every element is distinct.

    Ideas: Array.    
    """
    nums = [1,2,3,1]
    assert ContainsDuplicate.contains_duplicate(nums)

    nums = [1,2,3,4]
    assert not ContainsDuplicate.contains_duplicate(nums)

    nums = [1,1,1,3,3,4,3,2,4,2]
    assert ContainsDuplicate.contains_duplicate(nums)

def test_valid_anagram():
    """
    242. Valid Anagram.

    Given two strings s and t, return true if the two strings are anagrams of
    each other, otherwise return false.
    """
    s = "anagram"
    t = "nagaram"

    assert ValidAnagram.is_anagram(s, t)

    s = "rat"
    t = "car"

    assert not ValidAnagram.is_anagram(s, t)

    # https://neetcode.io/problems/is-anagram
    s = "racecar"
    t = "carrace"

    assert ValidAnagram.is_anagram(s, t)

    s = "jar"; t = "jam"

    assert not ValidAnagram.is_anagram(s, t)


def test_two_number_sum():
    array = [3, 5, -4, 8, 11, 1, -1, 6]
    target_sum = 10

    two_number_sum(array, target_sum)


def test_validate_subsequence():
    example_array = [5, 1, 22, 25, 6, -1, 8, 10]
    example_sequence = [1, 6, -1, 10]
    assert validate_subsequence(example_array, example_sequence)
    assert validate_subsequence_with_for_loop(example_array, example_sequence)


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


def test_bubble_sort_naive(bubble_sort_test_cases_fixture):

    test_cases = bubble_sort_test_cases_fixture

    for key in test_cases.keys():

        assert bubble_sort_naive(test_cases[key]) == sorted(test_cases[key])


def test_bubble_sort_optimized(bubble_sort_test_cases_fixture):

    test_cases = bubble_sort_test_cases_fixture

    for key in test_cases.keys():

        assert bubble_sort_optimized(test_cases[key]) == sorted(test_cases[key])


def test_bubble_sort_optimized(bubble_sort_test_cases_fixture):

    test_cases = bubble_sort_test_cases_fixture

    for key in test_cases.keys():

        assert (bubble_sort_no_extra_swaps(test_cases[key]) ==
            sorted(test_cases[key]))


def test_insertion_sort(bubble_sort_test_cases_fixture):

    test_cases = bubble_sort_test_cases_fixture

    for key in test_cases.keys():

        assert insertion_sort(test_cases[key]) == sorted(test_cases[key])


def test_insertion_sort_optimized(bubble_sort_test_cases_fixture):

    test_cases = bubble_sort_test_cases_fixture

    for key in test_cases.keys():

        assert (insertion_sort_optimized(test_cases[key]) ==
            sorted(test_cases[key]))


def test_selection_sort(bubble_sort_test_cases_fixture):

    test_cases = bubble_sort_test_cases_fixture

    for key in test_cases.keys():

        assert selection_sort(test_cases[key]) == sorted(test_cases[key])


def test_is_palindrome():

    # Sample input and Test Case 1
    x = "abcdcba"
    assert is_palindrome(x)

    # Test Case 2
    x = "a"
    assert is_palindrome(x)

    # Test Case 3
    x = "ab"
    assert not is_palindrome(x)

    # Test Case 4
    x = "aba"
    assert is_palindrome(x)

    # Test Case 5
    x = "abb"
    assert not is_palindrome(x)


def test_is_caesar_cipher_encrpytor():

    # Sample input and Test Case 1
    s = "xyz"
    result = caesar_cipher_encryptor(s, 2)
    assert result == "zab"

    # Test Case 2
    s = "abc"
    result = caesar_cipher_encryptor(s, 0)
    assert result == "abc"

    # Test Case 3
    result = caesar_cipher_encryptor(s, 3)
    assert result == "def"

    # Test Case 4
    s = "xyz"
    result = caesar_cipher_encryptor(s, 5)
    assert result == "cde"

    # Test Case 5
    s = "abc"
    result = caesar_cipher_encryptor(s, 26)
    assert result == "abc"

def test_run_length_encoding_sample_input():
    # Sample Input and Test Case 1

    s = "AAAAAAAAAAAAABBCCCCDD"
    result = run_length_encoding(s)
    assert result == "9A4A2B4C2D"

def test_run_length_encoding_test_cases():
    # Sample Input and Test Case 1

    s = "[(aaaaaaa,bbbbbbb,ccccc,dddddd)]"
    result = run_length_encoding(s)
    assert result == "1[1(7a1,7b1,5c1,6d1)1]"

def test_generate_document():

    # Sample and Test Case 1.
    characters = "Bste!hetsi ogEAxpelrt x "
    document = "AlgoExpert is the Best!"
      
    assert generate_document(characters, document)

    # Sample and Test case 2

    characters = "A"
    document = "a"
      
    assert not generate_document(characters, document)

    # Test Case 3

    characters = "a"
    document = "a"

    assert generate_document(characters, document)

    # Test Case 4

    characters = "a hsgalhsa sanbjksbdkjba kjx",
    document = ""

    assert generate_document(characters, document)

    # Test Case 5

    characters = " "
    document = "hello"

    assert not generate_document(characters, document)

def test_first_non_repeating_character_with_ordered_dict():

    # Sample Input and Test Case 1

    s = "abcdcaf"
    results = first_non_repeating_character_with_ordered_dict(s)
    assert results == 1

    # Test Case 2

    s = "faadabcbbebdf"
    results = first_non_repeating_character_with_ordered_dict(s)
    assert results == 6

    # Test Case 3

    s = "a"
    results = first_non_repeating_character_with_ordered_dict(s)
    assert results == 0

    # Test Case 4

    s = "ab"
    results = first_non_repeating_character_with_ordered_dict(s)
    assert results == 0

    # Test Case 5

    s = "abc"
    results = first_non_repeating_character_with_ordered_dict(s)
    assert results == 0
