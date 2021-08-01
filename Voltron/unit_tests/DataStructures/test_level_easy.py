###############################################################################
# @details
# Example Usage:
#
# pytest DataStructures/test_level_easy.py
###############################################################################

# Import in order of appearance.
from Voltron.DataStructures.level_easy import (
    BinaryTree,
    _make_disparate_input_nodes,
    _get_next_level_children,
    node_depths,
    find_closest_value_in_bst,
    _get_closest_value_iterative
    )
from Voltron.DataStructures.binary_search_tree import Node as BSTNode

from collections import deque

import pytest


test_values = [5, 15, 2, 5, 13, 22, 1, 14]
closest_value_bst_test_values = [
    5, 502, 55000, 1001, 4500, 204, 205, 207, 208, 206, 203, 15, 22, 57,
    60, 5, 2, 3, 1, 1, 1, 1, 1, -51, -403]

def test_test_case_1():
    test_nodes = _make_disparate_input_nodes(list(range(1, 10)))

    assert test_nodes[0].value == 1
    assert test_nodes[1].value == 2
    assert test_nodes[2].value == 3

    test_nodes[0].left = test_nodes[1]
    test_nodes[0].right = test_nodes[2]

    test_nodes[1].left = test_nodes[3]
    test_nodes[1].right = test_nodes[4]

    test_nodes[2].left = test_nodes[5]
    test_nodes[2].right = test_nodes[6]

    test_nodes[3].left = test_nodes[7]
    test_nodes[3].right = test_nodes[8]

    assert node_depths(test_nodes[0]) == 16


def test_find_closest_value_in_BST():
    r = BSTNode(10)
    r.insert_values(test_values)

    result = find_closest_value_in_bst(r, 12)

    assert result == 13

def test_find_closest_value_in_BST_test_case_2():
    r = BSTNode(100)
    r.insert_values([
        5, 502, 204, 55000, 1001, 4500, 203, 205, 207, 206, 208, 2, 15])

    result = find_closest_value_in_bst(r, 100)

    assert result == 100

def test_find_closest_value_in_BST_test_case_2b():
    r = BSTNode(100)
    r.insert_values(closest_value_bst_test_values)

    result = find_closest_value_in_bst(r, 100)

    assert result == 100


def test_find_closest_value_in_BST_test_case_5():
    r = BSTNode(100)
    r.insert_values(closest_value_bst_test_values)

    result = find_closest_value_in_bst(r, 4501)

    assert result == 4500

def test_find_closest_value_in_BST_test_case_6():
    r = BSTNode(100)
    r.insert_values(closest_value_bst_test_values)

    result = find_closest_value_in_bst(r, -51)

    assert result == -51

def test_find_closest_value_in_BST_test_case_7():
    r = BSTNode(1001)
    r.insert_values(closest_value_bst_test_values)

    result = find_closest_value_in_bst(r, 2000)

    assert result == 1001

def test_find_closest_value_in_BST_test_case_8():
    r = BSTNode(5)
    r.insert_values(closest_value_bst_test_values)

    result = find_closest_value_in_bst(r, 6)

    assert result == 5

def test_get_closest_value_iterative_for_binary_search_trees():
    r = BSTNode(100)
    r.insert_values(closest_value_bst_test_values)

    # Test case 2

    result = _get_closest_value_iterative(r, 100, r.value_)

    assert result == 100

    # Test case 5

    result = _get_closest_value_iterative(r, 4501, r.value_)

    assert result == 4500

    # Test case 6

    result = _get_closest_value_iterative(r, -51, r.value_)

    assert result == -51

    # Test case 7

    result = _get_closest_value_iterative(r, 2000, r.value_)

    assert result == 1001

    # Test case 8

    result = _get_closest_value_iterative(r, 6, r.value_)

    assert result == 5
