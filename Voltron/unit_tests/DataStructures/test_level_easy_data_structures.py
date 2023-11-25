###############################################################################
# Example Usage:
#
# pytest DataStructures/test_level_easy.py
# TODO: Figure out why pytest needs to have each file, no matter where it is,
# be a unique name. Otherwise, this is meant to be named test_level_easy.py and
# be disambiguated from the one in the Algorithms subdirectory.
###############################################################################

# Import in order of appearance.
from Voltron.DataStructures.level_easy import (
    BinaryTree,
    _make_disparate_input_nodes,
    _get_next_level_children,
    node_depths,
    find_closest_value_in_bst,
    _get_closest_value_iterative,
    branch_sums,
    TreeNode,
    remove_duplicates_from_linked_list
    )

from Voltron.DataStructures.binary_search_tree import Node as BSTNode
from Voltron.DataStructures.binary_tree import Node
from Voltron.DataStructures.linked_list import Element as LinkedListNode

from collections import deque

import pytest


test_values = [5, 15, 2, 5, 13, 22, 1, 14]
closest_value_bst_test_values = [
    5, 502, 55000, 1001, 4500, 204, 205, 207, 208, 206, 203, 15, 22, 57,
    60, 5, 2, 3, 1, 1, 1, 1, 1, -51, -403]


@pytest.fixture
def sample_branch_sums_test_binary_tree_fixture():

    r = Node(1)
    r.left_ = Node(2)
    r.left_.left_ = Node(4)
    r.left_.right_ = Node(5)
    r.left_.left_.left_ = Node(8)
    r.left_.left_.right_ = Node(9)
    r.left_.right_ = Node(5)
    r.left_.right_.left_ = Node(10)
    r.right_ = Node(3)
    r.right_.left_ = Node(6)
    r.right_.right_ = Node(7)

    return r


@pytest.fixture
def linked_list_duplicates_fixture():
    l = LinkedListNode(1)
    l.next = LinkedListNode(1)
    l.next.next = LinkedListNode(1)
    l.next.next.next = LinkedListNode(3)
    l.next.next.next.next = LinkedListNode(4)
    l.next.next.next.next.next = LinkedListNode(4)
    l.next.next.next.next.next.next = LinkedListNode(4)
    l.next.next.next.next.next.next.next = LinkedListNode(5)
    l.next.next.next.next.next.next.next.next = LinkedListNode(6)
    l.next.next.next.next.next.next.next.next.next = LinkedListNode(6)

    return l

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


def test_branch_sums(sample_branch_sums_test_binary_tree_fixture):
    r = sample_branch_sums_test_binary_tree_fixture
    result = branch_sums(r)

    assert result == [15, 16, 18, 10, 11]


def test_branch_sums_test_case_2():
    r = Node(1)
    result = branch_sums(r)
    assert result == [1,]


def test_branch_sums_test_case_3():
    r = Node(1)
    r.left_ = Node(2)
    result = branch_sums(r)
    assert result == [3,]


def test_branch_sums_test_case_6():
    r = Node(1)
    r.left_ = Node(2)
    r.right_ = Node(3)
    r.left_.left_ = Node(4)
    r.left_.right_ = Node(5)
    r.right_.left_ = Node(6)
    r.right_.right_ = Node(7)
    r.left_.left_.left_ = Node(8)
    r.left_.left_.right_ = Node(9)
    r.left_.right_.left_ = Node(10)
    r.left_.right_.right_ = Node(1)
    r.right_.left_.left_ = Node(1)
    r.right_.left_.right_ = Node(1)

    result = branch_sums(r)
    assert result == [15, 16, 18, 9, 11, 11, 11]


def test_depth_first_search_tree_sample_case():
    r = TreeNode('A')
    r.children_ = [TreeNode('B'), TreeNode('C'), TreeNode('D')]
    r.children_[0].children_ = [TreeNode('E'), TreeNode('F')]
    r.children_[0].children_[1].children_ = [TreeNode('I'), TreeNode('J')]
    r.children_[2].children_ = [TreeNode('G'), TreeNode('H')]
    r.children_[2].children_[0].children_ = [TreeNode('K'),]

    array = []

    r.depth_first_search_recursive(array)

    assert array == ['A', 'B', 'E', 'F', 'I', 'J', 'C', 'D', 'G', 'K', 'H']


def test_remove_duplicates_from_linked_list(linked_list_duplicates_fixture):
    l = linked_list_duplicates_fixture

    assert l.value == 1
    assert l.next.value == 1
    assert l.next.next.value == 1
    assert l.next.next.next.value == 3

    l = remove_duplicates_from_linked_list(l)

    assert l.value == 1
    assert l.next.value == 3
    assert l.next.next.value == 4
    assert l.next.next.next.value == 5
    assert l.next.next.next.next.value == 6
    assert l.next.next.next.next.next == None


def test_remove_duplicates_from_linked_list_test_cases():

    # Test case 2

    l = LinkedListNode(1)
    l.next = LinkedListNode(1)
    l.next.next = LinkedListNode(1)
    l.next.next.next = LinkedListNode(1)
    l.next.next.next.next = LinkedListNode(1)
    l.next.next.next.next.next = LinkedListNode(4)
    l.next.next.next.next.next.next = LinkedListNode(4)
    l.next.next.next.next.next.next.next = LinkedListNode(5)
    l.next.next.next.next.next.next.next.next = LinkedListNode(6)
    l.next.next.next.next.next.next.next.next.next = LinkedListNode(6)

    l = remove_duplicates_from_linked_list(l)

    assert l.value == 1
    assert l.next.value == 4
    assert l.next.next.value == 5
    assert l.next.next.next.value == 6
    assert l.next.next.next.next == None

    # Test case 3

    l = LinkedListNode(1)
    l.next = LinkedListNode(1)
    l.next.next = LinkedListNode(1)
    l.next.next.next = LinkedListNode(1)
    l.next.next.next.next = LinkedListNode(1)
    l.next.next.next.next.next = LinkedListNode(1)
    l.next.next.next.next.next.next = LinkedListNode(1)

    l = remove_duplicates_from_linked_list(l)

    assert l.value == 1
    assert l.next == None
