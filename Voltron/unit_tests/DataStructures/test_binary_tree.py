from Voltron.DataStructures.binary_tree import (
    # In order of usage or appearance
    Node,
    insert_values_into_binary_tree,
    dfs_inorder_traversal_recursive,
    dfs_inorder_traversal_iterative,
    dfs_inorder_traversal_iterative_with_visited,
    depth_first_search_recursive
    )

import pytest

expected_inorder_result = list(range(1, 16))

@pytest.fixture
def inorder_test_binary_tree():

    r = Node(8)
    r.left_ = Node(4)
    r.left_.left_ = Node(2)
    r.left_.right_ = Node(6)
    r.left_.left_.left_ = Node(1)
    r.left_.left_.right_ = Node(3)
    r.left_.right_.left_ = Node(5)
    r.left_.right_.right_ = Node(7)
    r.right_ = Node(12)
    r.right_.left_ = Node(10)
    r.right_.right_ = Node(14)
    r.right_.left_.left_ = Node(9)
    r.right_.left_.right_ = Node(11)
    r.right_.right_.left_ = Node(13)
    r.right_.right_.right_ = Node(15)

    return r


def test_dfs_inorder_traversal_recursive(inorder_test_binary_tree):
    r = inorder_test_binary_tree
    results = dfs_inorder_traversal_recursive(r)
    assert results == expected_inorder_result


def test_dfs_inorder_traversal_iterative(inorder_test_binary_tree):
    r = inorder_test_binary_tree
    results = dfs_inorder_traversal_iterative(r)
    assert results == expected_inorder_result


def test_dfs_inorder_traversal_iterative_with_visited(inorder_test_binary_tree):
    r = inorder_test_binary_tree
    results = dfs_inorder_traversal_iterative_with_visited(r)
    assert results == expected_inorder_result