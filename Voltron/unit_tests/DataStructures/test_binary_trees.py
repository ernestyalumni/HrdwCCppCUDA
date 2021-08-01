from Voltron.DataStructures.binary_trees import (
    # In order of usage or appearance
    Node,
    level_order_traversal
    )

import pytest


def test_level_order_traversal():
    """
    @ref https://leetcode.com/problems/binary-tree-level-order-traversal/description/
    @brief Given root of a binary tree, return level order traversal of its
    nodes' values, (i.e. from left to right, level by level).
    """
    r = Node(3)
    r.left = Node(9)
    r.right = Node(20)
    r.right.left = Node(15)
    r.right.right = Node(7)

    result = level_order_traversal(r)

    assert result == [[3], [9, 20], [15, 7]]

def test_level_order_traversal_test_case_15():
    """
    @ref https://leetcode.com/submissions/detail/531496238/
    """
    r = Node(1)
    r.left = Node(2)
    r.right = Node(3)
    r.left.left = Node(4)
    r.right.right = Node(5)

    result = level_order_traversal(r)

    assert result == [[1], [2, 3], [4, 5]]

