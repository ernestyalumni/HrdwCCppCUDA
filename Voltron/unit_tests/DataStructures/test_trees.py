from Voltron.DataStructures.trees import TreeNode

import pytest

def test_breadth_first_search():

    # Sample Input and Test Case #1
    t = TreeNode("A")
    t.add_child("B")
    t.add_child("C")
    t.add_child("D")
    t.children_[0].add_child("E")
    t.children_[0].add_child("F")
    t.children_[2].add_child("G")
    t.children_[2].add_child("H")
    t.children_[0].children_[1].add_child("I")
    t.children_[0].children_[1].add_child("J")
    t.children_[2].children_[0].add_child("K")

    expected = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

    array = []
    t.breadth_first_search(array)

    assert array == expected