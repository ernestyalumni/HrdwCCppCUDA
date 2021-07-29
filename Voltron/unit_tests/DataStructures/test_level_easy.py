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
    node_depths
    )

from collections import deque

import pytest


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