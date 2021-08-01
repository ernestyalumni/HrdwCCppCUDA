from Voltron.DataStructures.binary_tree import (
    # In order of usage or appearance
    Node,
    insert_values_into_binary_tree,
    depth_first_search_recursive
    )

import pytest

test_values = [5, 3, 9, 2, 4, 8, 11, 14]

@pytest.fixture
def sample_binary_tree_fixture():

    r = Node(6)

    r.insert_values(test_values)

    return r

