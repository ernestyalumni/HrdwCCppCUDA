from Voltron.Algorithms.Sorting.selection_sort import SelectionSorting

import pytest

def test_prefix_max():
    example_values = [8, 2, 4, 9, 3]

    assert (SelectionSorting.prefix_max(example_values, 0) == 0)
    assert (SelectionSorting.prefix_max(example_values, 1) == 0)
    assert (SelectionSorting.prefix_max(example_values, 2) == 0)
    assert (SelectionSorting.prefix_max(example_values, 3) == 3)
    assert (SelectionSorting.prefix_max(example_values, 4) == 3)

def test_mit_ocw_selection_sort():
    example_values = [8, 2, 4, 9, 3]

    SelectionSorting.mit_ocw_selection_sort(example_values)

    assert (example_values[0] == 2)
    assert (example_values[1] == 3)
    assert (example_values[2] == 4)
    assert (example_values[3] == 8)
    assert (example_values[4] == 9)