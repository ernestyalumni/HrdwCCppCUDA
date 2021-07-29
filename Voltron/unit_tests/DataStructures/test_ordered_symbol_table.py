from Voltron.DataStructures.ordered_symbol_table import OrderedSymbolTable

import pytest


@pytest.fixture
def empty_ordered_symbol_table_fixture():

    return OrderedSymbolTable()

@pytest.fixture
def example_symbol_table_fixture():
    st = OrderedSymbolTable()

    st['Sedgewick'] = 'Bob'
    st['Wayne'] = 'Kevin'
    st['Dondero'] = 'Bob'

    return st    

def test_ordered_symbol_table_constructs():

    st = OrderedSymbolTable()
    assert st._root == None

    assert True


def test_setitem(empty_ordered_symbol_table_fixture):

    st = empty_ordered_symbol_table_fixture

    st['Sedgewick'] = 'Bob'
    st['Wayne'] = 'Kevin'
    st['Dondero'] = 'Bob'

    assert 'Sedgewick' in st
    assert 'Wayne' in st
    assert 'Dondero' in st


def test_getitem(example_symbol_table_fixture):

    st = example_symbol_table_fixture

    assert st['Sedgewick'] == 'Bob'
    assert st['Wayne'] == 'Kevin'
    assert st['Dondero'] == 'Bob'


def test_contains(example_symbol_table_fixture):

    st = example_symbol_table_fixture

    assert 'Dondero' in st
    assert 'Kernighan' not in st


def test_iteration(example_symbol_table_fixture):
    st = example_symbol_table_fixture

    expected_keys = ['Dondero', 'Sedgewick', 'Wayne']
    expected_values = ['Bob', 'Bob', 'Kevin']

    counter = 0
    for key in st:
        assert key == expected_keys[counter]
        assert st[key] == expected_values[counter]

        counter += 1