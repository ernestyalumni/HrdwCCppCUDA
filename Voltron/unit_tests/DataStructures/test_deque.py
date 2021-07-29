###############################################################################
# @details
# Example Usage:
#
# pytest DataStructures/test_deque.py
###############################################################################

from collections import deque

import pytest


@pytest.fixture
def example_deque_fixture():

    return deque('ghi')


def test_deque_construction():
    x = deque('abcdefghijklmnopqrstuvwxyz')
    assert x[0] == 'a'
    assert x[1] == 'b'
    assert x[2] == 'c'


def test_iteration_order(example_deque_fixture):
    d = example_deque_fixture

def test_append_appends_to_top_of_stack(example_deque_fixture):

    d = example_deque_fixture

    assert d[-1] == 'i'

    d.append('j')
    assert d[-1] == 'j'

    out_element = d.pop()

    assert out_element == 'j'


def test_appendleft_appends_to_front_of_queue(example_deque_fixture):

    d = example_deque_fixture

    assert d[0] == 'g'

    d.appendleft('f')
    assert d[0] == 'f'

    out_element = d.pop()

    assert out_element == 'i'

    out_element = d.popleft()

    assert out_element == 'f'


def test_empty_deque_returns_false(example_deque_fixture):

    d = example_deque_fixture

    assert d

    while (d):
        assert d
        d.pop()

    assert not d