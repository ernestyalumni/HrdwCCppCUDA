from Voltron.Algorithms.LeetCode.ReduceNumberToZero import (
    # In order of appearance or usage.
    ReduceNumberToZero,
    one_line)

import pytest


@pytest.fixture
def setup_test_fixture():
    solution = ReduceNumberToZero()
    setup_fixture = {
        "solution1": solution
    }

    return setup_fixture


def test_ReduceNumberToZero(setup_test_fixture):
    sol1 = setup_test_fixture["solution1"]
    assert sol1.number_of_steps(14) == 6

    assert sol1.number_of_steps(8) == 4

    assert sol1.number_of_steps(123) == 12


def test_one_line():
    assert one_line(14) == 6

    assert one_line(8) == 4

    assert one_line(123) == 12
