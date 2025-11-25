"""
USAGE:
Voltron % pytest -s ./unit_tests/Algorithms/test_preeasy_exercises.py
"""
from Voltron.Algorithms.PreEasyExercies import ArrayIndexing

def test_iterate_over_array():
    array = [1, 2, 3, 4, 5]
    assert ArrayIndexing.iterate_over_array(array) == array
    #ArrayIndexing.iterate_over_array(array, True)