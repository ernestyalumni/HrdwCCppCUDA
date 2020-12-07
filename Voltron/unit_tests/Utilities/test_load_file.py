"""
@file test_load_file.py

@details

Example Usage:

pytest Utilities/test_load_file.py 
"""
from Voltron.Utilities.load_file import load_file_as_variable

from pathlib import Path

import os
import pytest

@pytest.fixture
def example_filepaths():

    class ExampleFilePaths:
        """
        @url https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory
        """

        # Current path can change depending upon where it's called, e.g. if
        # called in 
        # /home/topolo/PropD/HrdwCCppCUDA/Voltron/unit_tests
        # or called in
        # /home/topolo/PropD/HrdwCCppCUDA/Voltron/unit_tests/Utilities
        current_path = Path.cwd()

        # Since it's dependent upon this file, this path won't change depending
        # upon where user calls it. This gives the file path with file name.
        current_file_path = Path(__file__).resolve()
        # This gives the file path.
        current_file_parent = Path(__file__).parent
        current_file_greatgrandparent = Path(__file__).parent.parent.parent
        relative_problem_filepath = "data/Algorithms/PeakFinding/problem.py"

        problem_filepath = current_file_greatgrandparent / Path(
            relative_problem_filepath)

        problem_filepath = (problem_filepath
            if os.path.exists(problem_filepath) else None)

    return ExampleFilePaths()


def test_load_file_as_variable(example_filepaths):
    if example_filepaths.problem_filepath:

        result = load_file_as_variable(
            example_filepaths.problem_filepath,
            "problemMatrix")

        assert isinstance(result, list)     
        assert isinstance(result[0], list)
        assert len(result) == 11
        assert len(result[0]) == 11

    else:
        assert False