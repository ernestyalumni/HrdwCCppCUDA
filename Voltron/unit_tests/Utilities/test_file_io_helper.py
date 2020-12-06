"""
@file test_file_io_helper.py

@details

Example Usage:

pytest Utilities/test_file_io_helper.py 
"""
from Voltron.Utilities.file_io_helper import get_open_filename

"""
@url https://www.linuxjournal.com/content/testing-your-code-pythons-pytest-part-ii
StringIO implements API of a "file" object but exists only in memory and is
effectively a string.
"""
from io import StringIO

def test_get_open_filename_runs_with_no_argument(monkeypatch):
    """
    @fn test_get_open_filename_runs_on_default

    @url https://docs.pytest.org/en/stable/monkeypatch.html
    @details monkeypatch fixture helps you  safely set/delete an attribute,
    etc.

    1. Modifies behavior of function or property of class for a test, e.g.
    there's an API call, etc. Use monkeypatch.setattr to patch function or
    property with desired testing behavior.

    @url https://www.linuxjournal.com/content/testing-your-code-pythons-pytest-part-ii
    """
    example_filename = "problem.py"

    filename_input = StringIO(example_filename)

    # cf. https://stackoverflow.com/questions/51392889/python-pytest-occasionally-fails-with-oserror-reading-from-stdin-while-output-i
    # https://www.linuxjournal.com/content/testing-your-code-pythons-pytest-part-ii
    monkeypatch.setattr('sys.stdin', filename_input)

    result = get_open_filename()
    assert result == example_filename;


def test_get_open_filename_no_argument_no_input(monkeypatch):
    # If example_filename = "", obtain an EOFError: EOF when reading a line
    example_filename = "\n"

    filename_input = StringIO(example_filename)

    # cf. https://stackoverflow.com/questions/51392889/python-pytest-occasionally-fails-with-oserror-reading-from-stdin-while-output-i
    # https://www.linuxjournal.com/content/testing-your-code-pythons-pytest-part-ii
    monkeypatch.setattr('sys.stdin', filename_input)

    result = get_open_filename()
    assert result == "";