"""
@file file_io_helper.py


"""

import os


def get_open_filename(default = None):
    """
    Prompts the user to pick a file name. If the user doesn't enter a filename,
    returns the default.

    @url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/assignments/
    @ref utils.py of ps1 (Problem Set 1) of MIT OCW 6.006 Fall 2011

    @details This function prompts the user for a filename; you can specify
    during the function call (that happens before the user prompt) the default
    filename if the user doesn't provide a filename (an empty string, for
    instance)
    """
    prompt = "Enter a file name to load from"
    if default is not None:
        prompt += (" (default: %s)" % default)
    prompt += ": "

    # Starting with Python 3, raw_input() was renamed to input()
    # filename = raw_input(prompt)
    # input() reads a line from sys.stdin and returns it with trailing newline
    # stripped.
    filename = input(prompt)
    if filename == "" and not (default is None):
        filename = default

    return filename


def get_save_filename(default = None):
    """
    Prompts the user to pick a file name.

    @url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/assignments/
    @ref utils.py of ps1 (Problem Set 1) of MIT OCW 6.006 Fall 2011
    """

    prompt = "Enter a filename to save to"