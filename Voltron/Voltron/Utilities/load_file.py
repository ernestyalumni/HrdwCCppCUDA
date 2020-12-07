"""
@file load_file.py

"""

def load_file_as_namespace_dict(filename):
    """
    @brief Loads a matrix from a python file, and constructs a dict from it.

    @url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/assignments/
    @ref main.py of ps1 (Problem Set 1) of MIT OCW 6.006 Fall 2011
    """

    namespace = dict()
    with open(filename) as handle:
        """
        @url https://docs.python.org/3/library/functions.html#exec
        exec(object[, globals[, locals]])
        This function supports dynamic execution of Python code.

        Pass an explicit locals dictionary if you need to see effects of the
        code on locals after function exec() returns

        Return value is None.
        """
        # .read() will read the entire file as a string.
        exec(handle.read(), namespace)
    return namespace


def load_file_as_variable(filename, variable):
    """
    @brief Loads a matrix from a python file, and constructs a variable from it.

    @url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/assignments/
    @ref main.py of ps1 (Problem Set 1) of MIT OCW 6.006 Fall 2011
    """
    return load_file_as_namespace_dict(filename)[variable]
