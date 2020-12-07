"""
@file peak_problem.py

@url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/assignments/
@ref peak.py of ps1 (Problem Set 1) of MIT OCW 6.006 Fall 2011

1. Algorithmic Thinking, Peak Finding, MIT OCW
MIT 6.006 Introduction to Algorithms, Fall 2011
"""

################################################################################
########################### Class for Peak Problems ############################
################################################################################

# Inherit from object (i.e. class PeakProblem(object)) for Python 2
# compatibility; Python3 style is class PeakProblem:
class PeakProblem(object):
    """
    A class representing an instance of a peak-finding problem.
    """

    def __init__(self, array, bounds):
        """
        A method for initializing an instance of the PeakProblem class.
        Takes an array and an argument indicating which rows to include.

        RUNTIME: O(1)
        """

        (start_row, start_column, number_of_rows, number_of_columns) = bounds

        self.array = array
        self.bounds = bounds
        self.start_row = start_row
        self.start_column = start_column
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns

    def get(self, location):
        """
        @brief Returns value of the array at the given location, offset by the
        coordinates (start_row, start_column).

        RUNTIME: O(1)
        """

        (r, c) = location
        if not (0 <= r and r < self.number_of_rows):
            return 0
        if not (0 <= c and c < self.number_of_columns):
            return 0
        return self.array[self.start_row + r][self.start_column + c]


################################################################################
################################ Helper Methods ################################
################################################################################

def get_dimensions(array):
    """
    Gets the dimensions for a two-dimensional array. The first dimension
    is simply the number of items in the list; the second dimension is the
    length of the shortest row. This ensures that any location (row, col)
    that is less than the resulting bounds will in fact map to a valid
    location in the array.

    RUNTIME: O(len(array))
    """
    rows = len(array)
    columns = 0

    # Gets the maximum for all the row lengths, essentially.
    for row in array:
        if len(row) > columns:
            columns = len(row)

    return (rows, columns)

def create_problem(array):
    """
    Constructs an instance of the PeakProblem object for the given array,
    using bounds derived from the array using the get_dimensions function.

    RUNTIME: O(len(array))
    """

    (rows, columns) = get_dimensions(array)
    return PeakProblem(array, (0, 0, rows, columns))