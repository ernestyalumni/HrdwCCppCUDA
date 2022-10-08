import sys

def naive_interpolation(
        lhs_function,
        rhs_function,
        start_value = 1,
        max_iterations=sys.maxsize):
    n = start_value
    while (n < sys.maxsize):

        if (lhs_function(n) >= rhs_function(n)):
            return n - 1
        else:
            n += 1

