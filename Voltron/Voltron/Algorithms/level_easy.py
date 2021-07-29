################################################################################


def get_nth_fibonacci_recursive(n):
    """
    @brief nth number is the sum of the (n - 1)th and (n-2)th numbers.

    O(2^n) time. Since all computational branches always end in leaves valued in
    1.
    """
    assert n >= 0

    if (n == 0 or n == 1):
        return 0

    if (n == 2):
        return 1

    return (get_nth_fibonacci_recursive(n - 1) +
      get_nth_fibonacci_recursive(n - 2))

def fibonacci_no_branch(n, value = 1, previous = 0):
    """
    @ref https://stackoverflow.com/questions/47871051/big-o-time-complexity-for-this-recursive-fibonacci
    """

    if (n == 0 or n == 1):
        return previous
    if (n == 2):
        return value

    return fibonacci_no_branch(n - 1, value + previous, value)


def get_nth_fibonacci_recursive_no_branch(n):
    return fibonacci_no_branch(n)

