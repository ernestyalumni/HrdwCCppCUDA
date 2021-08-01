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


def non_constructible_change_all_permutations(coins):

    if not coins:
        return 1

    # cf. https://docs.python.org/3/howto/sorting.html
    # Python lists have built-in list.sort() method that modifies list in place.
    coins.sort()

    # Recognize that minimum_change is monotonically increasing with
    # monotonically increasing coins.
    minimum_change = 1

    possible_change = set()

    if coins[0] > minimum_change:

        return minimum_change

    for coin in coins:

        # O(N!) space

        # O(N!) time copy.
        possible_change_now = list(possible_change)

        possible_change_now.sort()

        for change in possible_change_now:

            possible_change.add(change + coin)

        possible_change.add(coin)

        possible_change_now = list(possible_change)
        possible_change_now.sort()

        for change in possible_change_now:

            if (minimum_change == change):
                minimum_change += 1

    return minimum_change


def non_constructible_change_observe_sum(coins):

    if not coins:
        return 1

    coins.sort()

    running_sum = 0
    minimum_change = 1

    if coins[0] > minimum_change:
        return minimum_change

    for coin in coins:

        if (coin > minimum_change):
            return minimum_change

        running_sum += coin

        if (coin <= minimum_change):

            minimum_change = running_sum + 1

        elif (minimum_change == running_sum):

            minimum_change += 1

    return minimum_change
