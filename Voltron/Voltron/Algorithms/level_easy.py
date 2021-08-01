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


def minimum_waiting_time(queries):

    if not queries:
        return 0

    # N number of elements.
    # O(N log N) time.
    queries.sort()

    running_sum = queries[0]

    for index, wait_time in enumerate(queries):

        if (index != 0 and index != len(queries) - 1):

            old_value = queries[index]
            queries[index] += running_sum
            running_sum += old_value

    return sum(queries[:-1])


def minimum_waiting_time_optimal(queries):

    if not queries:
        return 0

    # N number of elements.
    # O(N log N) time.
    queries.sort()

    running_sum = 0

    for index, wait_time in enumerate(queries):

        # Remaining queries to consider to add to minimum total wait time.
        queries_left = len(queries) - (index + 1)

        # Add the current wait time queries_left times because the other queries
        # of queries_left will have to wait this time as well.
        running_sum += wait_time * queries_left

    return running_sum


def class_photos(red_shirt_heights, blue_shirt_heights):
    """
    * Each student in the back row must be strictly taller than the student
    directly in front of them in the front row.

    @return Returns whether or not a class photo that follows the stated
    guidelines can be taken.

    @details

    O(nlog(n)) time. O(1) space. n = number of students.
    """
    red_shirt_heights.sort()
    blue_shirt_heights.sort()

    # The shirt color of tallest student will determine which students need to
    # be placed in the back row. Tallest student can't be placed in the front
    # row because there's no student taller than them who can be placed behind
    # them.

    if blue_shirt_heights[-1] > red_shirt_heights[-1]:
        back_row = blue_shirt_heights
        front_row = red_shirt_heights
    elif blue_shirt_heights[-1] < red_shirt_heights[-1]:
        back_row = red_shirt_heights
        front_row = blue_shirt_heights
    else:
        assert blue_shirt_heights[-1] == red_shirt_heights[-1]
        return False

    for index in range(len(red_shirt_heights)):
        i = -(index + 1)
        if (front_row[i] >= back_row[i]):
            return False

    return True


def tandem_bicycle(red_shirt_speeds, blue_shirt_speeds, fastest):
    """
    @details

    Person that pedals faster dictates speed of the bicycle.

    Given 2 lists of positive integers: 1 that contains speeds of riders wearing
    red shirts and 1 that contains speeds of riders wearing blue shirts.

    Your goal is to pair every rider wearing a red shirt with a rider wearing a
    blue shirt to operate tandem bicycle.
    """
    red_shirt_speeds.sort()
    blue_shirt_speeds.sort()



    return 0