################################################################################
#
################################################################################

def three_number_sum(array, target_sum):
    output = set()
    N = len(array)

    for i in range(N):
        x = array[i]

        remaining_sum = dict()

        for j in range(i + 1, N):
            z = target_sum - array[j] - x
            remaining_sum[z] = array[j]

        for j in range(i + 1, N):

            if (array[j] in remaining_sum and
                    remaining_sum[array[j]] != array[j]):
                solution = [x, array[j], remaining_sum[array[j]]]
                solution.sort()
                output.add(tuple(solution))

    output = [list(x) for x in output]
    output.sort()

    return output


def three_number_sum_with_sorted_array(array, target_sum):
    output = []
    N = len(array)
    array.sort()

    for i in range(N - 2):
        left_index = i + 1
        right_index = N - 1

        while left_index < right_index:

            current_sum = array[i] + array[left_index] + array[right_index]

            if current_sum == target_sum:
                output.append([array[i], array[left_index], array[right_index]])
                left_index += 1
                right_index -= 1

            # Use the monotonicity property of sorted array.
            if target_sum < current_sum:

                right_index -= 1

            if current_sum < target_sum:

                left_index += 1

    return output


def _find_closest_value_index(array, target_value, l, r):
    """
    @param l left index
    @param r right index
    """
    N = len(array)

    if N == 0:
        return None

    if N == 1 or l == r:
        return l

    if (l - r == 1):
        if (abs(array[l] - target_value) < abs(array[r] - target_value)):
            return l
        else:
            return r

    # TODO, do subtraction instead of addition for large arrays, due to
    # overflow possibility.

    midpoint_index = (l + r) // 2
    midpoint_value = array[midpoint_index]

    if (target_value < midpoint_value):
        return _find_closest_value_index(
            array,
            target_value,
            l,
            midpoint_index - 1)

    if (midpoint_value < target_value):
        return _find_closest_value_index(
            array,
            target_value,
            midpoint_index + 1,
            r)

    assert midpoint_value == target_value

    return midpoint_index


def smallest_difference_first_try(array_one, array_two):
    """
    @brief Find pair of numbers, one from each array, whose absolute difference
    is closest to 0, and returns an array containing these 2 numbers, with the
    number from the first array in the first position.
    """
    array_one.sort()
    array_two.sort()

    array_1 = []
    array_2 = []

    #if (len(array_one) <= len(array_two)):
    #    array_1 = array_one
    #    array_2 = array_two
    #else:
    #    array_1 = array_two
    #    array_2 = array_one

    value_difference = None
    current_index_1 = None
    current_index_2 = None

    for target_index, target_element in enumerate(array_one):

        resulting_index = _find_closest_value_index(
            array_two,
            target_element,
            0,
            len(array_two) - 1)

        difference = abs(array_two[resulting_index] - target_element)

        if (not value_difference or value_difference > difference):
            current_index_1 = target_index
            current_index_2 = resulting_index
            value_difference = difference

    return [array_one[current_index_1], array_two[current_index_2]]


def smallest_difference(array_one, array_two):
    array_one.sort()
    array_two.sort()

    ptr_to_array_one = 0
    ptr_to_array_two = 0

    output = []

    smallest_difference = abs(array_one[0] - array_two[0])
    current_difference = abs(array_one[0] - array_two[0])

    while (ptr_to_array_one < len(array_one) and
            ptr_to_array_two < len(array_two)):

        x = array_one[ptr_to_array_one]
        y = array_two[ptr_to_array_two]

        if x < y:
            current_difference = y - x
            ptr_to_array_one += 1
        elif y < x:
            current_difference = x - y
            ptr_to_array_two += 1
        else:
            assert x == y
            return [x, y]

        if current_difference < smallest_difference:
            smallest_difference = current_difference
            output = [x, y]

    return output