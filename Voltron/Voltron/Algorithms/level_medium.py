################################################################################
#
################################################################################

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


def smallest_difference(array_one, array_two):
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

    for target_element, target_index in enumerate(array_one):

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