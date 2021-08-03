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

def move_element_to_end(array, to_move):
    N = len(array)
    if (N == 0 or N == 1):
        return array

    # Index to a place where we can put an integer not to be moved to the end.
    # Note that this index monotonically increases.
    index_to_not_move = 0

    for index in range(N):
        if array[index] != to_move:
            if index != index_to_not_move:
                array[index_to_not_move] = array[index]
            index_to_not_move += 1


    
    # Go and assign all values "at the end" to be to_move.
    for index in range(index_to_not_move, N):
        array[index] = to_move

    return array


def is_monotonic(array):
    N = len(array)
    if (N == 0 or N == 1 or N == 2):
        return True

    non_increasing = None

    for index in range(1, N):

        if (array[index - 1] < array[index]):

            if non_increasing == None:

                non_increasing = False

            elif non_increasing == True:

                return False

        elif (array[index - 1] > array[index]):

            if non_increasing == None:
                non_increasing = True
            elif non_increasing == False:
                return False

    return True

def spiral_traverse(array):
    M = len(array)
    if M == 0:
        return []
    if M == 1:
        return array[0]

    N = len(array[0])
    if (N == 0):
        return []
    if (N == 1):
        return [array[i][0] for i in range(M)]

    output = []

    # Think of rectangular perimeter. Rectangular perimeter defined by starting
    # row, starting column, ending row, ending column.
    starting_row = 0 
    starting_column = 0
    ending_row = M - 1
    ending_column = N - 1

    while (starting_row <= ending_row and starting_column <= ending_column):
        # horizontal, top traversal.
        for j in range(starting_column, ending_column + 1):
            output.append(array[starting_row][j])

        # vertical, right traversal.
        for i in range(starting_row + 1, ending_row + 1):
            output.append(array[i][ending_column])

        # horizontal, bottom traversal
        for j in range(ending_column - 1, starting_column - 1, -1):
            # Handle the edge case when there's a single row in the middle of
            # the matrix. In this case, we don't want to double-count the values
            # in this row, which we've already counted in first for loop above.
            if starting_row == ending_row:
                break

            output.append(array[ending_row][j])

        # vertical, left traversal. Remember not to retraverse the beginning.
        for i in range(ending_row - 1, starting_row, -1):
            # Handle the edge case when there's a single column in the middle of
            # the matrix. In this case, we don't want to double-count the values
            # in this column, which we've already counted in the second for loop
            # above.
            if starting_column == ending_column:
                break

            output.append(array[i][starting_column])

        starting_row += 1
        ending_row -= 1
        starting_column += 1
        ending_column -= 1

    return output

def _find_peaks(array):
    N = len(array)
    if N < 3:
        return []

    output = []

    for index in range(1, N - 1):
        if array[index - 1] < array[index] and array[index] > array[index + 1]:
            output.append(index)

    return output

def _find_peak_length(array, peak_index):
    left_index = peak_index - 1
    right_index = peak_index + 1
    N = len(array)

    length = 3
    while (left_index - 1 >= 0 and array[left_index - 1] < array[left_index]):
        length += 1
        left_index -= 1
    while (right_index + 1 < N and array[right_index + 1] < array[right_index]):
        length += 1
        right_index += 1

    return length


def longest_peak(array):
    """
    @details

    Longest peak implies find all peaks, and then find longest one. Separate
    into 2 peaks.
    1. Find all peaks, and 2. find longest one.

    Define what a peak is. Identify peak as strictly greater than adjacent
    values.
    """
    peaks = _find_peaks(array)

    peak_lengths = []

    for peak_index in peaks:
        peak_lengths.append(_find_peak_length(array, peak_index))

    if len(peak_lengths) == 0:
        return 0

    return max(peak_lengths)

def _get_neighbors(matrix, i, j):
    neighbors = []

    M = len(matrix)
    N = len(matrix[i])

    # Make checks if it's within the matrix size.

    # up
    if i - 1 >= 0: 
        neighbors.append((i - 1, j))
    # down
    if i + 1 < M:
        neighbors.append((i + 1, j))
    # left
    if (j - 1) >= 0:
        neighbors.append((i, j - 1))
    # right
    if (j + 1) < N:
        neighbors.append((i, j + 1))

    return neighbors

def _find_ones_connected_to_border(
        matrix,
        start_i,
        start_j,
        ones_connected_to_border):
    stack = [(start_i, start_j),]

    while (len(stack) > 0):
        i, j = stack.pop()

        already_visited = ones_connected_to_border[i][j]

        if not already_visited:

            ones_connected_to_border[i][j] = True

            neighbors = _get_neighbors(matrix, i, j)

            for neighbor in neighbors:

                row, col = neighbor

                if matrix[row][col] == 1:

                    stack.append(neighbor)


def remove_islands(matrix):
    """
    @details

    Think about the non-islands that have 1. They are connected to the border.
    """
    M = len(matrix)
    N = len(matrix[0])
    ones_connected_to_border = [[False for j in range(N)] for i in range(M)]

    # Find all the 1s that are not islands.

    for i in range(M):
        for j in range(N):
            i_is_border = i == 0 or i == M - 1
            j_is_border = j == 0 or j == M - 1
            is_border = i_is_border or j_is_border

            if is_border and matrix[i][j] == 1:
                _find_ones_connected_to_border(
                    matrix,
                    i,
                    j,
                    ones_connected_to_border)

    # Locate islands from ones_connected_to_border and mark them as 0.
    # Check only interior; by definition only interior has possibly islands.
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if not ones_connected_to_border[i][j]:
                matrix[i][j] = 0


    return matrix




