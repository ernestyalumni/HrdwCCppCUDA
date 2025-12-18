from typing import List

class LongestSubstringWithoutRepeatingCharacters:
    """
    3. Longest Substring Without Repeating Characters

    https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
    """
    @staticmethod
    def length_of_longest_substring(s: str) -> int:

        seen_character_to_place = {}        

        maximum_length = 0
        start_index = 0

        N = len(s)

        # Original logic, deal with each case.
        """
        for i in range(N):
            if s[i] not in seen_character_to_place.keys():
                seen_character_to_place[s[i]] = i
                if (maximum_length < (i - start_index + 1)):
                    maximum_length = (i - start_index + 1)
            elif seen_character_to_place[s[i]] >= start_index:
                # When a repeating character is found, start index should be
                # updated to position right after the last occurrence of the
                # repeating character.
                start_index = seen_character_to_place[s[i]] + 1
                seen_character_to_place[s[i]] = i
            else:
                seen_character_to_place[s[i]] = i
                if (maximum_length < (i - start_index + 1)):
                    maximum_length = (i - start_index + 1)
        """
        for i in range(N):

            if (s[i] in seen_character_to_place.keys() and \
                seen_character_to_place[s[i]] >= start_index):
                # When a repeating character is found, start index should be
                # updated to position right after the last occurrence of the
                # repeating character.
                start_index = seen_character_to_place[s[i]] + 1

            seen_character_to_place[s[i]] = i

            maximum_length = max(maximum_length, i - start_index + 1)

        return maximum_length        

class ContainerWithMostWater:
    """
    11. Container With Most Water
    """
    @staticmethod
    def max_area(height):
        """
        :type height: List[int]
        :rtype: int
        """
        l = 0
        r = len(height) - 1
        max_area = 0

        while (l < r):

            area = min(height[l], height[r]) * (r - l)
            max_area = max(area, max_area)

            if (height[l] < height[r]):
                l += 1
            elif (height[r] < height[l]):
                r -= 1
            else:
                # TODO: why is it arbitrary which one we move when both heihgts 
                # are equal?
                r -= 1

        return max_area


def three_number_sum(array, target_sum):
    """
    15. 3Sum
    https://leetcode.com/problems/3sum/

    Three Integer Sum
    https://neetcode.io/problems/three-integer-sum
    """
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
    """
    15. 3Sum
    https://leetcode.com/problems/3sum/

    Three Integer Sum
    https://neetcode.io/problems/three-integer-sum
    """
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

class SetMatrixZeroes:
    """
    https://leetcode.com/problems/set-matrix-zeroes/description/
    73. Set Matrix Zeroes
    Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
    You must do it in place.
    """
    @staticmethod
    def set_zeroes_naive(matrix: List[List[int]]):
        M = len(matrix)
        if (M == 0):
            return matrix
        N = len(matrix[0])
        if (N == 0):
            return matrix
        seen_rows = set()
        seen_columns = set()

        # O(MN) time.
        for i in range(M):
            for j in range(N):
                if matrix[i][j] == 0:
                    seen_rows.add(i)
                    seen_columns.add(j)

        from copy import deepcopy
        new_matrix = deepcopy(matrix)

        # O(MN) time total.
        # O(M) time for range(M)
        for i in range(M):
            if i in seen_rows:
                # O(N) time for range(N)
                new_matrix[i] = [0 for j in range(N)]
        # O(MN) time
        for j in range(N):
            if j in seen_columns:
                for i in range(M):
                    new_matrix[i][j] = 0
        return new_matrix

    @staticmethod
    def set_zeroes_with_negative_1(matrix: List[List[int]]):
        M = len(matrix)
        if (M == 0):
            return matrix
        N = len(matrix[0])
        if (N == 0):
            return matrix

        # O(MN(N + M)) time.
        # if x_ij is marked as -1, then it won't be recognized as a 0, will
        # later be marked as a 0.
        for i in range(M):
            for j in range(N):
                if matrix[i][j] == 0:
                    for j_2 in range(N):
                        if matrix[i][j_2] != 0:
                            matrix[i][j_2] = -1
                    for i_2 in range(M):
                        if matrix[i_2][j] != 0:
                            matrix[i_2][j] = -1
        for i in range(M):
            for j in range(N):
                if matrix[i][j] == -1:
                    matrix[i][j] = 0
        return matrix

class ProductOfArrayExceptSelf:
    """
    https://leetcode.com/problems/product-of-array-except-self/
    https://neetcode.io/problems/products-of-array-discluding-self

    238. Product of Array Except Self
    """
    @staticmethod
    def product_except_self(nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        
        Idea:, given a target product,
        prod_{j=0, i\neq j}^{N-1} a[j], split the product into a "left" product
        and "right" product:
        prod_{j=0}^{i-1} a[j] if i > 0, else 1 and
        prod_{j=i+1}^{N-1} a[j] if i < N-1, else 1
        """        
        N = len(nums)

        # Use output also as an intermediary for left products.
        output = [1 for i in range(N)]

        # Reuse nums to be the right product.
        for i, num in enumerate(nums):

            for j in range(i + 1, N):
                output[j] *= num
            for j in range(0, i):
                nums[j] *= num
            nums[i] = 1

        for i in range(N):
            # Multiply left and right products
            output[i] *= nums[i]

        return output


class LongestRepeatingCharacterReplacement:
    """
    424. Longest Repeating Character Replacement

    Return the length of the longest substring containing the same letter you
    can get after performing the above operations.

    https://leetcode.com/problems/longest-repeating-character-replacement/description/    
    
    The key idea was to recognize that we only care about the total length of
    the string and not what characters are being replaced.

    Other key insight is that s consists only of uppercase English letters.

    Another insight is that the number of characters counted will only be the
    characters within the sliding window, as long as we decrement when we move
    the left pointer.

    https://youtu.be/gqXU1UyA8pk?si=GL5U6iIh3TUqHXmX
    """

    @staticmethod
    def character_replacement(s: str, k: int) -> int:
        N = len(s)

        if (N == k):
            return N

        character_index_to_count = [0 for i in range(26)]

        # Gets count of character with maximum number of counts.
        # O(26)
        def get_max_count():
            max_count = 0
            for i in range(26):
                max_count = max(max_count, character_index_to_count[i])
            return max_count

        # Use this index to shrink the window from the left.
        starting_index = 0

        max_length = 0

        # O(N) time.
        for i in range(N):

            character_index = ord(s[i]) - ord('A')

            character_index_to_count[character_index] += 1

            # Check if sliding window is valid.
            # O(26) time.
            if (i - starting_index + 1) - get_max_count() > k:
                # Remember to decrement the count of the character corresponding
                # to s[starting_index], not starting_index.
                character_index_to_count[ord(s[starting_index]) - ord('A')] -= 1
                starting_index += 1

            # We want to update the length for the longest substring. Update
            # against the length of the sliding window.
            max_length = max(max_length, i - starting_index + 1)

        return max_length


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



"""
Array of Products

Write a function that takes in a non-empty array of integers and returns an
array of the same length, where each element in the output array is equal to the
product of every other number in the input array.

In other words, the value at output[i] is equal to the product of every number
in the input array other than input[i]
"""

def array_of_products(array):
    """
    @details Hint 3. For each index in the input array, try calculating the
    product of every element to the left and the product of every element to the
    right. You can do this with 2 loops through the array: one from left to
    right and one from right to left. How can these products help us?
    """
    N = len(array)
    if (N == 0):
        return array

    products_towards_left = [1 for _ in range(N)]
    products_towards_right = [1 for _ in range(N)]

    product_towards_right = 1
    for i in range(N):
        if (i != 0):
            product_towards_right *= array[i - 1]
            products_towards_right[i] = product_towards_right

    product_towards_left = 1
    for i in range(N - 2, -1, -1):
        product_towards_left *= array[i + 1]
        products_towards_left[i] = product_towards_left

    return [products_towards_right[i] * products_towards_left[i]
        for i in range(N)]


def brute_force_array_of_products(array):
    """
    @details Hint 1. Think about the most naive approach to solving this
    problem. How can we do exactly what the problem wants to do without focusing
    at all on time and space complexity?

    O(N^2) time, O(1) space for product variable, O(N) space for output.
    """
    N = len(array)
    output = []
    for i in range(N):
        product = 1
        for j in range(N):
            if j != i:
                product *= array[j]
        output.append(product)
    return output


"""
First Duplicate Value

Given an array of integers between 1 and n, inclusive, where n is the length of
the array, write a function that returns the first integer that appears more
than once (when the array is read from left to right).

In other words, out of all the integers that might occur more than once in the
input array, your function should return the one whose first duplicate value has
the minimum index.

If no integer appears more than once, your function should return -1.

Note that you're allowed to mutate the input array.
"""

def brute_force_first_duplicate_value(array):
    """
    O(N^2) time.

    The brute-force solution can be done in O(N^2) time. Think about how you can
    determine if a value appears twice in an array.
    """
    N = len(array)
    minimum_index = N

    for i in range(N):
        temp = array[i]
        for j in range(i + 1, N):
            if temp == array[j]:
                if minimum_index > j:
                    minimum_index = j

    if (minimum_index < N):
        return array[minimum_index]

    return -1

def first_duplicate_value_with_ds(array):
    """
    O(N) time, O(N) space.

    Hint 2. You can use a data structure that has constant-time lookups to keep
    to keep track of integers that you've seen already. This leads the way to a
    linear-time solution.
    """
    N = len(array)
    minimum_index = N
    seen_already = [False for i in range(N)]

    for i in range(N):
        temp = array[i]
        if not seen_already[temp - 1]:
            seen_already[temp - 1] = True
        else:
            if (minimum_index > i):
                minimum_index = i

    if (minimum_index < N):
        return array[minimum_index]

    return -1


def first_duplicate_value(array):
    """

    Use the negative sign, at the index that represents an array element's
    value, to represent an element being "seen".
    """
    N = len(array)
    minimum_index = N
    for i in range(N):
        temp = array[i]
        temp = abs(temp) - 1
        if (array[temp] < 0):
            return temp + 1
        else:
            array[temp] *= -1

    return -1


"""
@brief Merge Overlapping Intervals

Write a function that takes a non-empty array of arbitrary intervals, merges any
overlapping intervals, and returns the new intervals in no particular order.
"""

def _is_overlapping(i1, i2):
    if i1 == None:

        return False

    return i1[0] < i2[1] or i2[0] < i1[1]

def _merge_intervals(i1, i2):
    coordinates = [i1[0], i1[1], i2[0], i2[1]]
    min_coordinate = min(coordinates)
    max_coordinate = max(coordinates)
    return [min_coordinate, max_coordinate]

def sorted_merge_overlapping_intervals(intervals):

    """
    @details Hint 2: Sort the intervals with respect to their starting values.
    This will allow you to merge all overlapping intervals in a single traversal
    through the sorted intervals.
    """

    # Sorting

    sorted_intervals = sorted(intervals, key=lambda interval: interval[0])

    result = []

    # Given nonempty array.
    current_interval = intervals[0]

    N = len(intervals)

    for i in range(1, N):

        if (_is_overlapping(current_interval, sorted_intervals[i])):

            current_interval = _merge_intervals(
                current_interval,
                sorted_intervals[i])

            if (i == N - 1):

                result.append(current_interval)

        else:
            result.append(current_interval)

            if (i != N - 1):

                current_interval = None

    return result


def merge_overlapping_intervals(intervals):
    """
    O(N log(N)) time | O(N) space - where N is length of input array.
    """    



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




