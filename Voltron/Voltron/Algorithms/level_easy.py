from collections import OrderedDict

def validate_subsequence(array, sequence):
    """
    \details Subsequence doesn't have to be adjacent. They have to appear in the
    same order.

    We're going to have to traverse both of the sequences.
    Subsequence cares about order so we look for the first element first. Thus,
    use a "pointer."
    """

    # O(1) space complexity.
    array_index = 0
    sequence_index = 0
    # O(N) time.
    while array_index < len(array) and sequence_index < len(sequence):
        if array[array_index] == sequence[sequence_index]:
            sequence_index += 1
        array_index +=1

    # Condition of success is if this is true:
    return sequence_index == len(sequence)


def validate_subsequence_with_for_loop(array, sequence):
    """
    \details Subsequence doesn't have to be adjacent. They have to appear in the
    same order.

    We're going to have to traverse both of the sequences.
    Subsequence cares about order so we look for the first element first. Thus,
    use a "pointer."
    """

    # O(1) space complexity.
    sequence_index = 0
    # O(N) time.

    for value in array:
        if sequence_index == len(sequence):
            break
        if sequence[sequence_index] == value:
            sequence_index += 1

    # Condition of success is if this is true:
    return sequence_index == len(sequence)


def two_number_sum(array, target_sum):

    solution = []

    result_number = dict()

    # O(N) time complexity, O(N) space for result number.
    for x in array:
        y = target_sum - x
        result_number[y] = x

    # O(N) time complexity.
    for y in array:
        # Given assumption of unique numbers.
        if y in result_number and result_number[y] != y:
            solution.append(y)
            solution.append(result_number[y])
            break
    return solution

def transpose_matrix(matrix):
    N = len(matrix[0])
    M = len(matrix)

    transposed_matrix = []
    for i in range(N):
        new_row = []
        for j in range(M):
            new_row.append(matrix[j][i])
        transposed_matrix.append(new_row)
    return matrix

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

def get_nth_fibonacci_memoization(n):
    if (n == 0 or n == 1):
        return 0
    if (n == 2):
        return 1
    last_two = [0, 1]

    # Use the counter because we want the nth fibonacci number. Continue
    # operations until we reach n, from counting from 3, 4, ...    
    counter = 3
    while (counter <= n):
        next_fibonacci_number_in_sequence = last_two[1] + last_two[0]
        last_two[0] = last_two[1]
        last_two[1] = next_fibonacci_number_in_sequence
        counter += 1    

    return last_two[1]


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

def _calculate_total_tandem_speed(a, b):
    sum = 0
    for x, y in zip(a, b):
        sum += max(x, y)
    return sum


def tandem_bicycle(red_shirt_speeds, blue_shirt_speeds, fastest):
    """
    @details

    Person that pedals faster dictates speed of the bicycle.

    Given 2 lists of positive integers: 1 that contains speeds of riders wearing
    red shirts and 1 that contains speeds of riders wearing blue shirts.

    Your goal is to pair every rider wearing a red shirt with a rider wearing a
    blue shirt to operate tandem bicycle.
    """
    # O(n log n) time.
    red_shirt_speeds.sort()
    blue_shirt_speeds.sort()

    if fastest:
        red_shirt_speeds.reverse()

    return _calculate_total_tandem_speed(red_shirt_speeds, blue_shirt_speeds)

def _process_nested_list_element_recursive(element, level):
    """
    @details Assume we know that element is already a nested list.
    """
    assert isinstance(element, list)

    level_sum = 0

    if len(element) == 0:
        return 0

    for x in element:

        if not isinstance(x, list):

            level_sum += x

        else:

            level_sum += _process_nested_list_element_recursive(x, level + 1)

    return level_sum * level


def product_sum(array):
    """
    @details "Special" arrays inside it are summed themselves and then
    multiplied by their level of depth.
    """

    return _process_nested_list_element_recursive(array, 1)

def _binary_search_helper(array, target, l, r):
    if (l > r):
        return -1

    # TODO: try calculating with r - l // 2 for midpoint index
    midpoint_index = (l + r ) // 2
    midpoint_value = array[midpoint_index]

    if target == midpoint_value:
        return midpoint_index

    if (target < midpoint_value):
        return _binary_search_helper(array, target, l, midpoint_index - 1)

    assert target > midpoint_value

    return _binary_search_helper(array, target, midpoint_index + 1, r)

def binary_search(array, target):
    N = len(array)

    return _binary_search_helper(array, target, 0, N - 1)


def find_three_largest_numbers(array):

    assert len(array) >= 3

    largest_three = [array[0], array[1], array[2]]
    largest_three.sort()

    for x in array[3:]:
        if (x > largest_three[0]):
            largest_three[0] = x
            largest_three.sort()

    return largest_three


def is_palindrome(input_string):

    N = len(input_string)
    l_ptr = 0
    r_ptr = N - 1

    result = True

    for i in range(N // 2):

        if input_string[l_ptr] != input_string[r_ptr]:

            result = False

            break

        l_ptr += 1
        r_ptr -= 1

    return result

def caesar_cipher_encryptor(input_s, key):
    """
    @brief Given non-empty string of lowercase letters and non-negative integer
    representing a key, return new string obtained by shifting every letter in
    input string by k positions in the alphabet, where k is the key.
    """
    lower_case_alphabet = "abcdefghijklmnopqrstuvwxyz"
    N_alphabet = len(lower_case_alphabet)
    letter_to_index = dict([
        (letter, index)
        for index, letter in enumerate(list(lower_case_alphabet))])

    new_string = []

    for index in range(len(input_s)):

        letter = input_s[index]

        letter_index = letter_to_index[letter]

        new_letter_index = (letter_index + key) % N_alphabet

        # str does not support item assignment.
        new_string.append(lower_case_alphabet[new_letter_index])

    return "".join(new_string)


def _divide_by_10(current_count, current_character):
    how_many_nines = current_count // 9
    remainder_dividing_by_9 = current_count % 9
    output = []
    for i in range(how_many_nines):
        output.append('9')
        output.append(current_character)
    if remainder_dividing_by_9 != 0:
        output.append(str(remainder_dividing_by_9))
        output.append(current_character)

    return "".join(output)


def run_length_encoding(input_s):
    """
    @details
    Takes in non-empty string.

    Input string can contain all sorts of special characters, including numbers.
    """
    N = len(input_s)
    assert N > 0
    if (N == 1):
        return "1" + input_s

    output = []

    current_character = input_s[0]
    current_count = 1

    for i in range(1, N - 1):

        if input_s[i] != current_character:
            encoded_current_character = _divide_by_10(
                current_count,
                current_character)
            output.append(encoded_current_character)
            current_character = input_s[i]
            current_count = 1

        else:
            current_count += 1

    if input_s[N - 1] != current_character:
        encoded_current_character = _divide_by_10(
            current_count,
            current_character)
        output.append(encoded_current_character)
        output.append('1')
        output.append(str(input_s[N-1]))
    else:
        encoded_current_character = _divide_by_10(
            current_count + 1,
            current_character)
        output.append(encoded_current_character)

    return "".join(output)


def generate_document(characters, document):

    if len(characters) < len(document):
        # Frequency of unique characters in the characters string must be
        # greater than or equal to frequency of unique characters in the
        # document string.

        return False

    if len(document) == 0:
        return True

    character_frequency_map = dict([])
    for character in list(characters):
        if character not in character_frequency_map:
            character_frequency_map[character] = 1
        else:
            character_frequency_map[character] += 1

    for letter in list(document):

        if letter not in character_frequency_map:
            return False

        elif character_frequency_map[letter] <= 0:
            return False
        else:
            character_frequency_map[letter] -= 1

    return True

def first_non_repeating_character_with_ordered_dict(input_s):
    character_frequency_map = OrderedDict()

    for letter in list(input_s):

        if letter not in character_frequency_map:

            character_frequency_map[letter] = 1

        else:

            character_frequency_map[letter] += 1

    for character in character_frequency_map.keys():

        if character_frequency_map[character] == 1:

            return input_s.find(character)

    return -1