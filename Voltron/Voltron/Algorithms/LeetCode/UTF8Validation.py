"""
@name 383. UTF-8 Validation
@ref https://leetcode.com/problems/utf-8-validation/
"""

def valid_utf8(data: list[int]) -> bool:
    """
    @ref https://leetcode.com/problems/utf-8-validation/solution/
    @details "As can be seen from the first example, the array can contain data
    for multiple characters all of which can be valid UTF-8 characters and hence
    the charset represented by the array is valid.
    """

    i = 0
    while (i < len(data)):
        result = check_segment(data, i)
        if not result[0]:
            return False
        else:
            i = result[1]
    return True

def check_segment(array, i):
    try:
        # Check 1-byte case
        if ((array[i] >> 7) & 1) == 0:
            return True, i + 1

        if ((array[i] >> 6) & 1) == 1:
            # Check 2-byte case
            if ((array[i] >> 5) & 1) == 0 and \
                check_most_significant_2_bits_of_byte(array[i + 1]):
                return True, i + 2
            if ((array[i] >> 5) & 1) == 1:
                # Check 3-byte case
                if ((array[i] >> 4) & 1) == 0 and \
                    check_most_significant_2_bits_of_byte(array[i + 1]) and \
                        check_most_significant_2_bits_of_byte(array[i + 2]):
                    return True, i + 3
                if ((array[i] >> 4) & 1) == 1:
                    # Check 4-byte case
                    if ((array[i] >> 3) & 1) == 0 and \
                        check_most_significant_2_bits_of_byte(array[i + 1]) and \
                            check_most_significant_2_bits_of_byte(array[i + 2]) and \
                                check_most_significant_2_bits_of_byte(array[i + 3]):
                        return True, i + 4
        return False, i
    except IndexError:
        return False, i


def check_most_significant_2_bits_of_byte(
        value,
        most_significant = 1,
        second_most_significant = 0):
    most_significant_result = (((value >> 7) & 1) == most_significant)
    second_most_significant_result = (((value >> 6) & 1) ==
        second_most_significant)
    return most_significant_result and second_most_significant_result


def int_to_bitfield(n):
    """
    @ref https://stackoverflow.com/questions/10321978/integer-to-bitfield-as-a-list
    @ref https://docs.python.org/3/library/functions.html#
    @details bin(x) - Convert an integer number to a binary string prefixed with
    "0b". the result is a valid Python expression. 
    """
    return [int(digit) for digit in bin(n)[2:]] # [2:] to chop off the "0b" part