"""
@name 383. UTF-8 Validation
@ref https://leetcode.com/problems/utf-8-validation/
"""

def valid_utf8(data: list[int]) -> bool:

    try:
        # Check 1-byte case
        if ((data[0] >> 7) & 1) == 0:
            if (len(data) > 1):
                return False
            else:
                return True
        
        if ((data[0] >> 6) & 1) == 1:
            # Check 2-byte case
            if ((data[0] >> 5) & 1) == 0 and \
                check_most_significant_2_bits_of_byte(data[1]):
                return True
            if ((data[0] >> 5) & 1) == 1:
                # Check 3-byte case
                if ((data[0] >> 4) & 1) == 0 and \
                    check_most_significant_2_bits_of_byte(data[1]) and \
                        check_most_significant_2_bits_of_byte(data[2]):
                    return True
                if ((data[0] >> 4) & 1) == 1:
                    # Check 4-byte case
                    if ((data[0] >> 3) & 1) == 0 and \
                        check_most_significant_2_bits_of_byte(data[1]) and \
                            check_most_significant_2_bits_of_byte(data[2]) and \
                                check_most_significant_2_bits_of_byte(data[3]):
                        if (len(data) > 4):
                            if check_most_significant_2_bits_of_bytes(data[4]):
                                return False
                            else:
                                return True
                        else:
                            return True
        else:
            return False

        return False

    except IndexError:
        return False

    return False

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