class ReduceNumberToZero:
    
    @classmethod
    def number_of_steps(cls, num: int) -> int:
        """
        @name 1342. Number of Steps to Reduce a Number to Zero
        @ref Leetcode, 1342. 
        https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero/
        
        @brief Given an integer num, return the number of steps to reduce it to
        zero.
        @details

        Runtime: 51 ms
        08/10/2021 09:06    Accepted    32 ms
        """
        number_of_steps = 0

        while (num > 0):
            # O(log_2(N)) time.

            is_even = num % 2 == 0

            if is_even:

                number_of_steps += 1

                num = num / 2

            else:

                number_of_steps += 1

                num -= 1

        return number_of_steps


def one_line(num: int) -> int:
    """
    @ref https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero/discuss/592198/Python-100-100-1-liner
    """

    # Each 0 in bin(num) represents a place to divide by 2, i.e. to bit shift to
    # right by 1 (assume they are 0s before nonzero most significant bit). Each
    # 1 needs to be subtracted by 1 and divided by 2 immediately. For case of 1
    # it works since bin will stick a "0b" on the front.
    return bin(num).count('1') * 2 + bin(num).count('0') - 2