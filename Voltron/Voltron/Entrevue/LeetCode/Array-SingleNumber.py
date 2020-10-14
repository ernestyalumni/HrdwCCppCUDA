"""
@file Array-SingleNumber.py

Log:
2020/10/13
8:31
9:07 working implementation
"""
example_input_1 = [2, 2, 1]
example_input_2 = [4, 1, 2, 1, 2]
example_input_3 = [1]

example_output_1 = 1
example_output_2 = 4
example_output_3 = 1

"""
Given a non-empty array of integers nums, every element appears twice except
for one.

Let N be total size of the array.
For 0, 1, ... N - 1 (finite set), there exists I in 0, 1, ... N -1 such that
a_I != a_i for all i != I.

If i != I, there exists unique j != I, j != i such that a_i = a_j and
a_k != a_i for all k != i, j


For all i in 0, 1, ... N - 1, either
a. exists unique j != i such that a_i = a_j and for all k != j, a_k != a_j, or
b. a_i != a_j for all j != i.

For any i, j in 0, 1, ... N - 1, i != j, either
a. a_i != a_j so either
    A. a_i or a_j is unique or
    B. a_i and a_j "have other matching pairs"
b. a_i == a_j and for all k != i, k != j, a_k != a_i
"""

# This is the base case.
def find_single_number_from_3(nums3):
    """
    Inputs
    nums3 - assumed to have length 3.
    """
    top_element = nums3.pop()
    if top_element in nums3:
        if top_element == nums3[0]:
            return nums3[1]
        else:
            return nums3[0]
    else:
        return top_element

def check_pair_from_3_nums(nums, traversed_numbers):
    """
    Suppose unique 1 is in nums + traversed_numberes.

    For nums, either
    a. all 3 have pairs in traversed_numbers, so traversed_numbers is of size 4
    b. nums contain unique number. Either
        A. traversed_numbers of size 2 so 2 numbers in nums has pairs in there
        B. traversed_numbers of size 0 and so a pair is in nums.
    """
    if len(traversed_numbers) == 4:
        for num in nums:
            traversed_numbers.remove(num)
        return traversed_numbers[0]

    if len(traversed_numbers) == 2:
        for num in nums:
            if num not in traversed_numbers:
                return num

    if len(traversed_numbers) == 0:
        return find_single_number_from_3(nums)


def check_pair(nums, traversed_numbers):
    if len(nums) == 3:
        return check_pair_from_3_nums(nums, traversed_numbers)

    pair = []
    pair.append(nums.pop())
    pair.append(nums.pop())

    if (pair[0] == pair[1]):
        return check_pair(nums, traversed_numbers)

    if pair[0] in traversed_numbers:
        traversed_numbers.remove(pair[0])
    else:
        traversed_numbers.append(pair[0])

    if pair[1] in traversed_numbers:
        traversed_numbers.remove(pair[1])
    else:
        traversed_numbers.append(pair[1])

    return check_pair(nums, traversed_numbers)


def find_single_number(nums):

    if len(nums) == 1:
        return nums[0]

    if len(nums) == 3:
        return find_single_number_from_3(nums)

    traversed_numbers = []

    return check_pair(nums, traversed_numbers)


class Solution:
    def singleNumber(self, nums) -> int:
        return 0

if __name__ == "__main__":

    print("\nArray-SingleNumber\n")