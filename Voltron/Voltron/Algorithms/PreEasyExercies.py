class ArrayIndexing:
    """
    https://blog.faangshui.com/p/before-leetcode
    1. Array Indexing
    Understanding how to navigate arrays is essential. Here are ten exercises,
    sorted in increasing difficulty, that build upon each other:
    """
    @staticmethod
    def iterate_over_array(array, is_print=False):
        """
        Iterate over an array and return the result.
        """
        result = []
        for element in array:
            result.append(element)
            if is_print:
                print(element)
        return result