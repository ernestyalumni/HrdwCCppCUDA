def selection_sort(array):
    
    N = len(array)

    for i in range(N):
        # Find the min element in the unsorted a[i .. N - 1]

        # Assume the min is the first element.
        minimum_element_index = i

        for j in range(i + 1, N):
            if (array[j] < array[minimum_element_index]):
                # Found a new minimum, remember its index.
                minimum_element_index = j

        if (minimum_element_index != i):
            # Move minimum element to the "front" or "left".
            array[i], array[minimum_element_index] = \
                array[minimum_element_index], array[i]

    return array

class SelectionSorting:

    # cf. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/mit6_006s20_lec3/
    # pp. 3, Lecture 3: Sorting

    # Create static method
    @staticmethod
    def mit_ocw_selection_sort(A, i = None):
        '''Sort A[:i + 1]'''
        if i is None:   # O(1)
            i = len(A) - 1  # O(1)

        # Base case: for i = 0, array has 1 element, so is sorted.

        # Induction: assume correct for i, last number of a sorted output is a
        # largest number of array, and algorithm puts one there; then A[i] is
        # sorted by induction.
        if i > 0:   # O(1)
            j = SelectionSorting.prefix_max(A, i)   # S(i)
            A[i], A[j] = A[j], A[i] # O(1)
            SelectionSorting.mit_ocw_selection_sort(A, i - 1)    # T(i - 1)

    @staticmethod
    def prefix_max(A, i):   # S(i)
        '''Return index of maximum in A[:i + 1]'''
        if i > 0:   # O(1)
            j = SelectionSorting.prefix_max(A , i - 1)   # S(i - 1)
            if A[i] < A[j]: # O(1)
                return j # O(1)

        # Base case: for i = 0, array has one element, so index of max is i
        return i # O(1)
