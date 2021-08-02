################################################################################
## \ref https://en.wikipedia.org/wiki/Bubble_sort
################################################################################

def bubble_sort_naive(array):
    is_sorted = False

    N = len(array)

    while not is_sorted:
        is_sorted = True

        for i in range(1, N):

            # If this pair is out of order.

            if array[i - 1] > array[i]:
                array[i - 1], array[i] = array[i], array[i - 1]

                is_sorted = False

    return array


def bubble_sort_optimized(array):
    is_sorted = False

    N = len(array)

    while not is_sorted:
        is_sorted = True

        for i in range(1, N):

            # If this pair is out of order.

            if array[i - 1] > array[i]:
                array[i - 1], array[i] = array[i], array[i - 1]

                is_sorted = False

        # Observe that the Nth pass finds the Nth largest element and puts it
        # into its final place.
        N = N - 1

    return array


def bubble_sort_no_extra_swaps(array):

    N = len(array)

    while N > 1:

        new_N = 0

        for i in range(1, N):

            # If this pair is out of order.

            if array[i - 1] > array[i]:
                array[i - 1], array[i] = array[i], array[i - 1]

                # We know that in the next pass, we will need to check and swap
                # up to this element.
                new_N = i

        # Observe that the Nth pass finds the Nth largest element and puts it
        # into its final place.
        N = new_N

    return array
