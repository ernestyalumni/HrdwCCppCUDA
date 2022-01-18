def insertion_sort(array):
    
    N = len(array)

    # Outer for-loop, such that for each iteration, k elements from 0, ... k
    # are to be assumed to be sorted or about to be sorted.    
    for k in range(1, N):

        # k, k - 1, ... 1
        for j in range(k, 0, -1):

            if (array[j - 1] > array[j]):

                array[j - 1], array[j] = array[j], array[j - 1]

            else:
                # As soon as we don't need to swap, the (k + 1)st in correct
                # location. It's because of induction case that we can do this.
                break

    return array


def insertion_sort_optimized(array):

    N = len(array)

    for k in range(1, N):

        temp = array[k]

        for j in range(k, 0, -1):

            if (array[j - 1] > temp):

                # Shift or move value at j - 1 into position j.
                array[j] = array[j - 1]
            else:

              array[j] = temp
              break

        if (array[0] > temp):
            # Only executed if temp < array[0]
            array[0] = temp

    return array