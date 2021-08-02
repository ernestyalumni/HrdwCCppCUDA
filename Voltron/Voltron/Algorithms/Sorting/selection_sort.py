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