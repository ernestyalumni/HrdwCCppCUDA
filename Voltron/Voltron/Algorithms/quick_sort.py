# @name quick_sort.py

def _partition_from_last(a, left_index, right_index):
    pivot_value = a[right_index]

    j = right_index

    for i in range(right_index - 1, left_index - 1, -1):
        if a[i] >= pivot_value:
            j-=1
            a[i], a[j] = a[j], a[i]

    a[j], a[right_index] = a[right_index], a[j]

    return j

def _quick_sort_from_last(a, left_index, right_index):
    if (left_index < right_index + 1):
        new_pivot = _partition_from_last(a, left_index, right_index)

        _quick_sort_from_last(a, left_index, new_pivot - 1)
        _quick_sort_from_last(a, new_pivot + 1, right_index)

    return a

def quick_sort_from_last(a):
    return _quick_sort_from_last(a, 0, len(a) - 1)
