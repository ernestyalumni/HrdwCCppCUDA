# @name binary_search.py

def calculate_midpoint(l, r):
    """
    @param l left index, included in the range of elements to consider
    @param r right index, included in range of elements to consider
    @return None if there's no elements to range over.

    @details index = l, l+1, ... r are all included indices to consider.
    """
    # Get the total number of elements to consider.
    L  = r - l + 1
    if (L <= 0):
        return None

    return (L//2 + l) if (L % 2  == 1) else (L//2 - 1 + l)

def compare_and_partition(\
    midpoint_value, \
    search_value, \
    midpoint_index, \
    l,
    r):
    
    if (search_value == midpoint_value):
        return midpoint_index

    if (search_value < midpoint_value):
        return (l, midpoint_index - 1)

    if (search_value > midpoint_value):
        return (midpoint_index + 1, r)

def binary_search_iteration(a, l, r, search_value):
    m = calculate_midpoint(l, r)
    if not m:
        return None

    try:
        l, r = \
            compare_and_partition(\
                a[m], \
                search_value, \
                m, \
                l, \
                r)
        return l, r

    except TypeError:
        return m

def binary_search(a, search_value):
    """
    @name binary_search
    @param a array
    """
    N = len(a)
    l = 0
    r = len(a) - 1
    while(True):
        try:
            result = binary_search_iteration(a, l, r, search_value)
            l, r = result
        except TypeError:
            return -1 if not result else result

