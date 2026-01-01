# https://blog.faangshui.com/p/before-leetcode

def generate_all_subsets_iterative(input_set):
    """
    7. Generate All Subsets of a Set
    Generate all possible subsets (the power set) of a set of numbers by
    including or excluding each element recursively.   
    https://blog.faangshui.com/i/149072585/recursion
    """
    power_set = []
    power_set.append([])

    for element in input_set:
        # For each element, we effectively double the number of subsets in the
        # power set by copying the existing subsets and adding the new element
        # to each of them, so that there's subsets with the element and subsets
        # without the element.
        current_subsets = list(power_set)
        for subset in current_subsets:
            power_set.append(subset + [element])
    return power_set

def generate_all_subsets_recursive(input_set):
    """
    7. Generate All Subsets of a Set
    Generate all possible subsets (the power set) of a set of numbers by
    including or excluding each element recursively.
    https://blog.faangshui.com/i/149072585/recursion
    """
    input_set = list(input_set)
    current_subset = []
    power_set = []
    if not input_set or len(input_set) == 0:
        power_set.append(current_subset)
        return power_set

    N = len(input_set)
    def generate_all_subsets_recursive_helper(
        input_set,
        i,
        N,
        current_subset,
        power_set):
        if i == N:
            # [:] creates a shallow copy; it creates a new list.
            power_set.append(current_subset[:])
            return

        generate_all_subsets_recursive_helper(
            input_set,
            i + 1,
            N,
            current_subset,
            power_set)
        current_subset.append(input_set[i])
        generate_all_subsets_recursive_helper(
            input_set,
            i + 1,
            N,
            current_subset,
            power_set)
        # Backtrack: remove current element to restore state.
        current_subset.pop()

    generate_all_subsets_recursive_helper(
        input_set,
        0,
        N,
        current_subset,
        power_set)
    return power_set