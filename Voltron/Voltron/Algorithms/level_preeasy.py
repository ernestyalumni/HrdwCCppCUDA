# https://blog.faangshui.com/p/before-leetcode

def generate_all_subsets_recursion_helper(
    input_set,
    current_subset,
    all_subsets):

    if not input_set or len(input_set) == 0:
        return

    

def generate_all_subsets(input_set):
    """
    7. Generate All Subsets of a Set
    Generate all possible subsets (the power set) of a set of numbers by
    including or excluding each element recursively.
    https://blog.faangshui.com/i/149072585/recursion
    """
    return

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