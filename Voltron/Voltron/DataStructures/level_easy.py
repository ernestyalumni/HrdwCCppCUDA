from collections import deque

################################################################################
# @details
#
# DataStructures : deque, Queue, Binary Tree, Binary Search Tree (BST)
################################################################################

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def _make_disparate_input_nodes(input_list):
    result = []

    for element in input_list:
        result.append(BinaryTree(element))

    return result


def _get_next_level_children(previous_deque):
    """
    @details This will pop all elements out of input previous_deque
    """
    next_level_children = deque()

    while (previous_deque):
        node = previous_deque.pop()

        if node.left:
            # Add nodes to the "left" so it acts like a queue (to the "back of
            # the line").
            next_level_children.appendleft(node.left)

        if node.right:
            next_level_children.appendleft(node.right)

    return next_level_children


def node_depths(root):

    if root == None:
        return 0

    counter = 0
    level_counter = 0

    level_queue = deque()
    level_queue.append(root)

    while (level_queue):
        level_counter += 1

        level_queue = _get_next_level_children(level_queue)

        counter += level_counter * len(level_queue)

    return counter


################################################################################
# Find Closest Value in BST
################################################################################

def _get_closest_value(node, target):
    """
    Complexity: Average O(log(n)) time. But O(log(n)) space because of
    recursion.
    """
    if node.value_ == target:
        return node.value_

    # Base case. Node with no leaves.
    if (node.left_ == None and node.right_ == None):
        return node.value_

    if (node.left_ and node.right_ == None):
        left_result = _get_closest_value(node.left_, target)
        delta_value = abs(target - node.value_)
        delta_left_result = abs(target - left_result)

        return left_result if delta_left_result < delta_value else node.value_

    if (node.right_ and node.left_ == None):
        right_result = _get_closest_value(node.right_, target)
        delta_value = abs(target - node.value_)
        delta_right_result = abs(target - right_result)

        return right_result if delta_right_result < delta_value else node.value_

    if node.left_.value_ == target:
        return node.left_.value_

    if node.right_.value_ == target:
        return node.right_.value_

    if target < node.left_.value_:

        left_result = _get_closest_value(node.left_, target)
        delta_value = abs(target - node.value_)
        delta_left_result = abs(target - left_result)
        return left_result if delta_left_result < delta_value else node.value_

    if target > node.right_.value_:

        right_result = _get_closest_value(node.right_, target)
        delta_value = abs(target - node.value_)
        delta_right_result = abs(target - right_result)
        return right_result if delta_right_result < delta_value else node.value_

    assert node.left_.value_ < target and target < node.right_.value_

    left_result = _get_closest_value(node.left_, target)
    right_result = _get_closest_value(node.right_, target)

    delta_left_result = abs(target - left_result)
    delta_right_result = abs(right_result - target)
    delta_value = abs(node.value_ - target)

    results = {
        delta_left_result : left_result,
        delta_right_result : right_result,
        delta_value : node.value_}

    return results[min(results.keys())]


def find_closest_value_in_bst(tree, target):
    """
    @brief Write function that takes Binary Search Tree (BST) and target integer
    value and returns closest value to target value contained in BST.

    You can assume that there'll be only one closest value.
    """
    return _get_closest_value(tree, target)


def _get_closest_value_iterative(node, target, closest_value):
    current_node = node

    while (current_node is not None):

        if (current_node.value_ == target):
            return current_node.value_

        delta_node = abs(target - current_node.value_)
        delta_closest = abs(target - closest_value)

        closest_value = (closest_value
            if delta_closest <= delta_node else current_node.value_)

        if (current_node.value_ < target):
            current_node = current_node.right_
        else:
            assert target < current_node.value_

            current_node = current_node.left_

    return closest_value


def _preorder_traversal(node, running_sum, sums):

    if node is not None:

        if node.left_ is None and node.right_ is None:
            sums.append(running_sum + node.value_)
            return

        running_sum += node.value_

        _preorder_traversal(node.left_, running_sum, sums)
        _preorder_traversal(node.right_, running_sum, sums)



def branch_sums(root):

    running_sum = 0
    sums = []

    _preorder_traversal(root, running_sum, sums)

    return sums