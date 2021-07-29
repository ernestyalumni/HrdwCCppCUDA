from collections import deque

################################################################################
# @details
#
# DataStructures : deque, Queue, Binary Tree
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





    