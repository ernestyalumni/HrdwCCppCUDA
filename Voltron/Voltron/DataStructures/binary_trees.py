# @name binary_trees.py
# @ref https://classroom.udacity.com/courses/ud513/lessons/7122604912/concepts/79211181170923
# @Details Quiz: Binary Tree Practice, Udacity

from collections import deque

class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class _BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)

class BinaryTree(_BinaryTree):
    def __init__(self, root):
        super().__init__(root)

    @staticmethod
    def _get_all_nodes_in_next_level(previous_level_deque):

        new_deque = deque()
        while (len(previous_level_deque)):
            v = previous_level_deque.pop()
            if v.left:
                new_deque.appendleft(v.left)
            if v.right:
                new_deque.appendleft(v.right)
        return new_deque

    def search(self, find_val):
        """Return True if the value
        is in the tree, return
        False otherwise.
        @details Breadth First Search
        """

        d = deque()
        if (self.root.value == find_val):
            return True
        d.appendleft(self.root)

        while (len(d) > 0):
            new_deque = BinaryTree._get_all_nodes_in_next_level(d)

            found_result = \
                any([(x.value == find_val) for x in list(new_deque)])

            if found_result:
                return True

            d = new_deque

        return False

    @staticmethod
    def _iterative_preorder_traversal(node, stack, results):
        if not node:
            return stack, results

        results.append(node.value)

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

        return stack, results

    @staticmethod
    def preorder_traversal(node):
        stack = []
        results = []
        stack, results = \
            BinaryTree._iterative_preorder_traversal(node, stack, results)

        while (len(stack) > 0):
            node = stack.pop()
            stack, results = \
                BinaryTree._iterative_preorder_traversal(node, stack, results)

        return results

    def print_tree(self):
        """Print out all tree nodes
        as they are visited in
        a pre-order traversal."""
        result = BinaryTree.preorder_traversal(self.root)
        return_string = ""
        for element in result[:-1]:
            return_string += str(element) + "-"
        return_string += str(result[-1])

        return return_string

    def preorder_search(self, start, find_val):
        """Helper method - use this to create a 
        recursive search solution."""
        stack = []
        results = []
        node = start
        if (node.value == find_val):
            return True, node

        stack, results = \
            BinaryTree._iterative_preorder_traversal(node, stack, results)

        while (len(stack) > 0):
            node = stack.pop()
            if (node.value == find_val):
                return True, node

            stack, results = \
                BinaryTree._iterative_preorder_traversal(node, stack, results)

        return False, None

    def preorder_print(self, start, traversal):
        """Helper method - use this to create a 
        recursive print solution."""
        return traversal

class BST(object):
    def __init__(self, root):
        self.root = Node(root)

    @staticmethod
    def _compare_to_node(node, target_value):
        if (node.value == target_value):
            return True, node

        if (node.value < target_value):
            return False, node.right

        if (node.value > target_value):
            return False, node.left

    @staticmethod
    def _insert_iteration(node, new_value):
        is_equal, next_node = BST._compare_to_node(node, new_value)
        if is_equal:
            if not node.left:
                node.left = Node(new_value)
            else:
                new_node = Node(new_val)
                new_node.left = node.left
                node.left = new_node
            return True, node

        if not next_node:
            if (node.value < new_value):
                node.right = Node(new_value)
            if (node.value > new_value):
                node.left = Node(new_value)
            return True, node

        return False, next_node


    def insert(self, new_val):

        inserted = False

        node = self.root

        while(True):

            is_inserted, next_node = BST._insert_iteration(node, new_val)

            if is_inserted:
                return

            node = next_node

        pass

    def search(self, find_val):

        found = False

        node = self.root

        while (not found):
            is_equal, next_node = BST._compare_to_node(node, find_val)

            if is_equal:

                return True

            if not next_node:
                return False
    
            node = next_node