# @name binary_trees_main_driver.py
"""
@ref https://classroom.udacity.com/courses/ud513/lessons/7122604912/concepts/79211181170923
"""
from binary_trees import (Node, _BinaryTree, BinaryTree, BST)

# Set up tree
test_tree = _BinaryTree(1)
test_tree.root.left = Node(2)
test_tree.root.right = Node(3)
test_tree.root.left.left = Node(4)
test_tree.root.left.right = Node(5)

tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)

tree1 = BinaryTree('D')
tree1.root.left = Node('B')
tree1.root.right = Node('E')
tree1.root.left.left = Node('A')
tree1.root.left.right = Node('C')
tree1.root.right.right = Node('F')

# Test search
# Should be True
print(tree.search(4))
# Should be False
print(tree.search(6))

# Test print_tree
# Should be 1-2-4-5-3
print(tree.print_tree())

# Set up tree
BSTtree = BST(4)

# Insert elements
BSTtree.insert(2)
BSTtree.insert(1)
BSTtree.insert(3)
BSTtree.insert(5)

# Check search
# Should be True
print(BSTtree.search(4))
# Should be False
print(BSTtree.search(6))

# test steps
test_root = Node(4)
result1 = BST._insert_iteration(test_root, 2)
