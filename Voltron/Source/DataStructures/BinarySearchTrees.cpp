//------------------------------------------------------------------------------
/// \file BinarySearchTrees.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Search Tree
/// \details 
///
/// \ref https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/140/introduction-to-a-bst/997/
/// \ref LeetCode, Binary Search Tree
//-----------------------------------------------------------------------------
#include "BinarySearchTrees.h"

// LIFO (last-in, first-out) data structure.
#include <stack>

using std::stack;

namespace DataStructures
{

namespace BinarySearchTrees
{

//----------------------------------------------------------------------------
/// \brief Default constructor.
//----------------------------------------------------------------------------
TreeNode::TreeNode() :
  value_{0},
  left_{nullptr},
  right_{nullptr}
{}

TreeNode::TreeNode(int x) :
  value_{x},
  left_{nullptr},
  right_{nullptr}
{}

TreeNode::TreeNode(int x, TreeNode *left, TreeNode *right) :
  value_{x},
  left_{left},
  right_{right}
{}

bool iterative_validate_binary_search_tree(TreeNode* root)
{
  if (root == nullptr)
  {
    return true;
  }

  stack<TreeNode*> tree_node_stack;

  TreeNode* prior_node_ptr {nullptr};
  TreeNode* current_node_ptr {root};

  while (current_node_ptr != nullptr || !tree_node_stack.empty())
  {
    while (current_node_ptr != nullptr)
    {
      // Traverse all the way to the left and don't stop until first left child
      // or left leaf.
      tree_node_stack.push(current_node_ptr);
      current_node_ptr = current_node_ptr->left_;
    }

    // top - accesses the top element
    current_node_ptr = tree_node_stack.top();
    // pop - removes the top element.
    tree_node_stack.pop();

    if (prior_node_ptr &&
      prior_node_ptr->value_ >= current_node_ptr->value_)
    {
      return false;
    }

    prior_node_ptr = current_node_ptr;
    current_node_ptr = current_node_ptr->right_;
  }

  return true;
}

// Recursive version.
// cf. https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/140/introduction-to-a-bst/997/discuss/32141/C++-simple-recursive-solution
bool inorder_validate_binary_search_tree(
  TreeNode* left_node,
  TreeNode* parent_node,
  TreeNode* right_node)
{
  // Base case, gets pointed to a nullptr
  if (parent_node == nullptr)
  {
    return true;
  }

  if ((left_node && left_node->value_ >= parent_node->value_) ||
    (right_node && parent_node->value_ >= right_node->value_))
  {
    return false;
  }

  // Do recursion on both the left and the right childs.
  return 
    inorder_validate_binary_search_tree(
      left_node,
      parent_node->left_,
      parent_node) &&
        inorder_validate_binary_search_tree(
          parent_node,
          parent_node->right_,
          right_node);
}
 
bool is_valid_binary_search_tree(TreeNode* root)
{
  return inorder_validate_binary_search_tree(nullptr, root, nullptr);
}

/*
BstIterator::BstIterator(TreeNode* root)
{}

int BstIterator::next()
{
  return 0;
}

bool BstIterator::has_next()
{
  return true;
}
*/

} // namespace BinarySearchTrees
} // namespace DataStructures

