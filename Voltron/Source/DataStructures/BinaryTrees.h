//------------------------------------------------------------------------------
/// \file BinaryTrees.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Tree and traversal.
/// \details 
///
/// \ref https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
/// \ref LeetCode, Binary Trees
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_BINARY_TREES_H
#define DATA_STRUCTURES_BINARY_TREES_H

#include <vector>

namespace DataStructures
{

namespace BinaryTrees
{

//-----------------------------------------------------------------------------
/// cf. https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
///
/// Pre-order Traversal.
/// Visit the root first.
/// Then traverse left subtree.
/// Finally, traverse right subtree.
///
/// In-order Traversal.
/// 1. Traverse the left subtree first.
/// 2. Then visit the root.
/// 3. Finally, traverse the right subtree.
///
/// Post-order Traversal.
/// 1. Traverse left subtree first.
/// 2. Then the right subtree.
/// 3. Finally, visit the root.
//-----------------------------------------------------------------------------

struct TreeNode
{
  int value_;
  TreeNode* left_;
  TreeNode* right_;

  TreeNode();
  TreeNode(int x);
  TreeNode(int x, TreeNode* left, TreeNode* right);
};

std::vector<int> preorder_traversal(TreeNode* root);

std::vector<int> preorder_traversal_iterative(TreeNode* root);

std::vector<int> preorder_traversal_morris(TreeNode* root);

void preorder_traversal_recursive_step(
  TreeNode* node_ptr,
  std::vector<int>& result);

std::vector<int> preorder_traversal_recursive(TreeNode* root);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/binary-tree-inorder-traversal/solution/
/// \details Time complexity: O(N),
/// Space complexity: O(N)
//------------------------------------------------------------------------------
std::vector<int> inorder_traversal_iterative(TreeNode* root);

std::vector<int> inorder_traversal_recursive(TreeNode* root);

void inorder_traversal_recursive_step(
  TreeNode* node_ptr,
  std::vector<int>& result);

std::vector<int> postorder_traversal_recursive(TreeNode* root);

void postorder_traversal_recursive_step(
  TreeNode* node_ptr,
  std::vector<int>& result);

std::vector<int> postorder_traversal_iterative(TreeNode* root);

// cf. https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/930/discuss/45551/Preorder-Inorder-and-Postorder-Iteratively-Summarization

std::vector<int> preorder_traversal_iterative_simple(TreeNode* root);
std::vector<int> inorder_traversal_iterative_simple(TreeNode* root);
std::vector<int> postorder_traversal_iterative_simple(TreeNode* root);

//-----------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/990/
/// \details Level Order Traversal or Breadth First Traversal.
//-----------------------------------------------------------------------------
std::vector<std::vector<int>> level_order_traversal(TreeNode* root);

int max_depth(TreeNode* root);

int max_depth_recursive_step(TreeNode* root);

} // namespace BinaryTrees
} // namespace DataStructures

#endif // DATA_STRUCTURES_BINARY_TREES_H