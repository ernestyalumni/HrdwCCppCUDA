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

#include <deque>
#include <sstream>
#include <string>
#include <utility> // std::make_pair, std::pair
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

// This is also similar to depth-first search.
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

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/990/
/// \details Level Order Traversal or Breadth First Traversal.
//------------------------------------------------------------------------------
std::vector<std::vector<int>> level_order_traversal(TreeNode* root);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/interview/card/top-interview-questions-medium/108/trees-and-graphs/787/
/// \brief Binary Tree Zigzag Level Order Traversal.
/// \details Medium Interview problem.
//------------------------------------------------------------------------------
std::vector<std::vector<int>> zigzag_level_order_traversal(TreeNode* root);

int max_depth(TreeNode* root);

int max_depth_recursive_step(TreeNode* root);

bool is_balanced(TreeNode* root);

//------------------------------------------------------------------------------
/// Runtime: 16 ms, faster than 82.52% of C++ online submissions for Balanced
/// Binary Tree.
/// Memory Usage: 22.7 MB, less than 5.70% of C++ online submissions for
/// Balanced Binary Tree.
//------------------------------------------------------------------------------
std::pair<bool, int> balance_max_height_recursive(TreeNode* root);

//------------------------------------------------------------------------------
/// Binary Tree - Interview Questions and Practice Problems, Medium
/// \url https://medium.com/techie-delight/binary-tree-interview-questions-and-practice-problems-439df7e5ea1f
/// Check if two given binary trees are identical or not
/// \url https://www.techiedelight.com/check-if-two-binary-trees-are-identical-not-iterative-recursive/
/// \url https://www.glassdoor.com/Interview/binary-trees-interview-questions-SRCH_KT0,12.htm
//------------------------------------------------------------------------------
bool is_same_recursive(TreeNode* root1, TreeNode* root2);

//------------------------------------------------------------------------------
/// \brief Symmetric Tree
/// \brief Given a binary tree, check whether it is a mirror of itself (i.e.
/// symmetric around its center).
/// \url https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/536/
//------------------------------------------------------------------------------

bool is_symmetric(TreeNode* root);

bool is_level_symmetric(std::deque<TreeNode*>);

//------------------------------------------------------------------------------
/// \brief Serialization is the process of converting a data structure or object
/// \ref https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
//------------------------------------------------------------------------------
std::string serialize(TreeNode* root);

//------------------------------------------------------------------------------
/// \ref https://leetcode.com/problems/serialize-and-deserialize-binary-tree/discuss/74259/Recursive-preorder-Python-and-C%2B%2B-O(n)
//------------------------------------------------------------------------------
//void serialize(TreeNode* root, std::ostringstream& )

/// \ref https://leetcode.com/problems/serialize-and-deserialize-binary-tree/discuss/74252/Clean-C%2B%2B-solution
int parse_node_value(std::string& data);

} // namespace BinaryTrees
} // namespace DataStructures

#endif // DATA_STRUCTURES_BINARY_TREES_H