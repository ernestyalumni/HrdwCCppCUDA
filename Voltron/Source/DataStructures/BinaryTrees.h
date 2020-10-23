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


} // namespace BinaryTrees
} // namespace DataStructures

#endif // DATA_STRUCTURES_BINARY_TREES_H