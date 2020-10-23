
//------------------------------------------------------------------------------
/// \file BinaryTrees.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Tree and traversal.
/// \details 
///
/// \ref https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
/// \ref LeetCode, Binary Trees
//-----------------------------------------------------------------------------

namespace DataStructures
{

namespace BinaryTrees
{

//-----------------------------------------------------------------------------
// cf. https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
//-----------------------------------------------------------------------------

TreeNode::TreeNode():
  value_{0},
  left_{nullptr},
  right_{nullptr}
{}

TreeNode::TreeNode(int x):
  value_{x},
  left_{nullptr},
  right_{nullptr}
{}

TreeNode::TreeNode(int x, TreeNode* left, TreeNode* right):
  value_{x},
  left_{left},
  right_{right}
{}

} // namespace BinaryTrees
} // namespace DataStructures
