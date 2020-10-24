//------------------------------------------------------------------------------
/// \file BinaryTrees.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Tree and traversal.
/// \details 
///
/// \ref https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
/// \ref LeetCode, Binary Trees
//-----------------------------------------------------------------------------
#include "BinaryTrees.h"

#include <algorithm> // std::max
#include <queue>
#include <stack>
#include <vector>

using std::max;
using std::queue;
using std::stack;
using std::vector;

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

vector<int> preorder_traversal(TreeNode* root)
{
  stack<TreeNode*> tree_node_ptr_stack;
  vector<int> result;

  if (root == nullptr)
  {
    return result;
  }

  // root then points to a node.
  TreeNode* current_ptr {root};
  tree_node_ptr_stack.push(root);

  // Do the first step.

  result.emplace_back(current_ptr->value_);

  // Move to the left node.
  // If the node was a leaf, then go to the next step.
  current_ptr = current_ptr->left_;

  while (!tree_node_ptr_stack.empty() || current_ptr != nullptr)
  {
    while (current_ptr != nullptr)
    {
      result.emplace_back(current_ptr->value_);

      tree_node_ptr_stack.push(current_ptr);

      current_ptr = current_ptr->left_;
    }

    current_ptr = tree_node_ptr_stack.top();
    tree_node_ptr_stack.pop();

    current_ptr = current_ptr->right_;
  } // End of while loop, stack is empty.

  return result;
}

// cf. https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/928/discuss/45267/3-Iterative-Solutions:-Stack-And-Morris-Traversal-(Complexity-Explained)
vector<int> preorder_traversal_iterative(TreeNode* root)
{
  vector<int> result;

  if  (root == nullptr)
  {
    return result;
  }

  stack<TreeNode*> tree_node_ptr_stack;
  tree_node_ptr_stack.push(root);

  while (!tree_node_ptr_stack.empty())
  {
    TreeNode* current_ptr = tree_node_ptr_stack.top();
    tree_node_ptr_stack.pop();

    // Effectively visit the "root" node first.
    result.emplace_back(current_ptr->value_);

    // Push right before left so that, due to property of a stack, left is
    // processed before right.
    if (current_ptr->right_ != nullptr)
    {
      tree_node_ptr_stack.push(current_ptr->right_);
    }
    // We've put the right subtree before the stack (so it's "below") so that
    // the left subtree will be popped off in the next iteration.
    if (current_ptr->left_ != nullptr)
    {
      tree_node_ptr_stack.push(current_ptr->left_);
    }
  }

  return result;
}

// cf. https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/928/discuss/45267/3-Iterative-Solutions:-Stack-And-Morris-Traversal-(Complexity-Explained)
vector<int> preorder_traversal_morris(TreeNode* root)
{
  vector<int> result;

  if (root == nullptr)
  {
    return result;
  }

  TreeNode* current_ptr {root};

  while (current_ptr != nullptr)
  {
    if (current_ptr->left_ == nullptr)
    {
      result.emplace_back(current_ptr->value_);
      current_ptr = current_ptr->right_;
    }
    else
    {
      TreeNode* node_ptr {current_ptr->left_};

      while (node_ptr->right_ != nullptr && node_ptr->right_ != current_ptr)
      {
        node_ptr = node_ptr->right_;
      }

      if (node_ptr->right_ == nullptr)
      {
        result.emplace_back(current_ptr->value_);
        node_ptr->right_ = current_ptr;
        current_ptr = current_ptr->left_;
      }
      else
      {
        node_ptr->right_ = nullptr;
        current_ptr = current_ptr->right_;
      }
    }
  }

  return result;
}

// https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/928/discuss/45466/C++-Iterative-Recursive-and-Morris-Traversal
void preorder_traversal_recursive_step(TreeNode* node_ptr, vector<int>& result)
{
  if (node_ptr == nullptr)
  {
    return;
  }

  // Otherwise, node_ptr actually points to a node with a value. Visit it.
  result.emplace_back(node_ptr->value_);

  preorder_traversal_recursive_step(node_ptr->left_, result);
  preorder_traversal_recursive_step(node_ptr->right_, result);
}

vector<int> preorder_traversal_recursive(TreeNode* root)
{
  vector<int> result;

  preorder_traversal_recursive_step(root, result);

  return result;
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/binary-tree-inorder-traversal/solution/
/// \details Time complexity: O(N),
/// Space complexity: O(N)
//------------------------------------------------------------------------------
vector<int> inorder_traversal_iterative(TreeNode* root)
{
  vector<int> result;

  if (root == nullptr)
  {
    return result;
  }

  stack<TreeNode*> tree_node_ptr_stack;
  TreeNode* current_ptr {root};

  while (!tree_node_ptr_stack.empty() || current_ptr != nullptr)
  {
    // Traverse the left subtree first. Until a leaf is reached.
    while (current_ptr != nullptr)
    {
      // Save the "root."
      tree_node_ptr_stack.push(current_ptr);

      current_ptr = current_ptr->left_;
    }
    // Exited because a leaf was found and current_ptr == nullptr now. Go back
    // to the stack for the previous "root".
    current_ptr = tree_node_ptr_stack.top();
    tree_node_ptr_stack.pop();

    result.emplace_back(current_ptr->value_);

    // Traverse the right subtree.
    current_ptr = current_ptr->right_;
  }

  return result;
}

vector<int> inorder_traversal_recursive(TreeNode* root)
{
  vector<int> result;

  inorder_traversal_recursive_step(root, result);

  return result;
}


void inorder_traversal_recursive_step(TreeNode* node_ptr, vector<int>& result)
{
  // Base case. If there's nothing to visit, exit the recursion.
  if (node_ptr == nullptr)
  {
    return;
  }

  // Traverse the left subtree first.
  inorder_traversal_recursive_step(node_ptr->left_, result);

  // Come back here, after traversing the left subtree and visit the "root".
  // We know that node_ptr is nonempty from above.
  result.emplace_back(node_ptr->value_);

  // Traverse the right subtree.
  inorder_traversal_recursive_step(node_ptr->right_, result);
}

vector<int> postorder_traversal_recursive(TreeNode* root)
{
  vector<int> result;

  postorder_traversal_recursive_step(root, result);

  return result;
}

void postorder_traversal_recursive_step(TreeNode* node_ptr, vector<int>& result)
{
  // Base case. If there's nothing to visit, exit the recursion.
  if (node_ptr == nullptr)
  {
    return;
  }

  // Traverse the left subtree first.
  postorder_traversal_recursive_step(node_ptr->left_, result);

  // Traverse the right subtree.
  postorder_traversal_recursive_step(node_ptr->right_, result);

  // Come back here.
  // We know that node_ptr is nonempty from above.
  result.emplace_back(node_ptr->value_);
}

vector<int> postorder_traversal_iterative(TreeNode* root)
{
  vector<int> result;

  if (root == nullptr)
  {
    return result;
  }

  stack<TreeNode*> tree_node_ptr_stack;
  TreeNode* current_ptr {root};
  TreeNode* previously_traversed {nullptr};

  while (!tree_node_ptr_stack.empty() || current_ptr != nullptr)
  {
    // Traverse left subtree first. Until a leaf is reached.
    while (current_ptr != nullptr)
    {

      // Push all left nodes into stack until it hits a leaf.
      tree_node_ptr_stack.push(current_ptr);
      current_ptr = current_ptr->left_;

      /*
      // Push the root's right child and then root to stack. Because of the
      // property of a stack, we'll have the root to pop off if we hit a left
      // leaf. Next, we'll have the right child, if it's "there" to traverse
      // prior to traversing the root.
      if (current_ptr->right_ != nullptr)
      {
        tree_node_ptr_stack.push(current_ptr->right_);
      }

      // Save the "root".
      tree_node_ptr_stack.push(current_ptr);

      // Set to the current_ptr or the current "root" to left child.
      current_ptr = current_ptr->left_;
      */
    }
    // Exited because a leaf was found and current_ptr == nullptr now. Go back
    // to the stack for the previous "root."
    // Had saved in stack all previous nodes along the "right" side of the
    // right subtree, for 2 * h = log(N) = height number of nodes. Then pop off
    // each node as new "roots" "up" the tree.
    current_ptr = tree_node_ptr_stack.top();

    // If popped item has a right child and right child is not processed yet,
    // then make sure right child is processed before root.
    if (current_ptr->right_ != nullptr &&
      current_ptr->right_ != previously_traversed)
    {
      current_ptr = current_ptr->right_;
    }
    else
    {
      result.emplace_back(current_ptr->value_);

      tree_node_ptr_stack.pop();
      previously_traversed = current_ptr;

      // Force checking the stack for other elements in the next iteration.
      current_ptr = nullptr;
    }
  }

    /*
      // Remove right child from stack.
      tree_node_ptr_stack.pop();

      // Push root back to stack.
      tree_node_ptr_stack.push(current_ptr);

      current_ptr = current_ptr->right_;
    }
    // We've completed traversing the right subtree.
    else
    {
      result.emplace_back(current_ptr->value_);

    }
  }
  */

  return result;
}

vector<int> preorder_traversal_iterative_simple(TreeNode* root)
{
  vector<int> result;

  if (root == nullptr)
  {
    return result;
  }

  stack<TreeNode*> tree_node_ptr_stack;
  TreeNode* current_ptr {root};  

  while (!tree_node_ptr_stack.empty() || current_ptr != nullptr)
  {
    if (current_ptr != nullptr)
    {
      // Push all left nodes into stack until it hits a leaf.
      tree_node_ptr_stack.push(current_ptr);
      // Visit the node first.
      result.emplace_back(current_ptr->value_);
      current_ptr = current_ptr->left_;
    }
    else
    {
      current_ptr = tree_node_ptr_stack.top();
      tree_node_ptr_stack.pop();

      current_ptr = current_ptr->right_;
    }
  }
  return result;
}

vector<int> inorder_traversal_iterative_simple(TreeNode* root)
{
  vector<int> result;

  if (root == nullptr)
  {
    return result;
  }

  stack<TreeNode*> tree_node_ptr_stack;
  TreeNode* current_ptr {root};  

  while (!tree_node_ptr_stack.empty() || current_ptr != nullptr)
  {
    if (current_ptr != nullptr)
    {
      // Push all left nodes into stack until it hits a leaf.
      tree_node_ptr_stack.push(current_ptr);
      current_ptr = current_ptr->left_;
    }
    else
    {
      current_ptr = tree_node_ptr_stack.top();
      tree_node_ptr_stack.pop();

      result.emplace_back(current_ptr->value_);

      current_ptr = current_ptr->right_;      
    }
  }
  return result;
}

vector<int> postorder_traversal_iterative_simple(TreeNode* root)
{
  vector<int> result;

  if (root == nullptr)
  {
    return result;
  }

  stack<TreeNode*> tree_node_ptr_stack;
  tree_node_ptr_stack.push(root);
  TreeNode* current_ptr {root};  
  TreeNode* previous_ptr {nullptr};

  while (!tree_node_ptr_stack.empty())
  {
    current_ptr = tree_node_ptr_stack.top();

    // If we are at a leaf, or we've already traversed the right node,
    if ((current_ptr->left_ == nullptr && current_ptr->right_ == nullptr) ||
      previous_ptr == current_ptr->right_ ||
        previous_ptr == current_ptr->left_)
    {
      // Deal with topological dependency.
      result.emplace_back(current_ptr->value_);

      tree_node_ptr_stack.pop();

      previous_ptr = current_ptr;
    }
    else
    {
      if (current_ptr->right_ != nullptr)
      {
        tree_node_ptr_stack.push(current_ptr->right_);
      }

      if (current_ptr->left_ != nullptr)
      {
        tree_node_ptr_stack.push(current_ptr->left_);
      }
    }
  }
  return result;
}

vector<vector<int>> level_order_traversal(TreeNode* root)
{
  vector<vector<int>> result;

  if (root == nullptr)
  {
    return result;
  }

  queue<TreeNode*> node_ptr_queue;
  node_ptr_queue.push(root);

  while (!node_ptr_queue.empty())
  {
    // TODO: Or use queue<TreeNode*>.size() to get the queue size before adding
    // nodes from another level.

    // Last node in the level
    TreeNode* level_end_node {node_ptr_queue.back()};

    TreeNode* current_ptr {node_ptr_queue.front()};

    vector<int> level_results;

    while (current_ptr != level_end_node)
    {
      level_results.emplace_back(current_ptr->value_);

      if (current_ptr->left_ != nullptr)
      {
        node_ptr_queue.push(current_ptr->left_);
      }

      if (current_ptr->right_ != nullptr)
      {
        node_ptr_queue.push(current_ptr->right_);
      }

      node_ptr_queue.pop();
      current_ptr = node_ptr_queue.front();
    }

    // Very last node in this level before the next level.
    if (current_ptr->left_ != nullptr)
    {
      node_ptr_queue.push(current_ptr->left_);
    }

    if (current_ptr->right_ != nullptr)
    {
      node_ptr_queue.push(current_ptr->right_);
    }

    level_results.emplace_back(current_ptr->value_);

    result.emplace_back(level_results);

    node_ptr_queue.pop();
  }

  return result;
}

int max_depth(TreeNode* root)
{
  return max_depth_recursive_step(root);
}

int max_depth_recursive_step(TreeNode* root)
{
  TreeNode* current_ptr {root};

  if (current_ptr == nullptr)
  {
    return 0;
  }

  // If the root node is a leaf,
  if ((current_ptr->left_ == nullptr) && (current_ptr->right_ == nullptr))
  {
    return 1;
  }

  int left_depth {max_depth_recursive_step(current_ptr->left_)};
  int right_depth {max_depth_recursive_step(current_ptr->right_)};

  // Add 1 to account for the root node itself.
  return (max(left_depth, right_depth) + 1);
}


} // namespace BinaryTrees
} // namespace DataStructures
