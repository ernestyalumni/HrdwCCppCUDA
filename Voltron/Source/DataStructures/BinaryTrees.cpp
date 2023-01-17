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
#include <cmath> // std::abs
#include <deque>
#include <queue>
#include <stack>
#include <string> // std::stoi
#include <utility> // std::make_pair, std::move, std::pair
#include <vector>

using std::deque;
using std::make_pair;
using std::max;
using std::pair;
using std::queue;
using std::stack;
using std::stoi;
using std::string;
using std::to_string;
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

  if (root == nullptr)
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

      // Go left first.
      if (current_ptr->left_ != nullptr)
      {
        node_ptr_queue.push(current_ptr->left_);
      }

      // Then go right next.
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

vector<vector<int>> zigzag_level_order_traversal(TreeNode* root)
{
  vector<vector<int>> result;

  if (root == nullptr)
  {
    return result;
  }

  queue<TreeNode*> node_ptr_queue;
  node_ptr_queue.push(root);

  bool go_left_to_right {true};

  while (!node_ptr_queue.empty())
  {
    const int node_queue_size {static_cast<int>(node_ptr_queue.size())};
    vector<int> row (node_queue_size);

    for (int i {0}; i < node_queue_size; ++i)
    {
      TreeNode* current_ptr {node_ptr_queue.front()};
      node_ptr_queue.pop();

      // Find position to fill node's value
      int index {(go_left_to_right) ? i : (node_queue_size - 1 - i)};

      row[index] = current_ptr->value_;

      if (current_ptr->left_)
      {
        node_ptr_queue.push(current_ptr->left_);
      }

      if (current_ptr->right_)
      {
        node_ptr_queue.push(current_ptr->right_);
      }
    }

    // After this level.
    go_left_to_right = !go_left_to_right;
    result.push_back(row);
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

bool is_balanced(TreeNode* root)
{
  const auto result = balance_max_height_recursive(root);

  return result.first;
}

pair<bool, int> balance_max_height_recursive(TreeNode* root)
{
  // Base cases.

  if (root == nullptr)
  {
    return make_pair<bool, int>(true, -1);
  }

  // Root node is a leaf.
  if (root->left_ == nullptr && root->right_ == nullptr)
  {
    return make_pair<bool, int>(true, 0);
  }

  const auto left_result {balance_max_height_recursive(root->left_)};
  const auto right_result {balance_max_height_recursive(root->right_)};

  int max_height {max(left_result.second, right_result.second)};
  int height_difference {std::abs(left_result.second - right_result.second)};
  bool is_balanced {
    height_difference <= 1 &&
      left_result.first &&
        right_result.first};

  return make_pair<bool, int>(std::move(is_balanced), max_height + 1);
}

bool is_same_recursive(TreeNode* root1, TreeNode* root2)
{
  if (root1 == nullptr)
  {
    return root2 == nullptr ? true : false;
  }

  // root1 != nullptr then.
  if (root2 == nullptr)
  {
    return false;
  }

  const bool is_values_equal {root1->value_ == root2->value_};

  return is_values_equal &&
    is_same_recursive(root1->left_, root2->left_) &&
      is_same_recursive(root1->right_, root2->right_);
}

bool is_symmetric(TreeNode* root)
{
  if (root == nullptr)
  {
    return true;
  }

  if (root->left_ == nullptr)
  {
    return (root->right_ == nullptr) ? true : false;
  }

  if (root->right_ == nullptr)
  {
    return false;
  }

  //queue<TreeNode*> node_ptr_stack;
  return true;
}

string serialize(TreeNode* root)
{
  if (root == nullptr)
  {
    return "null";
  }

  // For a leaf, we don't want to continue further in serialization.
  //if ((root->left_ == nullptr) && (root->right_ == nullptr))
  //{
  //  return to_string(root->value_);   
  //}

  return to_string(root->value_) +
    "," +
    serialize(root->left_) +
      "," +
        serialize(root->right_);
}

int parse_node_value(string& data)
{
  const auto next_delimiter_position = data.find(",");

  int value {stoi(data.substr(0, next_delimiter_position))};

  // Remove the parsed out value from the original string.
  // The range is [next_delimiter_position + 1, ) for only one argument.
  data = data.substr(next_delimiter_position + 1);

  return value;
}

} // namespace BinaryTrees
} // namespace DataStructures
