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
#include <vector>

using std::stack;
using std::vector;

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

NodeWithCounter::NodeWithCounter() :
  value_{0},
  counter_{0},
  left_{nullptr},
  right_{nullptr}
{}

NodeWithCounter::NodeWithCounter(int x) :
  value_{x},
  counter_{0},
  left_{nullptr},
  right_{nullptr}
{}

NodeWithCounter::NodeWithCounter(
  int x,
  NodeWithCounter *left,
  NodeWithCounter *right
  ) :
  value_{x},
  counter_{0},
  left_{left},
  right_{right}
{}

TreeWithCounter::TreeWithCounter():
  counter_stack_{},
  root_ptr_{nullptr}
{}

TreeWithCounter::TreeWithCounter(vector<int>& nums):
  counter_stack_{},
  root_ptr_{nullptr}
{
  for (auto num : nums)
  {
    insert_new_value(num);
  }
}

TreeWithCounter::~TreeWithCounter()
{
  auto is_leaf = [](auto node_ptr) -> bool
  {
    if (node_ptr->left_ == nullptr && node_ptr->right_ == nullptr)
    {
      return true;
    }

    return false;
  };

  stack<NodeWithCounter*> node_ptr_stack;

  NodeWithCounter* current_node_ptr = root_ptr_;

  if (current_node_ptr == nullptr)
  {
    return;
  }

  while (current_node_ptr != nullptr || !node_ptr_stack.empty())
  { 
    if (is_leaf(current_node_ptr))
    {
      delete current_node_ptr;
    }
    else
    {
      while (current_node_ptr != nullptr)
      {
        node_ptr_stack.push(current_node_ptr);
        current_node_ptr = current_node_ptr->left_;
      }

      NodeWithCounter* popped_item {node_ptr_stack.top()};
      node_ptr_stack.pop();

      current_node_ptr = popped_item->right_;
    }
  }
}

NodeWithCounter* TreeWithCounter::insert_new_value(const int num)
{
  NodeWithCounter* current_node_ptr {root_ptr_};
  NodeWithCounter* previous_node_ptr {nullptr};

  // Take a look at insert_into_bst for this iterative approach.
  // Traverse the tree, starting from the root, with comparison to num.
  // Stop when we get to a leaf.
  while (current_node_ptr != nullptr)
  {
    counter_stack_.push(current_node_ptr);

    previous_node_ptr = current_node_ptr;

    if (num < current_node_ptr->value_)
    {
      current_node_ptr = current_node_ptr->left_;
    }
    // Need to assume that num is not in the tree yet.
    // Update: Can't assume that.
    else if (num > current_node_ptr->value_)
    {
      current_node_ptr = current_node_ptr->right_;
    }
/*    else
    {
      break;
    }
*/
  }

  NodeWithCounter* new_node_ptr = new NodeWithCounter(num);

  // Attach this new node to either the left or right.
  if (previous_node_ptr != nullptr)
  {
    if (num < previous_node_ptr->value_)
    {
      previous_node_ptr->left_ = new_node_ptr;
    }
    else
    {
      previous_node_ptr->right_ = new_node_ptr;
    }
  }

  // Go "up" the tree and increment each.
  while (!counter_stack_.empty())
  {
    counter_stack_.top()->counter_ += 1;
    counter_stack_.pop();
  }

  // Do this after the counter increment.
  /*
  if (current_node_ptr != nullptr)
  {
    new_node_ptr->right_ = current_node_ptr;
    new_node_ptr->counter_ = current_node_ptr->counter_;
    current_node_ptr->counter_ -= 1;
  }
  */


  if (root_ptr_ == nullptr)
  {
    root_ptr_ = new_node_ptr;
  }

  return root_ptr_;
}

// cf. https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/142/conclusion/1018/discuss/386215/C++-solution-without-heap-(BST)
//NodeWithCounter* TreeWithCounter::find_kth_largest_element(const int k)
int TreeWithCounter::find_kth_largest_element(const int k)
{
  if (k < 1)
  {
//    return nullptr;
    return 0;
  }

  if (k > (root_ptr_->counter_ + 1));
  {
    //return nullptr;
    // Debug.
    //return root_ptr_->counter_ + 42;
  }

  //stack<NodeWithCounter*> node_ptr_stack;
  NodeWithCounter* current_node_ptr {root_ptr_};
  int mth_counter {k};

  while (current_node_ptr != nullptr)
  {
    // Number of nodes in the right subtree
    const int number_of_right_nodes {current_node_ptr->right_ == nullptr ? 0 :
      current_node_ptr->right_->counter_ + 1};

    // Include now the current_node_ptr.
    const int total_number_of_nodes {number_of_right_nodes + 1};

    if (mth_counter == total_number_of_nodes)
    {
//      return current_node_ptr;
      return current_node_ptr->value_;
    }
    // Not enough elements in the right half of nodes, right subtree, to be the
    // mth largest element.
    else if (mth_counter > total_number_of_nodes)
    {
      mth_counter -= total_number_of_nodes;
      current_node_ptr = current_node_ptr->left_;
    }
    // mth_counter < total_number_of_nodes; this means that the mth element
    // we're looking for is amongst the right sub-tree.
    else
    {
      current_node_ptr = current_node_ptr->right_;
    }
  }

  return 0;

  //return current_node_ptr;

  /*
  while (current_node_ptr != nullptr || !node_ptr_stack.empty())
  {
    if (current_node_ptr != nullptr && !node_ptr_stack.empty())
    {
      // This means we cannot find a kth counter'th element amongst the current
      // node or any of its children (i.e.not enough elements).
      if (kth_counter > current_node_ptr->counter_ + 1)
      {
        kth_counter -= (current_node_ptr->counter_ + 1);

        current_node_ptr = node_ptr_stack.top()
        node_ptr_stack.pop();
      }
      else if (kth_counter < current_node_ptr->counter_ + 1)
      {
        if (current_node_ptr->right_ != nullptr)
        {
          node_ptr_stack.push(current_node_ptr);
          current_node_ptr = current_node_ptr->right_;
        }
        else
        {

        }
      }
    }

    // This means we can find a kth_counter'th element amongst the current node
    // or any of its children.
    if (kth_counter <= current_node_ptr->counter_ + 1)
    {
    }
    else
    {

      current_node_ptr = node_ptr_stack.top();
      node_ptr_stack.pop();
    }
  }
  */
}

} // namespace BinarySearchTrees
} // namespace DataStructures

