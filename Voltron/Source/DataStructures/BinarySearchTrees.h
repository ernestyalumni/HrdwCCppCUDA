//------------------------------------------------------------------------------
/// \file BinarySearchTrees.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Binary Search Tree
/// \details 
///
/// \ref https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/140/introduction-to-a-bst/997/
/// \ref LeetCode, Binary Search Tree
//------------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_BINARY_SEARCH_TREES_H
#define DATA_STRUCTURES_BINARY_SEARCH_TREES_H

#include <stack>
#include <stdexcept> // std::runtime_error
#include <vector>

namespace DataStructures
{

namespace BinarySearchTrees
{

//------------------------------------------------------------------------------
/// \brief Definition for a binary tree node.
//------------------------------------------------------------------------------

struct TreeNode
{
  int value_;
  TreeNode* left_;
  TreeNode* right_;

  //----------------------------------------------------------------------------
  /// \brief Default constructor.
  //----------------------------------------------------------------------------
  TreeNode();

  TreeNode(int x);

  TreeNode(int x, TreeNode *left, TreeNode *right);
};

template <typename NodeType, typename F>
void inorder_traversal(NodeType* node_ptr, F& f)
{
  // Base case
  if (node_ptr == nullptr)
  {
    return;
  }

  // First recur (do recurrence) on left child.
  inorder_traversal(node_ptr->left_, f);
  // Returns out of the recurrence if the left is a nullptr.
  f(node_ptr->value_);

  // Now recur (recurrence) on right child.
  inorder_traversal(node_ptr->right_, f);
}

/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/140/introduction-to-a-bst/997/discuss/895234/C++-two-easy-recursive-and-iterative-DFS-solutions
bool iterative_validate_binary_search_tree(TreeNode* root);

/// \details Uses recursive inorder traversal.
bool is_valid_binary_search_tree(TreeNode* root);

//------------------------------------------------------------------------------
/// \class InorderBstIterator
/// \brief Definition for a binary tree node.
/// \url https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion/
/// \details Use a stack to traverse a tree without recursion. 
///
/// Algorithm:
/// 1. Create an empty stack S.
/// 2. Initialize current node with root.
/// 3. Push the current node to S and set current = current->left until the
/// current is nullptr.
/// 4. If current is nullptr, and stack is not empty, then
///   a) Pop the top item from the stack.
///   b) Print the popped item, set current = popped_item->right
///   c) Go to step 3.
/// 5. If current is nullptr, and the stack is empty, then we are done.
//------------------------------------------------------------------------------

template <typename NodeType, typename T>
class InorderBstIterator
{
  public:

    InorderBstIterator(NodeType* root):
      // 1. Create an empty stack S.
      node_stack_{},
      // 2. Initialize current node with the root
      current_node_ptr_{root}
    {}

    T next()
    {
      if (!has_next())
      {
        throw std::runtime_error("No more nodes");
      }

      // 3. Push current mode to stack, and set current = current->left until
      // current is nullptr.
      while (current_node_ptr_ != nullptr)
      {
        node_stack_.push(current_node_ptr_);
        current_node_ptr_ = current_node_ptr_->left_;
      }

      // Surely, node_stack_ is not empty.
      NodeType* popped_item {node_stack_.top()};
      // 4a. Pop the top item from the stack.
      node_stack_.pop();
      // 4b. Set current = popped_item->right.
      current_node_ptr_ = popped_item->right_;

      // 4.b) "Print" the popped item.
      return popped_item->value_;

      // Go to step 3.
    }

    bool has_next()
    {
      if ((current_node_ptr_ == nullptr) && (node_stack_.empty()))
      {
        return false;
      }

      return true;
    }

  private:

    std::stack<NodeType*> node_stack_;
    NodeType* current_node_ptr_;
};

//------------------------------------------------------------------------------
/// \class BSTIterator
/// \brief Definition for a binary tree node.
//------------------------------------------------------------------------------
/*
class BstIterator
{
  public:

    explicit BstIterator(TreeNode* root);

    // \return the next smallest number
    int next();

    // \return whether we have a next smallest number
    bool has_next();

  private:
};
*/

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/141/basic-operations-in-a-bst/1000/
/// \brief Search in a Binary Search Tree.
///
/// Given root node of a binary search tree (BST) and a value, find the node in
/// the BST that the node's value equals the given value.
//------------------------------------------------------------------------------
template <typename NodeType, typename T>
NodeType* search_bst(NodeType* node_ptr, T target_value)
{
  // Base cases.
  if (node_ptr == nullptr)
  {
    return node_ptr;
  }

  if (target_value == node_ptr->value_)
  {
    return node_ptr;
  }

  return target_value < node_ptr->value_ ?
    search_bst(node_ptr->left_, target_value) :
      search_bst(node_ptr->right_, target_value);
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/141/basic-operations-in-a-bst/1003/
/// \brief Insert in a Binary Search Tree.
///
/// Given root node of a binary search tree (BST) and a value, find the node in
/// the BST that the node's value equals the given value.
///
/// Assume that target_value does not exist in the original BST.
/// Iterative.
//------------------------------------------------------------------------------
template <typename NodeType, typename T>
NodeType* insert_into_bst(NodeType* root, T target_value)
{
  NodeType* current_node_ptr {root};
  NodeType* previous_node_ptr {nullptr};

  // Traverse the tree, starting from the root, with comparison to the
  // target_value.
  // Stop when we get to a leaf.
  while (current_node_ptr != nullptr)
  {
    /*
    previous_node_ptr = current_node_ptr;

    // It is guaranteed that new value does not exist in the original BST.
    target_value < current_node_ptr->value_ ?
      current_node_ptr = current_node_ptr->left_ :
        current_node_ptr = current_node_ptr->right_;
    */

    previous_node_ptr = current_node_ptr;

    if (target_value < current_node_ptr->value_)
    {
      current_node_ptr = current_node_ptr->left_;
    }
    else
    {
      current_node_ptr = current_node_ptr->right_;
    }
  }

  //NodeType new_node {target_value};
  // Need to use new to allocate dynamic memory; otherwise
  // unknown location(0): fatal error: in
  // "DataStructures/BinarySearchTrees_tests/InsertIntoBst": memory access
  // violation at address: 0x00000010: no mapping at fault address
  // =================================================================
  // ==31==ERROR: AddressSanitizer: stack-buffer-underflow on address 0x7ffdb743e2a8 at pc 0x0000003d7ab1 bp 0x7ffdb743de30 sp 0x7ffdb743de28
  // READ of size 8 at 0x7ffdb743e2a8 thread T0
  // #7 0x7f93c7c9b0b2  (/lib/x86_64-linux-gnu/libc.so.6+0x270b2)
  // Address 0x7ffdb743e2a8 is located in stack of thread T0 at offset 8 in frame
  // cf. https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/141/basic-operations-in-a-bst/1003/
  NodeType* new_node_ptr = new NodeType(target_value);

  // Attach this new node to either the left or right.
  if (previous_node_ptr != nullptr)
  {
    if (target_value < previous_node_ptr->value_)
    {
      previous_node_ptr->left_ = new_node_ptr;
    }
    else
    {
      previous_node_ptr->right_ = new_node_ptr;
    }
  }

  //target_value < previous_node_ptr->value_ ?
  //  previous_node_ptr->left_ = &new_node :
  //    previous_node_ptr->right_ = &new_node;

  return root ? root : new_node_ptr;
}

//------------------------------------------------------------------------------
/// \brief Deletion in a BST.
//------------------------------------------------------------------------------
template <typename NodeType, typename T>
class DeleteValue
{
  public:

    DeleteValue():
      stack_{}
    {}

    // https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/141/basic-operations-in-a-bst/1006/discuss/93394/Concise-and-Clear-C++-Solution
    NodeType* recursive_delete_value(NodeType* node_ptr, T target_value)
    {
      // Not found, if reached here.
      if (node_ptr == nullptr)
      {
        return node_ptr;
      }

      if (target_value < node_ptr->value_)
      {
        node_ptr->left_ = recursive_delete_value(node_ptr->left_, target_value);
      }
      else if (node_ptr->value_ < target_value)
      {
        node_ptr->right_ =
          recursive_delete_value(node_ptr->right_, target_value);
      }
      // target_value found. So node_ptr->value_ == target_value
      else
      {
        if (node_ptr->left_ == nullptr)
        {
          NodeType* right {node_ptr->right_};
          delete node_ptr;
          return right;
        }
        else if (node_ptr->right == nullptr)
        {
          NodeType* left {node_ptr->left_};
          delete node_ptr;
          return left;
        }
        else
        {
          // Find the inorder successor (the minimal node in right subtree).
          NodeType* successor {find_next_inorder_node(node_ptr->right_)};

          node_ptr->value_ = successor->value_;
          node_ptr->right_ =
            recursive_delete_value(node_ptr->right_, successor->value_);
        }
      }
      return node_ptr;
    }

    NodeType* find_next_inorder_node(NodeType* node_ptr)
    {
      if (node_ptr->left_ != nullptr)
      {
        return find_next_inorder_node(node_ptr->left_);
      }

      // Base case - node_ptr->left_ == nullptr
      return node_ptr;
    }


    // Still wrong, here's the test case it fails on.
    //Input: [4,null,7,6,8,5,null,null,9]
    // 7
    // Output: [4,null,8,6,null,5]
    // Expected: [4,null,8,6,9,5]

    NodeType* delete_value(NodeType* root, T target_value)
    {
      NodeType* current_node_ptr {root};

      while (current_node_ptr != nullptr)
      {
        stack_.push(current_node_ptr);

        if (current_node_ptr->value_ == target_value)
        {
          break;
        }

        // Traverse left or right depending upon if the target value is less
        // than or greater than the current node's value, the node we're on.
        target_value < current_node_ptr->value_ ?
          current_node_ptr = current_node_ptr->left_ :
            current_node_ptr = current_node_ptr->right_;
      }

      // Not able to find a node containing the target_value and so we exit:
      // nothing to delete.
      if (current_node_ptr == nullptr)
      {
        return root;
      }

      // Otherwise, now process the current node that we are in.
      // If the current node is childless, we can simply remove the node.
      if ((current_node_ptr->left_ == nullptr) &&
        (current_node_ptr->right_ == nullptr))
      {
        stack_.pop();

        if (!stack_.empty())
        {
          NodeType* previous_node_ptr {stack_.top()};

          if (previous_node_ptr->left_ != nullptr &&
            previous_node_ptr->left_->value_ == target_value)
          {
            // Effectively delete the target node.
            previous_node_ptr->left_ = nullptr;
          }
          else if (previous_node_ptr->right_ != nullptr &&
            previous_node_ptr->right_->value_ == target_value)
          {
            // Effectively delete the target node.
            previous_node_ptr->right_ = nullptr;
          }
        }
        else
        {
          // current_node_ptr contains the target_value and is meant to be
          // deleted, but it has no parent. Therefore, conclude to return
          // nullptr as it must be the root (by definition of a root node).
          return nullptr;
        }
      }

      // If the target node has one child, we can use its child to replace
      // itself.
      if (current_node_ptr->left_ != nullptr &&
        current_node_ptr->right_ == nullptr)
      {
        current_node_ptr->value_ = current_node_ptr->left_->value_;
        current_node_ptr->left_ = nullptr;
      }
      else if (current_node_ptr->left_ == nullptr &&
        current_node_ptr->right_ != nullptr)
      {
        current_node_ptr->value_ = current_node_ptr->right_->value_;
        current_node_ptr->right_ = nullptr;
      }
      // If the target node has 2 children, replace the node with its inorder
      // successor (can also be predecessor, but it's chosen to be successor
      // here) node and delete the node.
      else if (current_node_ptr->left_ != nullptr &&
        current_node_ptr->right_ != nullptr)
      {
        NodeType* successor_node_ptr {current_node_ptr->right_};

        // Helps us distinguish whether we will be deleting a left child or
        // right child.
        int axis {1}; // 0 for left, 1 for right.

        // Do one iteration first. Will continue if the while condition is
        // fulfilled.
        stack_.push(successor_node_ptr);
        successor_node_ptr = successor_node_ptr->left_;

        // Find the next inorder successor.
        while (successor_node_ptr != nullptr)
        {
          axis = 0;
          stack_.push(successor_node_ptr);
          successor_node_ptr = successor_node_ptr->left_;
        }

        // Surely, node stack is not empty, least because the node with the
        // target value had both its children.
        NodeType* popped_item {stack_.top()};
        stack_.pop();

        NodeType* parent_item {stack_.top()};
        stack_.pop();

        if (!stack_.empty())
        {
          // Replace value of the target node with its inorder successor.
          current_node_ptr->value_ = popped_item->value_;

          axis == 0 ? parent_item->left_ = nullptr :
            parent_item->right_ = nullptr;
        }
        else
        {
          popped_item->left_ = current_node_ptr->left_;
          current_node_ptr->left_ = nullptr;
          current_node_ptr->right_ = nullptr;
          root = popped_item;
        }
      }

      return root;
    }

  private:

    std::stack<NodeType*> stack_;
};

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/142/conclusion/1009/
/// \brief Problem: Design a class to find the kth largest element in a stream.
///
/// Given root node of a binary search tree (BST) and a value, find the node in
/// the BST that the node's value equals the given value.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \brief Definition for a binary tree node with a counter.
//------------------------------------------------------------------------------

struct NodeWithCounter
{
  int value_;
  int counter_;
  NodeWithCounter* left_;
  NodeWithCounter* right_;

  //----------------------------------------------------------------------------
  /// \brief Default constructor.
  //----------------------------------------------------------------------------
  NodeWithCounter();

  explicit NodeWithCounter(int x);

  NodeWithCounter(int x, NodeWithCounter *left, NodeWithCounter *right);
};

class TreeWithCounter
{
  public:

    TreeWithCounter();

    TreeWithCounter(std::vector<int>& nums);

    ~TreeWithCounter();

    // Take a look at insert_into_bst for this iterative approach.
    NodeWithCounter* insert_new_value(const int num);

    NodeWithCounter* root_ptr() const
    {
      return root_ptr_;
    }

    bool is_counter_stack_empty() const
    {
      return counter_stack_.empty();
    }

    //NodeWithCounter* find_kth_largest_element(const int k);
    int find_kth_largest_element(const int k);

  private:

    std::stack<NodeWithCounter*> counter_stack_;
    NodeWithCounter* root_ptr_;
};

int kth_largest_element_reverse_inorder_traversal(
  TreeNode* root_ptr,
  int k,
  int& count);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/interview/card/top-interview-questions-medium/108/trees-and-graphs/790/
/// \brief Kth smallest element in a BST
/// \details Given the root of a binary search tree, and an integer k, return
/// the kth (1-indexed) smallest element in the tree.
//------------------------------------------------------------------------------
int kth_smallest_element(TreeNode* root, int k);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/142/conclusion/1018/
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/143/appendix-height-balanced-bst/1015/
/// \ref https://www.geeksforgeeks.org/convert-normal-bst-balanced-bst/
//------------------------------------------------------------------------------
TreeNode* sorted_array_to_BST(std::vector<int>& nums);

TreeNode* sorted_array_to_BST_recursive_step(
  std::vector<int>& nums,
  const std::size_t start,
  const std::size_t end);

TreeNode* sorted_array_to_BST_iterative(std::vector<int>& nums);

} // namespace BinarySearchTrees
} // namespace DataStructures

#endif // DATA_STRUCTURES_BINARY_SEARCH_TREES_H