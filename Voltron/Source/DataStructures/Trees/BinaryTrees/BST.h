#ifndef DATA_STRUCTURES_TREES_BINARY_TREES_BST_H
#define DATA_STRUCTURES_TREES_BINARY_TREES_BST_H

#include <cassert>
#include <stdexcept>

namespace DataStructures
{
namespace Trees
{
namespace BinaryTrees
{

namespace ExpertIO
{

//------------------------------------------------------------------------------
/// \details Medium
/// BST Construction
/// Write a BST class for a Binary Search Tree. The class should support:
/// * Inserting values with the insert method.
/// * Removing values with the remove method; this method should only remove the
/// first instance of a given value.
/// * Searching for values with the contains method. 
/// \ref https://www.algoexpert.io/questions/BST%20Construction
//------------------------------------------------------------------------------

template <typename T>
class BST
{
  public:

    T value_;
    BST* left_;
    BST* right_;

    BST(const T val, BST* left=nullptr, BST* right=nullptr):
      value_{val},
      left_{left},
      right_{right}
    {}

    //virtual ~BST() = default;

    //--------------------------------------------------------------------------
    /// For insert, contains, remove
    /// Average: O(log(N)) time
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    /// Average: O(log(N)) time | O(log(N)) space
    /// Worst: O(N) time | O(N) space.
    /// \details O(log(N)) space and O(N) space respectively  for the recursion
    /// stack frames called.
    //--------------------------------------------------------------------------
    BST& insert_recursive_algoexpert(T val)
    {
      if (val < value_)
      {
        if (left_ == nullptr)
        {
          BST* new_bst {new BST{val}};
          left_ = new_bst;
        }
        else
        {
          left_->insert_recursive_algoexpert(val);
        }
      }
      else
      {
        if (right_ == nullptr)
        {
          BST* new_bst {new BST{val}};
          right_ = new_bst;
        }
        else
        {
          right_->insert_recursive_algoexpert(val);
        }
      }

      return *this;
    }

    /*
    {
      post_order_deletion(this);
    }

    void post_order_deletion(BST<T>* node_ptr)
    {
      if (node_ptr == nullptr)
      {
        return;
      }

      // Recursively traverse left subtree.
      post_order_deletion(node_ptr->left_);

      // Recursively traverse right subtree.
      post_order_deletion(node_ptr->right_);

      // Visit current node. This is guaranteed to be a leaf because to get to
      // this line of code, the code had to exit the above 2 statements by
      // visiting nullptr at the left, and nullptr at the right.
      // Release memory pointed to by pointer-variable.
      delete node_ptr;
      node_ptr = nullptr;
    }
    */

    //--------------------------------------------------------------------------
    /// \details Average: O(log(n)) time | O(1) space
    /// Worst: O(n) time | O(1) space
    /// Puts duplicate values to the right.
    //--------------------------------------------------------------------------
    BST& insert_iteratively(const T val)
    {
      BST* current_node_ptr = this;
      while (current_node_ptr != nullptr)
      {
        const T current_value {current_node_ptr->value_};

        if (val < current_value)
        {
          if (current_node_ptr->left_ == nullptr)
          {
            // cf. https://en.wikipedia.org/wiki/New_and_delete_(C%2B%2B)
            // new operator denotes a request for memory allocation on a
            // process's heap. If sufficient memory is available, new
            // initializes memory, call object ctors, returns address to newly
            // allocated and initialized memory.
            BST* new_node {new BST{val}};
            current_node_ptr->left_ = new_node;
            break;
          }
          else
          {
            current_node_ptr = current_node_ptr->left_;
          }
        }
        else if (val > current_value)
        {
          if (current_node_ptr->right_ == nullptr)
          {
            BST* new_node {new BST{val}};
            current_node_ptr->right_ = new_node;
            break;
          }
          else
          {
            current_node_ptr = current_node_ptr->right_;
          }
        }
        else
        {
          assert(val == current_value);
          BST* new_node {new BST{val}};
          // Put duplicate values to the right.
          if (current_node_ptr->right_ == nullptr)
          {
            current_node_ptr->right_ = new_node;
          }
          else
          {
            BST* temp_ptr {current_node_ptr->right_};
            current_node_ptr->right_ = new_node;
            current_node_ptr->right_->right_ = temp_ptr;
          }
          break;
        }
      }

      // Do not edit the return statement of this method.
      return* this;
    }

    bool contains(const T val)
    {
      BST* current_node_ptr {this};
      const T current_value {current_node_ptr->value_};

      if (current_value == val)
      {
        return true;
      }

      if (val < current_value && current_node_ptr->left_ != nullptr)
      {
        return current_node_ptr->left_->contains(val);
      }

      if (val > current_value && current_node_ptr->right_ != nullptr)
      {
        return current_node_ptr->right_->contains(val);
      }

      return false;
    }

    BST& remove_recursive_algoexpert(const T val, BST* parent = nullptr)
    {
      if (val < value_)
      {
        if (left_ != nullptr)
        {
          left_->remove(val, this);
        }
      }
      else if (val > value_)
      {
        if (right_ != nullptr)
        {
          right_->remove(val, this);
        }
      }
      else
      {
        if (left_ != nullptr && right_ != nullptr)
        {
          value_ = right_->get_min();
          // Remove that minimum value.
          return right_->remove(value_, this);
        }
        else if (parent == nullptr)
        {
          // All elements in the left subtree. Rebalance once.
          if (left_ != nullptr)
          {
            value_ = left_->value_;
            right_ = left_->right_;
            left_ = left_->left_;
          }
          // All elements in the right subtree. Rebalance once.
          else if (right_ != nullptr)
          {
            value_ = right_->value_;
            left_ = right_->left_;
            right_ = right_->right_;
          }
          else
          {
            // This is a single-node tree; do nothing.
          }
        }
        else if (parent->left_ == this)
        {
          parent->left_ = left_ != nullptr ? left_ : right_;
        }
        else if (parent->right_ == this)
        {
          parent->right_ = left_ != nullptr ? left_ : right_;
        }
        else
        {
          throw std::runtime_error("parent input is not a parent!");
        }
      }

      return *this;
    }

    //--------------------------------------------------------------------------
    /// \details Hint 2 Traverse the BST all the while applying the logic
    /// described in Hint #1. For insertion, add the target value to the BST
    /// once you reach a leaf (None / null) node. For searching, if you reach a
    /// leaf node without having found the target value that means the value
    /// isn't in the BST. For removal, consider the various cases that you might
    /// encounter; the node you need to remove might have 2 children nodes, 1,
    /// none; it might also be the root node; make sure to account for all of
    /// these cases.
    ///
    /// Video Explanation. Conceptual Overview. 12:53. Easy cases: leaf.
    /// Remove root: grab smallest value in the right subtree. e.g. 12. The
    /// smallest value in right subtree will still be greater than the value,
    /// and will be the smallest of all values greater than the value.
    //--------------------------------------------------------------------------

    BST& remove(const T val)
    {
      BST* current_node_ptr {this};
      BST* previous_node_ptr {this};

      /*
      while (current_node_ptr != nullptr)
      {
        const T current_value {current_node_ptr->value_};

        if (current_value == val)
        {
          if (current_node_ptr == previous_node_ptr)
          {
            return *this;
          }

          if (previous_node_ptr->left_ == current_node_ptr)
          {
            if 
          }
        }
      }
      */

      // Do not edit the return statement of this method.
      return* this;
    }

    static bool is_leaf(BST* node_ptr)
    {
      return (node_ptr->left_ == nullptr) && (node_ptr->right_ == nullptr);
    }

    T get_min()
    {
      if (left_ == nullptr)
      {
        return value_;
      }

      return left_->get_min();
    }
};

} // namespace ExpertIO

} // namespace BinaryTrees
} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_BINARY_TREES_BST_H