#ifndef DATA_STRUCTURES_TREES_BINARY_TREES_BST_H
#define DATA_STRUCTURES_TREES_BINARY_TREES_BST_H

#include <cassert>

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

    virtual ~BST() = default;

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
      return false;
    }

    BST& remove(const T val)
    {
      // Do not edit the return statement of this method.
      return* this;
    }

    static bool is_leaf(BST* node_ptr)
    {
      return (node_ptr->left_ == nullptr) && (node_ptr->right_ == nullptr);
    }
};

} // namespace ExpertIO

} // namespace BinaryTrees
} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_BINARY_TREES_BST_H