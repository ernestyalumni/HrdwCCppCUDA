#ifndef DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_NODE_H
#define DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_NODE_H

#include <algorithm>
#include <cstddef>

namespace DataStructures
{
namespace Trees
{
namespace BinaryTrees
{

namespace DWHarder
{

//------------------------------------------------------------------------------
/// \ref 5.01.Binary_trees.pptx
//------------------------------------------------------------------------------
template <typename Type>
class BinaryNode
{
  public:

    BinaryNode(const Type&);

    // Accessors.

    Type value() const;
    BinaryNode* left() const;
    BinaryNode* right() const;

    bool is_leaf() const;

    //--------------------------------------------------------------------------
    /// Recursive size function runs in O(N) time and O(h) memory.
    /// \ref 5.01.Binary_trees.pptx, Slide 15
    //--------------------------------------------------------------------------
    std::size_t size() const;
    std::size_t height() const;
    void clear();

    void set_left(BinaryNode* left)
    {
      p_left_tree_ = left;
    }

    void set_right(BinaryNode* right)
    {
      p_right_tree_ = right;
    }

  protected:

    Type node_value_;
    BinaryNode* p_left_tree_;
    BinaryNode* p_right_tree_;
};

template <typename Type>
BinaryNode<Type>::BinaryNode(const Type& object):
  node_value_{object},
  p_left_tree_{nullptr},
  p_right_tree_{nullptr}
{
  // Empty constructor.
}

template <typename Type>
void BinaryNode<Type>::clear()
{
  if (left() != nullptr)
  {
    left()->clear();
  }

  if (right() != nullptr)
  {
    right()->clear();
  }

  // Reach a leaf.
  delete this;
}

// Accessors.

template <typename Type>
Type BinaryNode<Type>::value() const
{
  return node_value_;
}

template <typename Type>
BinaryNode<Type>* BinaryNode<Type>::left() const
{
  return p_left_tree_;
}

template <typename Type>
BinaryNode<Type>* BinaryNode<Type>::right() const
{
  return p_right_tree_;
}

template <typename Type>
bool BinaryNode<Type>::is_leaf() const
{
  return (left() == nullptr) && (right() == nullptr);
}

} // namespace DWHarder

template <typename T>
class Node
{
  public:

    Node(const T& object = T{}):
      value_{object},
      left_{nullptr},
      right_{nullptr}
    {}

    Node(
      const T& object,
      Node* left,
      Node* right = nullptr
      ):
      value_{object},
      left_{left},
      right_{right}
    {}

    virtual ~Node() = default;

    bool is_leaf() const
    {
      return (left_ == nullptr) && (right_ == nullptr);
    }

    std::size_t size() const
    {
      if (left_ == nullptr)
      {
        // Reached a leaf or keep going along right subtree.
        return (right_ == nullptr) ? 1 : 1 + right_->size();
      }
      else
      {
        return (right_ == nullptr) ?
          1 + left_->size() :
          1 + left_->size() + right_->size();
      }
    }

    std::size_t height() const
    {
      if (left_ == nullptr)
      {
        // Reached a leaf or keep going along right subtree.
        return (right_ == nullptr) ? 0 : 1 + right_->height();
      }
      else
      {
        return (right_ == nullptr) ?
          1 + left_->height() :
          1 + std::max(left_->height(), right_->height());
      }
    }

    T value_;
    Node* left_;
    Node* right_;
};

} // namespace BinaryTrees
} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_NODE_H