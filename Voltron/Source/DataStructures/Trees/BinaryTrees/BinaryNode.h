#ifndef DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_NODE_H
#define DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_NODE_H

#include <algorithm>
#include <cstddef>
#include <iostream>

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

//------------------------------------------------------------------------------
/// \ref Exercise 10.4-2, pp. 248, Ch. 10 "Elementary Data Structures", Cormen,
/// Leiserson, Rivest, Stein.
//------------------------------------------------------------------------------

template <typename T>
void print_preorder_traversal(Node<T>* node)
{
  if (node == nullptr)
  {
    return;
  }

  // Visit current node.
  std::cout << node->value_ << " ";

  // Recursively traverse left subtree.
  print_preorder_traversal<T>(node->left_);

  // Recursively traverse right subtree.
  print_preorder_traversal<T>(node->right_);
}

template <typename T>
void print_postorder_traversal(Node<T>* node)
{
  if (node == nullptr)
  {
    return;
  }

  // Recursively traverse left subtree.
  print_postorder_traversal<T>(node->left_);

  // Recursively traverse right subtree.
  print_postorder_traversal<T>(node->right_);

  // Visit current node.
  std::cout << node->value_ << " ";
}

template <typename T>
void print_inorder_traversal(Node<T>* node)
{
  if (node == nullptr)
  {
    return;
  }

  // Recursively traverse left subtree.
  print_inorder_traversal<T>(node->left_);

  // Visit current node.
  std::cout << node->value_ << " ";

  // Recursively traverse right subtree.
  print_inorder_traversal<T>(node->right_);
}

//------------------------------------------------------------------------------
/// TODO: This exhibits an interesting order that isn't easily achieved by the
/// recursive version.
//------------------------------------------------------------------------------
template <typename T, template<typename> class StackT>
void print_inorder_traversal_with_stack_v1(Node<T>* node)
{
  if (node == nullptr)
  {
    return;
  }

  StackT<Node<T>*> s {};
  Node<T>* current_ptr {node};

  s.push(node);

  while (!s.is_empty())
  {
    Node<T>* current_node {s.pop()};

    if (current_node != nullptr)
    {
      std::cout << current_node->value_ << " ";

      // Reverse since a stack is LIFO.
      s.push(current_node->right_);
      s.push(current_node->left_);
    }
  }
}

//------------------------------------------------------------------------------
/// \ref https://medium.com/@ajinkyajawale/inorder-preorder-postorder-traversal-of-binary-tree-58326119d8da
//------------------------------------------------------------------------------
template <typename T, template<typename> class StackT>
void print_inorder_traversal_with_stack(Node<T>* node)
{
  if (node == nullptr)
  {
    return;
  }

  StackT<Node<T>*> s {};
  Node<T>* current_ptr {node};

  while (!s.is_empty() || current_ptr != nullptr)
  {
    // Traverse left subtree first, until leaf is reached.
    while (current_ptr != nullptr)
    {
      // Save the "root", i.e. push the "root" onto the stack and set root =
      // root.left and continue until it hits nullptr.
      s.push(current_ptr);

      current_ptr = current_ptr->left_;
    }
    // Exited because a leaf was found and current_ptr == nullptr now. Go back
    // to the stack for the previous "root".
    current_ptr = s.pop();

    std::cout << current_ptr->value_ << " ";

    // Traverse the right subtree.
    current_ptr = current_ptr->right_;
  }
}

template <typename T, template<typename> class StackT>
void print_preorder_traversal_with_stack(Node<T>* node)
{
  if (node == nullptr)
  {
    return;
  }

  StackT<Node<T>*> s {};
  s.push(node);

  while (!s.is_empty())
  {
    Node<T>* current_ptr {s.pop()};

    // Effectively visit the "root" node first.
    std::cout << current_ptr->value_ << " ";

    // Push right before left so that, due to property of a stack, left is
    // processed before right.
    if (current_ptr->right_ != nullptr)
    {
      s.push(current_ptr->right_);
    }
    // We've put the right subtree before the stack (so it's "below") so that
    // the left subtree will be popped off in the next iteration.
    if (current_ptr->left_ != nullptr)
    {
      s.push(current_ptr->left_);
    }
  }
}

template <typename T, template<typename> class StackT>
void print_postorder_traversal_with_stack(Node<T>* node)
{
  if (node == nullptr)
  {
    return;
  }

  StackT<Node<T>*> s {};
  s.push(node);
  Node<T>* current_ptr {node};
  Node<T>* previous_ptr {nullptr};

  while (!s.is_empty())
  {
    current_ptr = s.top();

    // If we are at a leaf or we've already traversed the right node or left
    // node. This predicate, if previous_ptr is any of the left or right
    // children, reproduces the recursive stack calling since we had already
    // visited the children.
    if ((current_ptr->left_ == nullptr && current_ptr->right_ == nullptr) ||
      previous_ptr == current_ptr->right_ || previous_ptr == current_ptr->left_)
    {
      // Deal with topological dependency.
      std::cout << current_ptr->value_ << " ";

      s.pop();

      previous_ptr = current_ptr;
    }
    else
    {
      if (current_ptr->right_ != nullptr)
      {
        s.push(current_ptr->right_);
      }

      if (current_ptr->left_ != nullptr)
      {
        s.push(current_ptr->left_);
      }
    }
  }
}

} // namespace BinaryTrees
} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_BINARY_TREES_BINARY_NODE_H