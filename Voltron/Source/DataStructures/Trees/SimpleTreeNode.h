#ifndef DATA_STRUCTURES_TREES_SIMPLE_TREE_NODE_H
#define DATA_STRUCTURES_TREES_SIMPLE_TREE_NODE_H

#include <algorithm> // std::max;

#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "DataStructures/LinkedLists/DoubleNode.h"

namespace DataStructures
{
namespace Trees
{

template <typename T>
class SimpleTreeNode
{
  public:

    template <typename U>
    using LinkedList = DataStructures::LinkedLists::DoublyLinkedList<U>;

    template <typename V>
    using Node = DataStructures::LinkedLists::Nodes::DoubleNode<V>;

    SimpleTreeNode(const T& value, SimpleTreeNode* p = nullptr):
      value_{value},
      parent_node_{p},
      children_{}
    {}

    virtual ~SimpleTreeNode() = default;
    /* TODO: determine recursive way to delete nodes
    {
      post_order_deletion(this);
    }
    */

    T value() const
    {
      return value_;
    }

    SimpleTreeNode* parent() const
    {
      return parent_node_;
    }

    bool is_root() const
    {
      return parent() == nullptr;
    }

    std::size_t degree() const
    {
      return children_.size();
    }

    bool is_leaf() const
    {
      return degree() == 0;
    }

    //--------------------------------------------------------------------------
    /// \brief Accessing the nth Child.
    /// \ref 4.2.2.2 Accessing the nth Child, 4.02.Abstract_trees.pdf, D.W.
    /// Harder, 2011. ECE 250.
    /// \details O(N) time complexity because for loop.
    //--------------------------------------------------------------------------
    SimpleTreeNode<T>* child(const std::size_t n) const
    {
      if (n >= degree())
      {
        return nullptr;
      }

      const Node<SimpleTreeNode*>* ptr {children_.head()};

      for (std::size_t i {0}; i < n; ++i)
      {
        ptr = ptr->next();
      }

      return ptr->value_;
    }

    //--------------------------------------------------------------------------
    /// \ref 4.2.2.3 Attaching and Detaching children, 4.02.Abstract_trees.pdf,
    /// D.W. Harder, 2011, ECE 250
    //--------------------------------------------------------------------------

    /* TODO: determine way to delete this child.
    void add_child(const T& value)
    {
      children_.push_back(new SimpleTreeNode(value, this));
    }
    */

    void add_child(SimpleTreeNode* child_ptr)
    {
      children_.push_back(child_ptr);
    }

    //--------------------------------------------------------------------------
    /// \details Do nothing if this is root.
    //--------------------------------------------------------------------------
    void detach_from_parent()
    {
      if (is_root())
      {
        return;
      }

      parent()->children_.erase(this);
      parent_node_ = nullptr;
    }

    void attach(SimpleTreeNode<T>* tree)
    {
      // First, if tree we're attaching is attached to a different tree, we must
      // detach i t from its parent.
      if (!tree->is_root())
      {
        tree->detach_from_parent();
      }

      tree->parent_node_ = this;
      children_.push_back(tree);
    }

    //--------------------------------------------------------------------------
    /// \ref 4.2.2.4 A Recursive Size and Height Member Functions,
    /// 4.02.Abstract_trees.pdf, D.W. Harder, 2011, ECE 250
    //--------------------------------------------------------------------------

    std::size_t size() const
    {
      std::size_t s {1};

      for (Node<SimpleTreeNode*>* ptr {children_.head()};
        ptr != nullptr;
        ptr = ptr->next())
      {
        s += ptr->value_->size();
      }

      return s;
    }

    std::size_t height() const
    {
      std::size_t h {0};

      for (Node<SimpleTreeNode*>* ptr {children_.head()};
        ptr != nullptr;
        ptr = ptr->next())
      {
        h = std::max(h, 1 + ptr->value_->height());
      }

      return h;
    }

    //--------------------------------------------------------------------------
    /// \details For each node within a tree, there's a unique path from root
    /// node to that node. Length of this path is referred to as depth of the
    /// node. Depth of the root node is 0.
    /// \ref 4.2c. Write a recursive member function that finds the depth of a
    /// node. 4.02.Abstract_trees.Questions.pdf, 2013 D.W. Harder. ECE 250.
    //--------------------------------------------------------------------------
    std::size_t depth() const
    {
      if (is_root())
      {
        return 0;
      }

      return 1 + parent()->depth();
    }

    //--------------------------------------------------------------------------
    /// \ref 4.2c. Write an iterative member function that finds the depth of a
    /// node. 4.02.Abstract_trees.Questions.pdf, 2013 D.W. Harder. ECE 250.
    //--------------------------------------------------------------------------
    std::size_t iterative_depth() const
    {
      if (is_root())
      {
        return 0;
      }

      std::size_t depth {0};
      for (
        SimpleTreeNode* current_ptr {parent()};
        current_ptr != nullptr;
        current_ptr = current_ptr->parent())
      {
        depth += 1;
      }

      return depth;
    }

  // protected:

    /* TODO: Determine best way to recursively delete tree.
    void post_order_deletion(SimpleTreeNode* tree_node_ptr)
    {
      if (tree_node_ptr->is_leaf())
      {
        return;
      }

      for (std::size_t i {0}; i < degree(); ++i)
      {
        post_order_deletion(child(i));
      }

      // Visit the current node. This is guaranteed to be a parent of all leaves
      // now because to get to this line of code, the code had to exit the above
      // for loop by visiting a leaf for all children.
      while (!children_.is_empty())
      {
        children_.pop_front();
      }
    }
    */

  private:

    T value_;
    SimpleTreeNode* parent_node_;
    LinkedList<SimpleTreeNode*> children_;
};

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_SIMPLE_TREE_NODE_H