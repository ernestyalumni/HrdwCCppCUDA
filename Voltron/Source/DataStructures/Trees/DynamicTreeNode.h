#ifndef DATA_STRUCTURES_TREES_DYNAMIC_TREE_NODE_H
#define DATA_STRUCTURES_TREES_DYNAMIC_TREE_NODE_H

#include "DataStructures/Arrays/DynamicArray.h"
#include "DataStructures/LinkedLists/DoubleNode.h"
#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "DataStructures/Queues/DynamicQueue.h"
#include "DataStructures/Stacks/DynamicStack.h"

#include <utility> // std::move

namespace DataStructures
{
namespace Trees
{

template <typename T>
class DynamicTreeNode
{
  public:

    template <typename U>
    using LinkedList = DataStructures::LinkedLists::DoublyLinkedList<U>;

    template <typename V>
    using Node = DataStructures::LinkedLists::Nodes::DoubleNode<V>;

    template <typename W>
    using Queue = DataStructures::Queues::AsHierarchy::DynamicQueue<W>;

    template <typename X>
    using Stack = DataStructures::Stacks::AsHierarchy::DynamicStack<X>;

    template <typename Y>
    using Array = DataStructures::Arrays::PrimitiveDynamicArray<Y>;

    DynamicTreeNode(const T value, DynamicTreeNode* p = nullptr):
      value_{value},
      parent_node_{p},
      children_{}
    {}

    ~DynamicTreeNode()
    {
      if (is_root())
      {
        Queue<DynamicTreeNode*> queue {};

        postorder_traversal_of_nodes_recursive_step(this, queue);

        const std::size_t total_number_of_nodes {queue.size()};

        for (std::size_t i {0}; i < total_number_of_nodes - 1; ++i)
        {
          delete queue.dequeue();
        }
      }
    }

    T value() const
    {
      return value_;
    }

    DynamicTreeNode* parent() const
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
    /// \brief Accessing the nth Child by 0-based indexing.
    /// \ref 4.2.2.2 Accessing the nth Child, 4.02.Abstract_trees.pdf, D.W.
    /// Harder, 2011. ECE 250.
    /// \details O(N) time complexity because for loop.
    //--------------------------------------------------------------------------
    DynamicTreeNode<T>* child(const std::size_t n) const
    {
      if (n >= degree())
      {
        return nullptr;
      }

      const Node<DynamicTreeNode*>* ptr {children_.head()};

      for (std::size_t i {0}; i < n; ++i)
      {
        ptr = ptr->next();
      }

      return ptr->value_;
    }

    void add_child(const T value)
    {
      children_.push_back(new DynamicTreeNode(value, this));
    }

    //--------------------------------------------------------------------------
    /// \ref 4.2.2.3 Attaching and Detaching children, 4.02.Abstract_trees.pdf,
    /// D.W. Harder, 2011, ECE 250
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    /// \details Do nothing if this is root.    
    //--------------------------------------------------------------------------
    void detach_from_parent()
    {
      if (is_root())
      {
        return;
      }

      parent()->children_.list_delete(this);
      parent_node_ = nullptr;
    }

    void attach(DynamicTreeNode<T>* tree)
    {
      // First, if tree we're attaching is attached to a different tree, we must
      // detach it from its parent.
      if (!tree->is_root())
      {
        tree->detach_from_parent();
      }

      tree->parent_node_ = this;
      children_.push_back(tree);
    }

    std::size_t size() const
    {
      std::size_t s {1};

      for (
        const Node<DynamicTreeNode*>* ptr {children_.head()};
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

      for (
        const Node<DynamicTreeNode*>* ptr {children_.head()};
        ptr != nullptr;
        ptr = ptr->next())
      {
        h = std::max(h, 1 + ptr->value_->height());
      }

      return h;
    }

    friend Array<T> breadth_first_traversal(DynamicTreeNode* u)
    {
      Array<T> results {};
      Queue<DynamicTreeNode*> queue {};
      queue.enqueue(u);

      while (!queue.is_empty())
      {
        DynamicTreeNode* v {queue.dequeue()};

        results.append(v->value());

        for (
          const Node<DynamicTreeNode*> ptr {u.children_.head()};
          ptr != nullptr;
          ptr = ptr->next())
        {
          queue.enqueue(ptr->value());
        }
      }

      return std::move(results);
    }

    friend Array<T> preorder_traversal_recursive(DynamicTreeNode* v)
    {
      Array<T> results {};

      preorder_traversal_recursive_step(v, results);

      return std::move(results);
    }

    friend Array<T> postorder_traversal_recursive(DynamicTreeNode* v)
    {
      Array<T> results {};

      postorder_traversal_recursive_step(v, results);

      return std::move(results);
    }

  protected:

    // Interestingly, this function can be called by the user, but we will still
    // mark this as protected to let the developer know that this isn't meant to
    // be part of the interface.

    friend void postorder_traversal_of_nodes_recursive_step(
      DynamicTreeNode* node_ptr,
      Queue<DynamicTreeNode*>& queue)
    {
      if (node_ptr == nullptr)
      {
        return;
      }

      for (std::size_t i {0}; i < node_ptr->degree(); ++i)
      {
        postorder_traversal_of_nodes_recursive_step(node_ptr->child(i), queue);
      }

      queue.enqueue(node_ptr);
    }

    friend void preorder_traversal_recursive_step(
      DynamicTreeNode* v,
      Array<T>& result)
    {
      if (v == nullptr)
      {
        return;
      }

      result.append(v->value());

      // Traverse children next, from "left" child to "right" child.
      for (std::size_t i {0}; i < v->degree(); ++i)
      {
        preorder_traversal_recursive_step(v->child(i), result);
      }
    }

    friend void postorder_traversal_recursive_step(
      DynamicTreeNode* v,
      Array<T>& result)
    {
      if (v == nullptr)
      {
        return;
      }

      // Traverse children next, from "left" child to "right" child.
      for (std::size_t i {0}; i < v->degree(); ++i)
      {
        postorder_traversal_recursive_step(v->child(i), result);
      }

      result.append(v->value());
    }

  private:

    T value_;
    DynamicTreeNode* parent_node_;
    LinkedList<DynamicTreeNode*> children_;
};

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_DYNAMIC_TREE_NODE_H