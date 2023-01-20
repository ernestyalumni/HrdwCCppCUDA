#ifndef DATA_STRUCTURES_TREES_SIMPLE_TREE_NODE_H
#define DATA_STRUCTURES_TREES_SIMPLE_TREE_NODE_H

#include <algorithm> // std::max;
#include <cstddef> // std::size_t
#include <iostream>
#include <vector>

#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "DataStructures/LinkedLists/DoubleNode.h"
#include "DataStructures/Queues/DynamicQueue.h"
#include "DataStructures/Stacks/DynamicStack.h"

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

    template <typename W>
    using Queue = DataStructures::Queues::AsHierarchy::DynamicQueue<W>;

    template <typename X>
    using Stack = DataStructures::Stacks::AsHierarchy::DynamicStack<X>;

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

      parent()->children_.list_delete(this);
      parent_node_ = nullptr;
    }

    void attach(SimpleTreeNode<T>* tree)
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

    //--------------------------------------------------------------------------
    /// \ref 3.03.Queues.pdf, 2011 D.W. Harder, ECE 250. 3.3.5 Applications.
    /// \return nullptr if the value is not found.
    //--------------------------------------------------------------------------
    SimpleTreeNode* breadth_first_search(const T value)
    {
      Queue<SimpleTreeNode*> queue {};

      queue.enqueue(this);

      while (!queue.is_empty())
      {
        SimpleTreeNode* v {queue.dequeue()};

        if (v->value() == value)
        {
          return v;
        }

        for (auto* ptr {v->children_.head()}; ptr != nullptr; ptr = ptr->next())
        {
          queue.enqueue(ptr->get_value());
        }
        // Alternatively, we could've implemented this:
        //for (std::size_t i {0}; i != v->degree(); ++i)
        //{
        //  queue.enqueue(v->child(i));
        //}
      }

      // Exited due to queue being empty.
      return nullptr;
    }    

    void breadth_first_traversal()
    {
      Queue<SimpleTreeNode*> queue {};

      queue.enqueue(this);

      while (!queue.is_empty())
      {
        SimpleTreeNode* v {queue.dequeue()};

        std::cout << v->value() << ", ";

        for (auto* ptr {v->children_.head()}; ptr != nullptr; ptr = ptr->next())
        {
          queue.enqueue(ptr->get_value());
        }
      }
    }

    friend std::vector<T> preorder_traversal(SimpleTreeNode* v)
    {
      std::vector<T> result {};

      if (v == nullptr)
      {
        return result;
      }

      Stack<SimpleTreeNode*> tree_node_ptr_stack {};
      tree_node_ptr_stack.push(v);

      while (!tree_node_ptr_stack.is_empty())
      {
        SimpleTreeNode* current_ptr {tree_node_ptr_stack.pop()};

        // Effectively visit the current node first.
        result.emplace_back(current_ptr->value());

        if (!current_ptr->is_leaf())
        {
          // Push the last child, i.e. push children from "right" to "left" so
          // that, due to the properties of a stack, "left" is processed before
          // the "right" child.
          for (
            std::size_t ith_child {current_ptr->degree()};
            ith_child != 0;
            --ith_child)
          {
            // Calling child is zero-based indexing.
            tree_node_ptr_stack.push(current_ptr->child(ith_child - 1));
          }
        }
      }

      return result;
    }

    friend std::vector<T> postorder_traversal_recursive(SimpleTreeNode* v)
    {
      std::vector<T> result {};
      postorder_traversal_recursive_step(v, result);
      return result;
    }

  protected:

    friend void postorder_traversal_recursive_step(
      SimpleTreeNode* node_ptr,
      std::vector<T>& result)
    {
      if (node_ptr == nullptr)
      {
        return;
      }

      // Traverse all the children first, from "left" child to "right" child.
      for (std::size_t i {0}; i < node_ptr->degree(); ++i)
      {
        postorder_traversal_recursive_step(node_ptr->child(i), result);
      }

      // Come back here. We know that the nod_ptr is nonempty from above.
      result.emplace_back(node_ptr->value());
    }

    // Interestingly, this function can be called by the user, but we will still
    // mark this as protected to let the developer know that this isn't meant to
    // be part of the interface.
    friend void postorder_traversal_of_nodes_recursive_step(
      SimpleTreeNode* node_ptr,
      Queue<SimpleTreeNode*>& queue)
    {
      if (node_ptr == nullptr)
      {
        return;
      }

      for (
        std::size_t ith_child {1};
        ith_child <= node_ptr->degree();
        ++ith_child)
      {
        postorder_traversal_of_nodes_recursive_step(
          node_ptr->child(ith_child - 1),
          queue);
      }

      queue.enqueue(node_ptr);
    }

  private:

    T value_;
    SimpleTreeNode* parent_node_;
    LinkedList<SimpleTreeNode*> children_;
};

} // namespace Trees
} // namespace DataStructures

#endif // DATA_STRUCTURES_TREES_SIMPLE_TREE_NODE_H