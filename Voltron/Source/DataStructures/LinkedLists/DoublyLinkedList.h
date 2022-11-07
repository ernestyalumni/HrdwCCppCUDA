#ifndef DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_H

#include "DoubleNode.h"

#include <cassert>
#include <utility>

namespace DataStructures
{
namespace LinkedLists
{

//------------------------------------------------------------------------------
/// \ref pp. 236-237 Ch. 10.2 Linked Lists. Cormen, Leiserson, Rivest, and Stein
/// (2009)
//------------------------------------------------------------------------------
template <typename T>
class DoublyLinkedList
{
  public:

    using Node = DataStructures::LinkedLists::Nodes::DoubleNode<T>;

    DoublyLinkedList():
      head_{nullptr},
      tail_{nullptr}
    {}

    // Copy ctor.
    DoublyLinkedList(const DoublyLinkedList& rhs):
      head_{nullptr},
      tail_{nullptr}
    {
      if (rhs.empty())
      {
        return;
      }

      // Copy the first node-we no longer modify head_;
      for (Node* n {rhs.head_}; n; n = n->next_)
      {
        push_back(n->value_);
      }
    }

    // Copy assignment.
    DoublyLinkedList& operator=(const DoublyLinkedList& rhs)
    {
      // Ensure we're not assigning something to itself.
      if (this == &rhs)
      {
        return *this;
      }

      // If the right-hand side is empty, it's straight forward: just empty this
      // list.
      if (rhs.is_empty())
      {
        while (!is_empty())
        {
          pop_front();
        }
        return *this;
      }

      if (is_empty())
      {
        head_ = new Node{rhs.front()};
      }
      else
      {
        begin()->value_ = rhs.front();
      }

      //------------------------------------------------------------------------
      /// Step through the right-hand side list and for each node,
      /// * if there's a corresponding node in this, copy over the value, else
      /// * There's no corresponding node, create a new node and append it.
      //------------------------------------------------------------------------
      Node* this_node {head_};
      Node* rhs_node {rhs.begin()->next_};

      while (rhs_node != nullptr)
      {
        // There's no corresponding node; create a new node and append it.
        if (this_node->next_ == nullptr)
        {
          this_node->next_ = new Node(rhs_node->value_);
          this_node = this_node->next_;
        }
        else
        {
          this_node = this_node->next_;
          this_node->value_ = rhs_node->value_;
        }

        rhs_node = rhs_node->next_;
      }

      // If there are any nodes remaining in this, delete them.
      while (this_node->next_ != nullptr)
      {
        Node* tmp {this_node->next_};
        this_node->next_ = this_node->next_->next_;
        delete tmp;
      }

      return *this;
    }

    virtual ~DoublyLinkedList()
    {
      while (!is_empty())
      {
        pop_front();
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Adding the value at the front of the linked list.
    /// \details If x.previous_ = nullptr, the element x has no predecessor and
    /// is therefore the first element, or head, of the list.
    //--------------------------------------------------------------------------
    void push_front(const T value)
    {
      Node* previous_head {head_};

      head_ = new Node{value, previous_head};

      if (previous_head != nullptr)
      {
        previous_head->previous_ = head_;
      }
      else
      {
        tail_ = head_;
      }
    }

    void push_back(const T value)
    {
      if (is_empty())
      {
        push_front(value);

        return;
      }

      tail_->next_ = new Node{value, nullptr, tail_};

      tail_ = tail_->next_;
    }

    void list_delete(Node* x)
    {
      // If x is not the head,
      if (x->previous_ != nullptr)
      {
        x->previous_->next_ = x->next_;        
      }
      else
      {
        head_ = x->next_;
      }

      // If x is not a tail,
      if (x->next_ != nullptr)
      {
        x->next_->previous_ = x->previous_;
      }
      else
      {
        tail_ = x->previous_;
      }

      delete x;
    }

    void list_delete(const T value)
    {
      Node* x {search(value)};

      if (x != nullptr)
      {
        list_delete(x);
      }
    }

    T pop_front()
    {
      T value {head_->value_};

      /* Explicit deletion of head
      if (head() == tail())
      {
        tail_ = nullptr;
        delete head_;
        return std::move(value);
      }

      Node* previous_head {head_}

      head_ = previous_head->next_;
      head_->previous_ = nullptr;

      delete previous_head;
      */

      list_delete(head_);

      return std::move(value);
    }

    T pop_back()
    {
      T value {tail_->value_};

      list_delete(tail_);

      return std::move(value);
    }

    Node* search(const T value)
    {
      Node* x {head_};

      while (x != nullptr && x->retrieve() != value)
      {
        x = x->next_;
      }

      return x;
    }

    //--------------------------------------------------------------------------
    /// \brief If L.head = nullptr, the list is empty.
    //--------------------------------------------------------------------------
    bool is_empty() const
    {
      return head_ == nullptr;
    }

    Node* head()
    {
      return head_;
    }

    Node* tail()
    {
      return tail_;
    }

    T front() const
    {
      assert(!is_empty());
      return head_->value_;
    }

    //--------------------------------------------------------------------------
    /// \url https://github.com/dkedyk/ImplementingUsefulAlgorithms/blob/master/Utils/GCFreeList.h
    //--------------------------------------------------------------------------
    class Iterator
    {
      public:
        Iterator(Node* n):
          current_{n}
        {}

        Node* get_current()
        {
          return current_;
        }

        // To next item.
        Iterator& operator++()
        {
          assert(current_);
          current_ = current_->next_;
          return *this;
        }

        // To previous item.
        Iterator& operator--()
        {
          assert(current_);
          current_ = current_->previous_;
          return *this;
        }

        T& operator*() const
        {
          assert(current_);
          return current_->value_;
        }

        T* operator->() const
        {
          assert(current_);
          return &current_->value_;
        }

        bool operator==(const Iterator& rhs) const
        {
          return current_ == rhs.current_;
        }

        bool operator!=(const Iterator& rhs) const
        {
          return current_ != rhs.current_;
        }

      private:

        Node* current_;
    };

    Iterator begin()
    {
      return Iterator{head_};
    }

    Iterator rbegin()
    {
      return Iterator{tail_};
    }

    Iterator end()
    {
      return Iterator{nullptr};
    }

    Iterator rend()
    {
      return Iterator{nullptr};
    }

    void move_before(Iterator what, Iterator where)
    {
      assert(what != end());
      // First check for self-reference.
      if (what != where)
      {
        Node* n {what.get_current()};
        Node* w {where.get_current()};

        // Cut Node n out of the linked list.
        assert(n);
        // This "cuts" the "left side link" or previous link of Node n.
        (n == tail_ ? tail_ : n->next_->previous_) = n->previous_;
        // This "cuts" the "right side link" or next link of Node n.
        (n == head_ ? head_ : n->previous_->next_) = n->next_;

        n->next_ = w;
        if (w)
        {
          n->previous_ = w->previous_;
          w->previous_ = n;
        }
        else
        {
          n->previous_ = tail_;
          tail_ = n;
        }
        if (n->previous_)
        {
          n->previous_->next_ = n;
        }
        if (w == head_)
        {
          head_ = n;
        }
      }
    }

  private:

    Node* head_;
    Node* tail_;
};

} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_H