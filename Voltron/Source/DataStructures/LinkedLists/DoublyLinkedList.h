#ifndef DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_H

#include "DoubleNode.h"

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

  private:

    Node* head_;
    Node* tail_;
};

} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_H