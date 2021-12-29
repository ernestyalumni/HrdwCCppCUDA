#ifndef DATA_STRUCTURES_LINKED_LISTS_SENTINEL_DOUBLY_LINKED_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_SENTINEL_DOUBLY_LINKED_LIST_H

#include <utility>

namespace DataStructures
{
namespace LinkedLists
{
namespace CLRS
{

//------------------------------------------------------------------------------
/// \ref pp. 238-239 Ch. 10.2 Linked Lists. Cormen, Leiserson, Rivest, and Stein
/// (2009)
//------------------------------------------------------------------------------
template <typename T>
class SentinelDoublyLinkedList
{
  public:

    class Node
    {
      public:

        Node(
          const T& element = T{},
          Node* next_node = nullptr,
          Node* previous_node = nullptr
          ):
          value_{element},
          next_{next_node},
          previous_{previous_node}
        {}

        virtual ~Node() = default;

        T value_;
        Node* next_;
        Node* previous_;
    };

    SentinelDoublyLinkedList():
      sentinel_{new Node{T{}}}
    {
      sentinel_->next_ = sentinel_;
      sentinel_->previous_ = sentinel_;
    }

    virtual ~SentinelDoublyLinkedList()
    {
      while(!is_empty())
      {
        pop_front();
      }

      delete sentinel_;
    }

    void push_front(const T& value)
    {
      Node* x {new Node(value, sentinel_->next_, sentinel_)};

      sentinel_->next_->previous_ = x;
      sentinel_->next_ = x;
    }

    void list_delete(Node* x)
    {
      x->previous_->next_ = x->next_;
      x->next_->previous_ = x->previous_;

      delete x;
    }

    void list_delete(const T& value)
    {
      Node* x {search(value)};

      if (!is_sentinel(x))
      {
        list_delete(x);
      }
    }

    T pop_front()
    {
      T value {head()->value_};

      /* Replace the following lines with list_delete.
      x->previous_->next_ = x->next_;
      x->next_->previous_ = x->previous_;

      delete x;
      */
      list_delete(head());

      return std::move(value);
    }

    Node* head()
    {
      return sentinel_->next_;
    }

    Node* tail()
    {
      return sentinel_->previous_;
    }

    bool is_empty() const
    {
      return (sentinel_->next_ == sentinel_ &&
        sentinel_->previous_ == sentinel_);
    }

    bool is_sentinel(const Node* node_ptr) const
    {
      return node_ptr == sentinel_;
    }

    //--------------------------------------------------------------------------
    /// \details O(N) time.
    //--------------------------------------------------------------------------
    Node* search(const T& key)
    {
      Node* x {sentinel_->next_};

      while (x != sentinel_ && x->value_ != key)
      {
        x = x->next_;
      }

      return x;
    }

  private:

    Node* sentinel_;
};

} // namespace CLRS
} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_SENTINEL_DOUBLY_LINKED_LIST_H