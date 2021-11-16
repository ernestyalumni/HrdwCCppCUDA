#ifndef DATA_STRUCTURES_LINKED_LISTS_LINKED_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_LINKED_LIST_H

#include "Node.h"

#include <cstddef>
#include <stdexcept>

namespace DataStructures
{
namespace LinkedLists
{
namespace DWHarder
{

//------------------------------------------------------------------------------
/// \ref 3.05.Linked_lists.pptx
//------------------------------------------------------------------------------
template <typename T>
class LinkedList
{
  public:

    using Node = DataStructures::LinkedLists::Nodes::Node<T>;

    LinkedList():
      head_{nullptr}
    {}

    //--------------------------------------------------------------------------
    /// \brief Copy constructor.
    //--------------------------------------------------------------------------
    LinkedList(const LinkedList&);

    //--------------------------------------------------------------------------
    /// \brief Move constructor.
    /// \ref Slide 119. 3.05.Linked_lists.pptx, D.W. Harder
    /// Move ctor called when an rvalue is being assigned - as an rvalue, it'll
    /// be deleted anyway. The instance ls is being deleted as soon as it's
    /// copied.
    /// Instead, if a move ctor is defined, it'll be called instead of a copy
    /// ctor.
    //--------------------------------------------------------------------------
    LinkedList(LinkedList&& list):
      head_{list.head_}
    {
      list.head_ = nullptr;
    }

    //--------------------------------------------------------------------------
    /// \brief Assignment
    /// \ref Slide 120 3.05.Linked_lists.pptx, D.W. Harder
    /// \details The RHS is passed by value (a copy is made). Return value is
    /// passed by reference.
    /// Note that rhs is a copy of the list, it's not a pointer to a list.
    //--------------------------------------------------------------------------
    LinkedList& operator=(LinkedList rhs)
    {
      // We will swap all the values of the member variables between the left-
      // and right-hand sides.
      // rhs is already a copy, so we swap all member variables of it and *this
      std::swap(*this, rhs);

      // Memory for rhs was allocated on the stack and the dtor will delete it.

      return *this;
    }

    //--------------------------------------------------------------------------
    /// \brief Assignment
    /// \ref Slide 131 3.05.Linked_lists.pptx, D.W. Harder
    //--------------------------------------------------------------------------
    LinkedList& operator=(const LinkedList& rhs)
    {
      if (this == &rhs)
      {
        return *this;
      }

      if (rhs.empty())
      {
        while (!empty())
        {
          pop_front();
        }

        return *this;
      }

      if (empty())
      {
        head_ = new Node(rhs.front());
      }      
      else
      {
        begin()->value_ = rhs.front();
      }

      Node* this_node {head_}, rhs_node = rhs.begin()->next_;

      // 
      while (rhs_node != nullptr)
      {
        // There's no corresponding node; create a new node and append it.
        if (this_node->next_ == nullptr)
        {
          this_node->next_ = new Node(rhs_node->value_);
          this_node = this_node->next_;
        }
        // If there's a corresponding node in this, copy over the value.
        else
        {
          this_node = this_node->next_;
          this_node->value_ = rhs_node->value_;
        }

        rhs_node = rhs_node->next_;
      }

      while (this_node->next_ != nullptr)
      {
        Node* tmp {this_node->next_};
        this_node->next_ = this_node->next_->next_;
        delete tmp;
      }

      return *this;
    }

    //--------------------------------------------------------------------------
    /// \brief Move Assignment
    /// \ref Slide 134 3.05.Linked_lists.pptx, D.W. Harder
    //--------------------------------------------------------------------------
    LinkedList& operator=(LinkedList&& rhs)
    {
      while (!empty())
      {
        pop_front();
      }

      head_ = rhs.begin();
      rhs.head_ = nullptr;

      return *this;
    }

    virtual ~LinkedList()
    {
      while (!empty())
      {
        pop_front();
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Adding the value at the front of the linked list.
    //--------------------------------------------------------------------------
    void push_front(const T value);

    //--------------------------------------------------------------------------
    /// \brief Retrieving the value at the front of the linked list.
    //--------------------------------------------------------------------------
    T front() const;

    //--------------------------------------------------------------------------
    /// \brief Removing the value at the front of the linked list.
    //--------------------------------------------------------------------------
    T pop_front();

    //--------------------------------------------------------------------------
    /// \brief Access the head of the linked list.
    //--------------------------------------------------------------------------
    Node* begin() const;

    //--------------------------------------------------------------------------
    /// \details The member function Node* end() const equals whatever the last
    /// node in the linked list points to - in this case, nullptr.
    /// \ref 3.05.Linked_lists.pptx, D.W. Harder 2001 Waterloo Engineering.
    //--------------------------------------------------------------------------
    Node* end() const;

    //--------------------------------------------------------------------------
    /// \brief Find the number of instances of an integer in the list.
    //--------------------------------------------------------------------------
    std::size_t count(const T value) const;

    //--------------------------------------------------------------------------
    /// \brief Remove all instances of a value from the list.
    //--------------------------------------------------------------------------
    std::size_t erase(const T value);

    //--------------------------------------------------------------------------
    /// \brief Is the linked list empty?
    //--------------------------------------------------------------------------
    bool empty() const;

    //--------------------------------------------------------------------------
    /// \brief How many objecst are in the list?
    /// \details The list is empty when the head pointer is set to nullptr.
    /// O(N) time.
    //--------------------------------------------------------------------------
    std::size_t size() const;

  private:

    Node* head_;
};

template <typename T>
LinkedList<T>::LinkedList(const LinkedList& list):
  head_{nullptr}
{
  if (list.empty())
  {
    return;
  }

  // Copy the first node-we no longer modify head_.
  push_front(list.front());

  // We modify the next pointer of the node pointed to by copy.
  for (
    Node* original {list.begin()->next_}, copy {begin()};
    original != list.end();
    // Then we move each pointer forward.
    original = original->next_, copy = copy->next_)
  {
    copy->next_ = new Node(original->value_, nullptr);
  }

}

template <typename T>
void LinkedList<T>::push_front(const T value)
{
  // When list is empty, head_ == 0, i.e. head_ == nullptr;
  head_ = new Node(value, head_);
}

template <typename T>
T LinkedList<T>::front() const
{
  if (empty())
  {
    throw std::runtime_error("Attempted front() on empty LinkedList");
  }

  return head_->value_;
}

template <typename T>
T LinkedList<T>::pop_front()
{
  if (empty())
  {
    throw std::runtime_error("Attempted pop_front() on empty LinkedList");
  }

  T result {front()};

  // Assign a temporary pointer to point to the node being deleted.

  Node* head_ptr {head_};

  head_ = head_->next_;

  delete head_ptr;

  return result;
}

template <typename T>
bool LinkedList<T>::empty() const
{
  return (head_ == nullptr);
}

template <typename T>
LinkedList<T>::Node* LinkedList<T>::begin() const
{
  return head_;
}

template <typename T>
LinkedList<T>::Node* LinkedList<T>::end() const
{
  return nullptr;
}

template <typename T>
std::size_t LinkedList<T>::count(const T value) const
{
  std::size_t result {0};

  for (Node* ptr {begin()}; ptr != end(); ptr = ptr->next_)
  {
    if (ptr->value_ == value)
    {
      ++result;
    }
  }

  return result;
}

template <typename T>
std::size_t LinkedList<T>::erase(const T value)
{
  std::size_t counter {0};

  if (empty())
  {
    return counter;
  }

  Node* previous_ptr {nullptr};

  Node* ptr {begin()};

  while (ptr != end())
  {
    if (ptr->value_ == value && previous_ptr == nullptr)
    {
      pop_front();

      ptr = head_;

      ++counter;
    }
    else if (ptr->value_ == value)
    {
      Node* ptr_to_delete {ptr};

      previous_ptr->next_ = ptr->next_;

      ptr = ptr->next_;

      delete ptr_to_delete;

      ++counter;
    }
    else
    {
      previous_ptr = ptr;
      ptr = ptr->next_;
    }
  }

  return counter;
}

template <typename T>
std::size_t LinkedList<T>::size() const
{
  std::size_t counter {0};

  for (Node* ptr {begin()}; ptr != end(); ptr = ptr->next_)
  {
    ++counter;
  }

  return counter;
}

} // namespace DWHarder
} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_ARRAY_H