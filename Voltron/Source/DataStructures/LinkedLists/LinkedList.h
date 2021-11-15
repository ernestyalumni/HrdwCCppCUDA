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
    //--------------------------------------------------------------------------
    std::size_t size() const;

  private:

    Node* head_;
};

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

  T result {head_->value};

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

} // namespace DWHarder
} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_ARRAY_H