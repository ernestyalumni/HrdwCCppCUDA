#ifndef DATA_STRUCTURES_LINKED_LISTS_LINKED_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_LINKED_LIST_H

#include <cstddef>

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

    class Node;

    LinkedList():
      list_head_{nullptr}
    {}

    class Node
    {
      public:

        Node(const T value = static_cast<T>(0), Node* next_ptr = nullptr);

        virtual ~Node() = default;

        T get_value() const;

        Node* next_ptr() const;

      private:

        T value_;
        Node* next_node_;
    };

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
    void pop_front();

    //--------------------------------------------------------------------------
    /// \brief Access the head of the linked list.
    //--------------------------------------------------------------------------
    Node* begin() const;

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
    /// \details The list is empty when the list_head pointer is set to nullptr.
    //--------------------------------------------------------------------------
    std::size_t size() const;

  private:

    Node* list_head_;
};

template <typename T>
LinkedList<T>::Node::Node(const T value, Node* next_ptr):
  value_{value},
  next_node_{next_ptr}
{}

template <typename T>
T LinkedList<T>::Node::get_value() const
{
  return value_;
}

template <typename T>
LinkedList<T>::Node* LinkedList<T>::Node::next_ptr() const
{
  return next_node_;
}

template <typename T>
bool LinkedList<T>::empty() const
{
  return (list_head_ == nullptr);
}

template <typename T>
LinkedList<T>::Node* LinkedList<T>::begin() const
{
  return nullptr;
}

} // namespace DWHarder
} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_ARRAY_H