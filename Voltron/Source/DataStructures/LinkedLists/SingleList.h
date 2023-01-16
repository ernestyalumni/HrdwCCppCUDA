#ifndef DATA_STRUCTURES_LINKED_LISTS_SINGLE_LIST_H
#define DATA_STRUCTURES_LINKED_LISTS_SINGLE_LIST_H

#include "SingleNode.h"

#include <cstddef>
#include <stdexcept>

namespace DataStructures
{
namespace LinkedLists
{
namespace DWHarder
{

//------------------------------------------------------------------------------
/// \url https://ece.uwaterloo.ca/~dwharder/aads/Projects/1/Single_list/src/Single_list.h
//------------------------------------------------------------------------------
template <typename T>
class SingleList
{
  public:

    using Node = DataStructures::LinkedLists::Nodes::SingleNode<T>;

    SingleList();
    SingleList(const SingleList&);
    ~SingleList();

    // Accessors

    std::size_t size() const;

    bool empty() const;

    T front() const;
    T back() const;

    Node* head() const
    {
      return head_;
    }

    Node* tail() const
    {
      return tail_;
    }

    // Mutators.

    void push_front(const T&);

    T pop_front();

    std::size_t erase(const T&);

  private:

    Node* head_;
    Node* tail_;
    std::size_t size_;
};

template <typename T>
SingleList<T>::SingleList():
  head_{nullptr},
  tail_{nullptr},
  size_{0}
{}

template <typename T>
SingleList<T>::SingleList(const SingleList<T>& list):
  head_{nullptr},
  tail_{nullptr},
  size_{0}
{
  if (list.empty())
  {
    return;
  }

  // Copy the first node.
  push_front(list.front());

  for (
    Node* original {list.head()->next_node_},
    *copy {head()};
    original != list.tail();
    // Then we move each pointer forward.
    original = original->next_node_, copy = copy->next_node_)
  {
    copy->next_node_ = new Node(original->retrieve(), nullptr);
  }
}

template <typename T>
SingleList<T>::~SingleList()
{
  while (!empty())
  {
    pop_front();
  }
}

template <typename T>
std::size_t SingleList<T>::size() const
{
  return size_;
}

template <typename T>
bool SingleList<T>::empty() const
{
  return (head_ == nullptr);
}

template <typename T>
T SingleList<T>::front() const
{
  if (empty())
  {
    throw std::runtime_error("Attempted front() on empty LinkedList");
  }

  return head_->retrieve();
}

template <typename T>
T SingleList<T>::back() const
{
  if (empty())
  {
    throw std::runtime_error("Attempted front() on empty LinkedList");
  }

  return tail_->retrieve();
}

template <typename T>
void SingleList<T>::push_front(const T& obj)
{
  const bool no_head {empty()};

  // When list is empty, head_ == nullptr, so this will work in 1 step.
  head_ = new Node(obj, head_);

  if (no_head)
  {
    tail_ = head_;
  }

  ++size_;
}

template <typename T>
T SingleList<T>::pop_front()
{
  if (empty())
  {
    throw std::runtime_error("Failed to pop_front() on empty SingleList");
  }

  const T result {front()};

  Node* head_ptr {head_};

  if (head_->next() == nullptr)
  {
    tail_ = nullptr;
  }

  head_ = head_->next_node_;

  // Given that delete is approximately 100x slower than most other instructions
  // (it does call the OS). cf. Slide 76, Destructor. Waterloo Engineering.
  delete head_ptr;

  --size_;
  return result;
}

template <typename T>
std::size_t SingleList<T>::erase(const T& obj)
{
  std::size_t counter {0};

  if (empty())
  {
    return counter;
  }

  Node* previous_ptr {nullptr};
  Node* ptr {head()};

  while (ptr != nullptr)
  {
    // We're at the front.
    if (ptr->retrieve() == obj && previous_ptr == nullptr)
    {
      pop_front();

      ptr = head_;

      ++counter;
    }
    else if (ptr->retrieve() == obj)
    {
      Node* ptr_to_delete {ptr};

      previous_ptr->next_node_ = ptr->next_;

      ptr = ptr->next();

      delete ptr_to_delete;

      ++counter;
    }
    else
    {
      previous_ptr = ptr;
      ptr = ptr->next();
    }
  }

  return counter;
}


} // namespace DWHarder
} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_SINGLE_LIST_H