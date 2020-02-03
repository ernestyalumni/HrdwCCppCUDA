//------------------------------------------------------------------------------
/// \file LinkedLists.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating LinkedLists.
/// @ref https://gist.github.com/charlierm/5691020
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_SORTING_QUICK_SORT_H
#define ALGORITHMS_SORTING_QUICK_SORT_H

#include <iostream>
#include <memory>
#include <stdexcept>

namespace DataStructures
{
namespace LinkedLists
{

template <typename T>
class Element
{
  public:

    explicit Element(T value):
      value_{value},
      next_{nullptr}
    {}

    ~Element() = default;

    T value() const
    {
      return value_;
    }

    void value(T value)
    {
      value_ = value;
    }

    bool has_next() const
    {
      return static_cast<bool>(next_);      
    }

    Element<T>& next()
    {
//      if (!has_next())
//      {
//        throw std::runtime_error("No associated managed object for next");
//      }
      return *next_;      
    }

    /*
    Element<T>* next(Element<T>& element)
    {
      if (has_next())
      {
        Element<T>* old_next = next_.release();
        next_ = std::make_unique<Element<T>>(std::move(element));
        return old_next;
      }

      //next_ = std::make_unique<Element<T>>(element);

      return nullptr;
    }
    */

    void next(Element<T>& element)
    {
      next_.release();
      next_ = std::make_unique<Element<T>>(std::move(element));
    }

  private:

    T value_;
    std::unique_ptr<Element<T>> next_;
};

template <typename T>
struct Node
{
  T value_;
  std::unique_ptr<Node> next_;

  Node(T value) :
    value_{value},
    next_{nullptr}
  {}
};

// https://github.com/sol-prog/cpp17_sllist_smart_pointers/blob/master/SLList_06.cpp
template <typename T>
struct LinkedList
{
  LinkedList() :
    head_{nullptr}
  {}

  void push(T value)
  {
    auto temp {std::make_unique<Node<T>>(value)};
    if (head_)
    {
      temp->next_ = std::move(head_);
      head_ = std::move(temp);
    }
    else
    {
      head_ = std::move(temp);
    }
  }

  void pop()
  {
    if (head_ == nullptr)
    {
      return;
    }

    std::unique_ptr<Node<T>> temp = std::move(head_);
    head_ = std::move(temp->next_);
  }

  void clean()
  {
    while(head_)
    {
      head_ = std::move(head_->next_);
    }
  }

  LinkedList(const LinkedList& linked_list)
  {
    Node<T>* root = linked_list.head_.get();

    std::unique_ptr<Node<T>> new_head {nullptr};

    Node<T>* ptr_new_head {nullptr};

    while(root)
    {
      auto temp {std::make_unique<Node<T>>(root->value_)};
      if (new_head == nullptr)
      {
        new_head = std::move(temp);
        ptr_new_head = new_head.get();
      }
      else
      {
        ptr_new_head->next_ = std::move(temp);
        ptr_new_head = ptr_new_head->next_.get();
      }
      root = root->next_.get();
    }
    head_ = std::move(new_head);
  }

  LinkedList(LinkedList&& linked_list)
  {
    head_ = std::move(linked_list.head_);
  }

  void reverse()
  {
    LinkedList temp;
    Node<T>* root = head_.get();
    while (root)
    {
      temp.push(root->data);
      root = root->next_.get();
    }
    clean();
    head_ = std::move(temp.head_);
  }

  ~LinkedList()
  {
    clean();
  }

  /*
  template <typename U>
  friend std::ostream& operator<<(
    std::ostream& stream,
    const LinkedList<U>& linked_list);
  */

  std::unique_ptr<Node<T>> head_;
};

/*
template <typename T>
std::ostream& LinkedList<T>::operator<<(
  std::ostream& stream,
  const LinkedList<T>& linked_list)
{
  Node<T>* head = linked_list.head_.get();
  while (head)
  {
    stream << head->value_ << ' ';
    head = head->next_.get();
  }
  return stream;
}
*/

} // namespace LinkedLists
} // namespace DataStructures

#endif // ALGORITHMS_SORTING_QUICK_SORT_H