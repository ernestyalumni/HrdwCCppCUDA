//------------------------------------------------------------------------------
/// \file LinkedLists.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating LinkedLists.
/// @ref https://gist.github.com/charlierm/5691020
///-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_LINKED_LISTS_LINKED_LISTS_H
#define DATA_STRUCTURES_LINKED_LISTS_LINKED_LISTS_H

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

//------------------------------------------------------------------------------
/// Nodes as Unique and Shared Poitners.
//------------------------------------------------------------------------------

template <typename T>
struct NodeAsUniquePtr
{
  T value_;
  std::unique_ptr<NodeAsUniquePtr<T>> next_;

  NodeAsUniquePtr(T value) :
    value_{value},
    next_{nullptr}
  {}

  NodeAsUniquePtr(T value, std::unique_ptr<NodeAsUniquePtr<T>>& next) :
    value_{value},
    next_{next.release()}
  {}
};

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/merge-two-sorted-lists/description/
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \brief Definition for singly-linked list.
//------------------------------------------------------------------------------

struct ListNode
{
  int value_;
  ListNode* next_;

  ListNode();

  ListNode(int x);

  ListNode(int x, ListNode* next);
};

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/merge-two-sorted-lists/description/
/// \brief Merge 2 sorted linked lists and return as new sorted list.
/// \details New list should be made by splicing together nodes of first 2
/// lists.
///
/// Solution
/// \ref https://youtu.be/GfRQvf7MB3k
/// Merge 2 Sorted Lists - A Fundamental Merge Sort Subroutine
/// ("Merge Two Sorted Lists" on LeetCode), Back To Back SWE
//------------------------------------------------------------------------------
ListNode* merge_two_sorted_lists_iterative(ListNode* l1, ListNode* l2);

void splice_nodes(ListNode*& ptr1, ListNode*& ptr2);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/merge-two-sorted-lists/description/
/// \brief Merge 2 sorted linked lists and return as new sorted list.
/// \details New list should be made by splicing together nodes of first 2
/// lists.
//------------------------------------------------------------------------------

ListNode* merge_two_sorted_lists_by_splice(ListNode* l1, ListNode* l2);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/merge-two-sorted-lists/discuss/10065/Clean-simple-O(n%2Bm)-C%2B%2B-Solution-without-dummy-head-and-recurtion
/// \url https://youtu.be/GfRQvf7MB3k
/// Zeit Time complexity O(M+N)
/// O(1) space complexity.
//------------------------------------------------------------------------------
ListNode* merge_two_sorted_lists_simple(ListNode* l1, ListNode* l2);

} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_LINKED_LISTS_H