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

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/linked-list/209/singly-linked-list/1287/
//------------------------------------------------------------------------------
template <typename T>
struct SinglyListNode
{
  T value_;
  SinglyListNode* next_;

  SinglyListNode():
    value_{},
    next_{nullptr}
  {}

  explicit SinglyListNode(T value):
    value_{value},
    next_{nullptr}
  {}

  ~SinglyListNode()
  {
    // Recursively delete the rest of the list since it wasn't detached
    // beforehand.
    if (next_ != nullptr)
    {
      delete next_;
    }
  }
};

template <typename T>
class SinglyLinkedList
{
  public:

    SinglyLinkedList():
      front_ptr_{nullptr},
      back_ptr_{nullptr},
      length_{0}
    {}

    ~SinglyLinkedList()
    {
      if (front_ptr_ != nullptr)
      {
        delete front_ptr_;
      }
    }

    T get(const std::size_t index)
    {
      /*if (front_ptr_ == nullptr)
      {
        return -1;
      }

      if (index >= length_)
      {
        return -1;
      }

      SinglyListNode<T>* current_ptr {front_ptr_};

      std::size_t i {0};

      while (current_ptr != nullptr && i < index)
      {
        current_ptr = current_ptr->next_;
        ++i;
      }

      if (current_ptr == nullptr)
      {
        return -1;
      }

      return current_ptr->value_;
      */
      if (index >= length_)
      {
        return -1;
      }

      SinglyListNode<T>* current_ptr {front_ptr_};

      for (std::size_t i {0}; i < index; ++i)
      {
        current_ptr = current_ptr->next_;
      }

      return current_ptr->value_;
    }

    // Add a node of value val before the first element of the linked list.
    // After the insertion, the new node will be the first node of the linked
    // list.
    void add_at_head(T val)
    {
      /*
      if (front_ptr_ == nullptr && back_ptr_ == nullptr)
      {
        front_ptr_ = new SinglyListNode<T>{val};
        back_ptr_ = front_ptr_;

        ++length_;
        return;
      }

      SinglyListNode<T>* new_head {new SinglyListNode<T>{val}};
      new_head->next_ = front_ptr_;
      front_ptr_ = new_head;
      ++length_;
      */

      SinglyListNode<T>* new_head {new SinglyListNode<T>(val)};

      if (front_ptr_ == nullptr)
      {
        front_ptr_ = new_head;

        ++length_;
        return;
      }

      SinglyListNode<T>* current_ptr {front_ptr_};
      front_ptr_ = new_head;
      front_ptr_->next_ = current_ptr;
      ++length_;
    }

    // Append a node of value val to the last element of the linked list.
    void add_at_tail(T val)
    {
      /*
      if (front_ptr_ == nullptr && back_ptr_ == nullptr)
      {
        front_ptr_ = new SinglyListNode<T>{val};
        back_ptr_ = front_ptr_;
        ++length_;
        return;
      }

      SinglyListNode<T>* new_tail {new SinglyListNode<T>{val}};

      if (front_ptr_ == back_ptr_)
      {
        front_ptr_->next_ = new_tail;
        back_ptr_ = new_tail;
      }
      else
      {
        back_ptr_->next_ = new_tail;
        back_ptr_ = new_tail;
      }

      ++length_;
      */

      SinglyListNode<T>* new_tail {new SinglyListNode<T>(val)};

      if (front_ptr_ == nullptr)
      {
        front_ptr_ = new_tail;

        ++length_;
        return;
      }

      SinglyListNode<T>* current_ptr {front_ptr_};

      while (current_ptr->next_ != nullptr)
      {
        current_ptr = current_ptr->next_;
      }

      current_ptr->next_ = new_tail;

      ++length_;
      return;
    }

    // Add a node of value val before the index-th node in the linked list. If
    // index equals to the length of the linked list, the node will be appended
    // to the end of linked list.
    void add_at_index(const std::size_t index, T val)
    {
      /*
      if (length_ < index)
      {
        return;
      }

      if (length_ == index)
      {
        add_at_tail(val);
        return;
      }

      if (index == 0)
      {
        add_at_head(val);
        return;
      }

      SinglyListNode<T>* new_node {new SinglyListNode<T>{val}};

      SinglyListNode<T>* current_ptr {front_ptr_};

      std::size_t i {0};

      while (i < (index- 1))
      {
        current_ptr = current_ptr->next_;
        ++i;
      }

      SinglyListNode<T>* rest_of_list {current_ptr->next_};
      new_node->next_ = rest_of_list;
      current_ptr->next_ = new_node;
      ++length_;
      */

      if (index > length_)
      {
        return;
      }

      if (length_ == index)
      {
        add_at_tail(val);
        return;
      }

      if (index == 0)
      {
        add_at_head(val);
        return;
      }

      SinglyListNode<T>* new_node {new SinglyListNode<T>{val}};

      SinglyListNode<T>* current_ptr {front_ptr_};     

      for (std::size_t i {0}; i < (index- 1); ++i)
      {
        current_ptr = current_ptr->next_;
      }

      SinglyListNode<T>* rest_of_list {current_ptr->next_};
      current_ptr->next_ = new_node;
      current_ptr->next_->next_ = rest_of_list;
      ++length_;
    }

    // Delete the index-th node in the linked list, if the index is valid.
    void delete_at_index(const std::size_t index)
    {
      /*
      if (index >= length_)
      {
        return;
      }

      if (index == 0)
      {
        SinglyListNode<T>* rest_of_list {front_ptr_->next_};
        front_ptr_->next_ = nullptr;
        delete front_ptr_;
        front_ptr_ = rest_of_list;
        --length_;
        return;
      }

      SinglyListNode<T>* current_ptr {front_ptr_};

      std::size_t i {0};

      while (i < (index - 1))
      {
        current_ptr = current_ptr->next_;
        ++i;
      }

      SinglyListNode<T>* previous_ptr {current_ptr};
      current_ptr = current_ptr->next_;
      previous_ptr->next_ = nullptr;
      SinglyListNode<T>* rest_of_list {current_ptr->next_};
      current_ptr->next_ = nullptr;
      delete current_ptr;
      previous_ptr->next_ = rest_of_list;

      --length_;
      */
      if (index >= length_)
      {
        return;
      }

      if (index == 0)
      {
        SinglyListNode<T>* rest_of_list {front_ptr_->next_};
        front_ptr_->next_ = nullptr;
        delete front_ptr_;
        front_ptr_ = rest_of_list;
        --length_;
        return;
      }

      SinglyListNode<T>* current_ptr {front_ptr_};     

      for (std::size_t i {0}; i < (index- 1); ++i)
      {
        current_ptr = current_ptr->next_;
      }

      // Includes element at position index.
      SinglyListNode<T>* rest_of_list {current_ptr->next_};
      current_ptr->next_ = rest_of_list->next_;
      
      rest_of_list->next_ = nullptr;
      --length_;
      // Free the memory of element at position index.
      delete rest_of_list;
    }

    SinglyListNode<T>* front_ptr()
    {
      return front_ptr_;
    }

    SinglyListNode<T>* back_ptr()
    {
      return back_ptr_;
    }

    std::size_t length() const
    {
      return length_;
    }

  private:

    SinglyListNode<T>* front_ptr_;
    SinglyListNode<T>* back_ptr_;
    std::size_t length_;
};

} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_LINKED_LISTS_H