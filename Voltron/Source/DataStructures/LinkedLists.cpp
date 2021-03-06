//------------------------------------------------------------------------------
/// \file LinkedLists.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating LinkedLists.
/// @ref https://gist.github.com/charlierm/5691020
///-----------------------------------------------------------------------------
#include "LinkedLists.h"

// Values of variadic arguments may be accessed using facilities from here.
#include <cstdarg>

#include <cstddef> // std::size_t
#include <initializer_list>

using std::initializer_list;
using std::size_t;

// va_list is a complete object type suitable for holding information needed by
// va_start, va_copy, va_arg, va_end.
using std::va_list;

namespace DataStructures
{
namespace LinkedLists
{

namespace UsingPointers
{

ListNode::ListNode():
  value_{0},
  next_{nullptr}
{}

ListNode::ListNode(int x):
  value_{x},
  next_{nullptr}
{}

ListNode::ListNode(int x, ListNode* next):
  value_{x},
  next_{next}
{}

size_t get_size(ListNode* ln)
{
  size_t counter {0};
  ListNode* ln_ptr {ln};
  while (ln_ptr)
  {
    ln_ptr = ln_ptr->next_;
    ++counter;
  }

  return counter;
}

ListNode* get_tail(ListNode* head)
{
  ListNode* tail {head};
  ListNode* after_tail {head->next_};

  while (after_tail)
  {
    tail = after_tail;
    after_tail = after_tail->next_;
  }

  return tail;
}

// TODO: Implement variadic arguments base case.
// Also consider the warning in one of the solution about using variadic:
// https://stackoverflow.com/questions/1657883/variable-number-of-arguments-in-c
ListNode** setup_ListNode_linked_list(const int values, ...)
{
  va_list args;
  
  // va_start - Enables access to variable arguments.
  // Initializes va_list args to retrieve additional arguments after parameter
  // values. Arguments extracted by subsequent calls to va_arg are those after
  // values.
  va_start(args, values);

  ListNode** list_node_ptr_array = new ListNode*[values];
  ListNode* ln_ptr {nullptr};

  for (size_t counter {0}; counter < values; ++counter)
  {
    const int value {va_arg(args, int)};

    ListNode* list_node_ptr = new ListNode(value);
    list_node_ptr_array[counter] = list_node_ptr;

    if (counter == 0)
    {
      ln_ptr = list_node_ptr;
    }
    else
    {
      ln_ptr->next_ = list_node_ptr;
      ln_ptr = list_node_ptr;
    }
  }

  va_end(args);

  return list_node_ptr_array;
}


ListNode** setup_ListNode_linked_list(const initializer_list<int> l)
{
  ListNode** list_node_ptr_array = new ListNode*[l.size()];
  size_t counter {0};

  ListNode* ln_ptr {nullptr};

  for (int element : l)
  {
    ListNode* list_node_ptr = new ListNode(element);
    list_node_ptr_array[counter] = list_node_ptr;

    if (counter == 0)
    {
      ln_ptr = list_node_ptr;
    }
    else
    {
      ln_ptr->next_ = list_node_ptr;
      ln_ptr = list_node_ptr;
    }

    ++counter;
  }

  return list_node_ptr_array;
}

void clean_up_ListNode_setup(ListNode** l, size_t N)
{
  for (size_t index {0}; index < N; ++index)
  {
    delete l[index];
  }

  delete l;
}


// cf. https://stackoverflow.com/questions/823426/passing-references-to-pointers-in-c
void splice_nodes(ListNode*& ptr1, ListNode*& ptr2)
{
  if (ptr2->next_ != nullptr)
  {
    const int splice_stop_value {ptr2->next_->value_};

    ListNode* splice_stop {ptr1};

    while (true)
    {
      if (splice_stop->next_ == nullptr ||
        splice_stop->next_->value_ > splice_stop_value)
      {
        break;
      }

      splice_stop = splice_stop->next_;
    }

    ListNode* rest_of_list2 {ptr2->next_};
    ListNode* rest_of_list1 {splice_stop->next_};

    // Insert spliced segment.
    splice_stop->next_ = rest_of_list2;
    ptr2->next_ = ptr1;

    // Move pointers "forward".

    // Ok to point to end of list, being nullptr.
    ptr1 = rest_of_list1;
    ptr2 = rest_of_list2;
  }
  // "End of the list" for ptr2.
  else
  {
    ptr2->next_ = ptr1;

    ptr1 = nullptr;

    /*
    while (ptr1 != nullptr)
    {
      ptr1 = ptr1->next_;
    }
    // Clearly ptr1 should be nullptr to break out of this.
    ptr2 = ptr1;
    */
  }
}


ListNode* merge_two_sorted_lists_by_splice(ListNode* l1, ListNode* l2)
{
  if (l1 == nullptr)
  {
    return l2 == nullptr ? nullptr : l2;
  }

  // l1 != nullptr clearly.
  if (l2 == nullptr)
  {
    return l1;
  }

  // l1, l2 points to at least one node each.

  ListNode* returned_list {l1->value_ >= l2->value_ ? l2 : l1};
  ListNode* splice_list {l1->value_ >= l2->value_ ? l1 : l2};

  ListNode* list1_ptr {splice_list};
  ListNode* list2_ptr {returned_list};

  /*
  auto splice_nodes = [](ListNode*& ptr1, ListNode*& ptr2)
  {
    if (ptr2->next_ != nullptr)
    {
      const int splice_stop_value {ptr2->next_->value_};

      ListNode* splice_stop {ptr1};

      while (true)
      {
        if (splice_stop->next_ == nullptr ||
          splice_stop->next_->value_ > splice_stop_value)
        {
          break;
        }

        splice_stop = splice_stop->next_;
      }

      ListNode* rest_of_list2 {ptr2->next_};
      ListNode* rest_of_list1 {splice_stop->next_};

      // Insert spliced segment.
      splice_stop->next_ = rest_of_list2;
      ptr2->next_ = ptr1;

      // Move pointers "forward".

      // Ok to point to end of list, being nullptr.
      ptr1 = rest_of_list1;
      ptr2 = rest_of_list2;
    }
    // "End of the list" for ptr2.
    else
    {
      ptr2->next_ = ptr1;

      while (ptr1 != nullptr)
      {
        ptr1 = ptr1->next_;
      }
      // Clearly ptr1 should be nullptr to break out of this.
      ptr2 = ptr1;
    }
  };
  */

  while (list1_ptr != nullptr)
  {
    splice_nodes(list1_ptr, list2_ptr);
  }

  return returned_list;
}

/// \details Draw a picture to see this.
/// \ref https://leetcode.com/problems/merge-two-sorted-lists/discuss/10065/Clean-simple-O(n%2Bm)-C%2B%2B-Solution-without-dummy-head-and-recurtion
ListNode* merge_two_sorted_lists_iterative(ListNode* l1, ListNode* l2)
{
  if (l1 == nullptr)
  {
    return l2 == nullptr ? nullptr : l2;
  }

  // l1 != nullptr clearly.
  if (l2 == nullptr)
  {
    return l1;
  }

  ListNode* current_ptr {nullptr};

  // Find the first element (can use a dummy node to put this part inside the
  // while loop below.)
  if (l1->value_ <= l2->value_)
  {
    current_ptr = l1;
    l1 = l1->next_;
  }
  else
  {
    current_ptr = l2;
    l2 = l2->next_;
  }

  // Assume ptr1, ptr2 != nullptr
  auto compare_nodes = [](ListNode*& ptr1, ListNode*& ptr2, ListNode*& new_head)
  {
    if (ptr1->value_ <= ptr2->value_)
    {
      new_head->next_ = ptr1;
      ptr1 = ptr1->next_;
    }
    else
    {
      new_head->next_ = ptr2;
      ptr2 = ptr2->next_;
    }
  };

  // See the above initialization.
  //compare_nodes(l1, l2, current_ptr);

  // This is what we'll return.
  ListNode* new_root {current_ptr};

  while (l1 != nullptr && l2 != nullptr)
  {
    compare_nodes(l1, l2, current_ptr);

    // Remember, we had only set the "next" of the current_ptr or the "head",
    // i.e. the "chain" to the next to point to the next value in the sorted
    // sequence. We hadn't moved the current_ptr to point to the new "tail" of
    // our enlengthened string.
    current_ptr = current_ptr->next_;
  }

  if (l1 != nullptr)
  {
    // Then the rest of l1 is sorted.
    current_ptr->next_ = l1;
  }

  if (l2 != nullptr)
  {
    current_ptr->next_ = l2;
  }

  return new_root;
}

// Time Complexity O(m + n), if m = l1 length, n = l2 length. Worst case is
// O(m + n) because need to make m + n comparisons inserting one from each list
// at a time. Best case is if one list is starts bigger than the other so
// attach the "bigger" list at the end e.g. O(n).
// O(1) Space Complexity.
ListNode* merge_two_sorted_lists_simple(ListNode* l1, ListNode* l2)
{
  if (l1 == nullptr)
  {
    return l2;
  }

  if (l2 == nullptr)
  {
    return l1;
  }

  // Thus, both l1, l2 != nullptr.
  ListNode* current_ptr {nullptr};

  // Do the first step.
  if (l1->value_ <= l2->value_)
  {
    current_ptr = l1;
    l1 = l1->next_;
  }
  else
  {
    current_ptr = l2;
    l2 = l2->next_;
  }

  // This is what we'll return, the new "head" or new list's starting element.
  ListNode* new_head {current_ptr};

  while (l1 != nullptr && l2 == nullptr)
  {
    if (l1->value_ <= l2->value_)
    {
      // Connect up the "tail" of the current new list to the next node, l1.
      current_ptr->next_ = l1;
      // Advance ptr l1 since l1's node was chosen.
      l1 = l1->next_;
    }
    else
    {
      current_ptr->next_ = l2;
      l2 = l2->next_;
    }
    // Don't forget that current_ptr must advance and stay at the "tail" of 
    // this new growing list; we only had connected up the nodes and not had
    // advanced.
    current_ptr = current_ptr->next_;
  }

  // Then l2 == nullptr, and so we just connect l1 to the rest of the growing
  // new list, since l1 is already sorted.
  if (l1 != nullptr)
  {
    current_ptr->next_ = l1;
  }

  if (l2 != nullptr)
  {
    current_ptr->next_ = l2;
  }

  return new_head;
}

} // namespace UsingPointers

} // namespace LinkedLists
} // namespace DataStructures
