#include "DataStructures/LinkedLists.h"

#include <cstddef> // std::size_t

using DataStructures::LinkedLists::UsingPointers::ListNode;
using DataStructures::LinkedLists::UsingPointers::get_size;
using DataStructures::LinkedLists::UsingPointers::get_tail;

using std::size_t;

namespace Algorithms
{
namespace Sorting
{

namespace InsertionSort
{

ListNode* insertion_sort_list(ListNode* head)
{
  // Time complexity: O(N) (iterates through entire linked list)
  const size_t N {get_size(head)};

  if (N == 0 || N == 1)
  {
    return head;
  }

  if (N == 2)
  {
    const int temp {head->next_->value_};
    if (temp < head->value_)
    {
      ListNode* temp_ptr {head->next_};
      temp_ptr->next_ = head;
      head = temp_ptr;
    }
    return head;
  }


  ListNode* new_head {head};
  ListNode* temp_ptr {head->next_};
  new_head->next_ = nullptr;

  while (temp_ptr)
  {
    const int temp_value {temp_ptr->value_};

    // Move the ptr before the head.
    if (temp_value < new_head->value_)
    {
      ListNode* temp_new_head {temp_ptr};
      // Make sure to fulfill the while loop termination condition.
      temp_ptr = temp_ptr->next_;

      temp_new_head->next_ = new_head;

      new_head = temp_new_head;
    }
    else
    {
      ListNode* sorted_tail {get_tail(new_head)};

      if (sorted_tail->value_ <= temp_value)
      {
        ListNode* temp_ptr_copy {temp_ptr};
        // Make sure to fulfill the while loop termination condition.
        temp_ptr = temp_ptr->next_;

        sorted_tail->next_ = temp_ptr_copy;
        sorted_tail->next_->next_ = nullptr;
      }
      else
      {
        ListNode* iterate_sorted {new_head->next_};
        ListNode* before_iterate_sorted {new_head};

        while (iterate_sorted)
        {
          if (temp_value < iterate_sorted->value_)
          {
            ListNode* temp_ptr_copy {temp_ptr};

            // Make sure to fulfill the while loop termination condition.
            temp_ptr = temp_ptr->next_;

            before_iterate_sorted->next_ = temp_ptr_copy;
            before_iterate_sorted->next_->next_ = iterate_sorted;

            break;
          }

          iterate_sorted = iterate_sorted->next_;
          before_iterate_sorted = before_iterate_sorted->next_;
        }
      }
    }
  }

  return new_head;
}

} // namespace InsertionSort

} // namespace Sorting
} // namespace Algorithms
