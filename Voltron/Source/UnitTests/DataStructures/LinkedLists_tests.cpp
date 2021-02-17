//------------------------------------------------------------------------------
/// \file LinkedLists_tests.cpp
//------------------------------------------------------------------------------
#include "DataStructures/LinkedLists.h"

#include <array>
#include <boost/test/unit_test.hpp>
#include <deque>
#include <forward_list>
#include <list>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using DataStructures::LinkedLists::Element;
using DataStructures::LinkedLists::LinkedList;
using DataStructures::LinkedLists::Node;
using DataStructures::LinkedLists::NodeAsUniquePtr;
using DataStructures::LinkedLists::SinglyLinkedList;
using DataStructures::LinkedLists::SinglyListNode;
using DataStructures::LinkedLists::UsingPointers::ListNode;
using DataStructures::LinkedLists::UsingPointers::clean_up_ListNode_setup;
using DataStructures::LinkedLists::UsingPointers::get_size;
using DataStructures::LinkedLists::UsingPointers::get_tail;
using DataStructures::LinkedLists::UsingPointers::
  merge_two_sorted_lists_by_splice;
using DataStructures::LinkedLists::UsingPointers::
  merge_two_sorted_lists_iterative;
using DataStructures::LinkedLists::UsingPointers::merge_two_sorted_lists_simple;
using DataStructures::LinkedLists::UsingPointers::setup_ListNode_linked_list;
using DataStructures::LinkedLists::UsingPointers::splice_nodes;
using std::make_unique;
using std::unique_ptr;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ElementIsNotDefaultConstructor)
{
  BOOST_TEST(!std::is_default_constructible<Element<int>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ElementConstructs)
{
  {
    Element<int> int_element {42};
    BOOST_TEST(int_element.value() == 42);
    // fatal error: memory access violation at address: 0x00000000: no mapping
    // at fault address
    // BOOST_TEST(int_element.next());
    BOOST_TEST(!int_element.has_next());
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ElementSetValueChangesValue)
{
  {
    Element<int> int_element {42};
    BOOST_TEST(int_element.value() == 42);
    int_element.value(69);
    BOOST_TEST(int_element.value() == 69);    
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ElementSetNextSwapsElements)
{
  {
    Element<int> int_element {42};
    Element<int> new_int_element {69};

//    auto old_next = int_element.next(new_int_element);    

    //int_element.next(new_int_element);
//    BOOST_TEST(int_element.has_next());

    //BOOST_TEST(int_element.next().value() == 69);
    //BOOST_TEST(!int_element.next().has_next());    
  }
}

// cf. https://github.com/sol-prog/cpp17_sllist_smart_pointers/blob/master/SLList_06.cpp
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateLinkedList)
{
  {
    LinkedList<int> linked_list; 

    for (int i {0}; i < 10; ++i)
    {
      linked_list.push(i);
    }

    BOOST_TEST(linked_list.head_->value_ == 9);
    BOOST_TEST(linked_list.head_->next_->value_ == 8);
    BOOST_TEST(linked_list.head_->next_->next_->value_ == 7);
    BOOST_TEST(linked_list.head_->next_->next_->next_->value_ == 6);
  }
}

BOOST_AUTO_TEST_SUITE(NodeAsUniquePtr_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithIntegerAndAnotherUniquePtr)
{
  unique_ptr<NodeAsUniquePtr<int>> leaf_ptr {
    make_unique<NodeAsUniquePtr<int>>(69)};
  NodeAsUniquePtr<int> node {42, leaf_ptr};

  BOOST_TEST(!static_cast<bool>(leaf_ptr));
  BOOST_TEST(node.value_ == 42);
  BOOST_TEST(node.next_->value_ == 69);
}

BOOST_AUTO_TEST_SUITE_END() // NodeAsUniquePtr_tests

BOOST_AUTO_TEST_SUITE(ListNode_tests)

/// \url https://leetcode.com/problems/merge-two-sorted-lists/description/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  ListNode node {};
  BOOST_TEST(node.value_ == 0);
  BOOST_TEST(node.next_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInteger)
{
  ListNode node {42};
  BOOST_TEST(node.value_ == 42);
  BOOST_TEST(node.next_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithIntegerAndNode)
{
  ListNode leaf {42};
  ListNode node {69, &leaf};
  BOOST_TEST(node.value_ == 69);
  BOOST_TEST(node.next_->value_ == 42);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CanConstructLinkedList)
{
  ListNode l11 {1};
  ListNode l12 {2};
  ListNode l13 {4};

  l11.next_ = &l12;
  l12.next_ = &l13;

  BOOST_TEST(l11.value_ == 1);
  BOOST_TEST(l11.next_->value_ == 2);
  BOOST_TEST(l11.next_->next_->value_ == 4);
}

ListNode l11 {1};
ListNode l12 {2};
ListNode l13 {4};

//l11.next_ = &l12;
//l12.next_ = &l13;

ListNode l21 {1};
ListNode l22 {3};
ListNode l23 {4};

//l21.next_ = &l22;
//l22.next_ = &l23;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CanCreatePointersToMoveAcrossLinkedListAndSplice)
{
  l11.next_ = &l12;
  l12.next_ = &l13;
  l21.next_ = &l22;
  l22.next_ = &l23;
  BOOST_TEST(l11.value_ == 1);
  BOOST_TEST(l11.next_->value_ == 2);
  BOOST_TEST(l11.next_->next_->value_ == 4);
  BOOST_TEST(l21.value_ == 1);
  BOOST_TEST(l21.next_->value_ == 3);
  BOOST_TEST(l21.next_->next_->value_ == 4);

  ListNode* l11_ptr {&l11};
  ListNode* l21_ptr {&l21};

  ListNode* list1_ptr {l11_ptr};
  ListNode* list2_ptr {l21_ptr};

  ListNode* rest_of_list2 {list2_ptr->next_};
  ListNode* rest_of_list1 {list1_ptr->next_};
  list1_ptr->next_ = rest_of_list2;
  list2_ptr->next_ = list1_ptr;

  list2_ptr = list2_ptr->next_;
  list1_ptr = rest_of_list1;


  BOOST_TEST(rest_of_list1->value_ == 2);
  BOOST_TEST(rest_of_list1->next_->value_ == 4);

  BOOST_TEST(l21.value_ == 1);
  BOOST_TEST(l21.next_->value_ == 1);
  BOOST_TEST(l21.next_->next_->value_ == 3);
  BOOST_TEST(l21.next_->next_->next_->value_ == 4);
}

BOOST_AUTO_TEST_SUITE(SetupListNode_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InitializerListSetsUpListNodes)
{
  ListNode** lnl {setup_ListNode_linked_list({2, 6, 4})};
  BOOST_TEST(lnl[0]->value_ == 2);
  BOOST_TEST(lnl[1]->value_ == 6);
  BOOST_TEST(lnl[2]->value_ == 4);

  BOOST_TEST(lnl[0]->next_->value_ == 6);
  BOOST_TEST(lnl[0]->next_->next_->value_ == 4);

  delete lnl[0];
  delete lnl[1];
  delete lnl[2];
  delete lnl;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CleanUpListNodeSetupDeletesDynamicallyAllocatedNodes)
{
  ListNode** lnl {setup_ListNode_linked_list({5, 3, 7, 8})};

  clean_up_ListNode_setup(lnl, 4);

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(VariadicArgumentsSetsUpListNodes)
{
  ListNode** lnl {setup_ListNode_linked_list(4, 4, 2, 1, 3)};
  BOOST_TEST(lnl[0]->value_ == 4);
  BOOST_TEST(lnl[1]->value_ == 2);
  BOOST_TEST(lnl[2]->value_ == 1);
  BOOST_TEST(lnl[3]->value_ == 3);

  BOOST_TEST(lnl[0]->next_->value_ == 2);
  BOOST_TEST(lnl[0]->next_->next_->value_ == 1);
  BOOST_TEST(lnl[0]->next_->next_->next_->value_ == 3);
  BOOST_TEST(lnl[0]->next_->next_->next_->next_ == nullptr);

  clean_up_ListNode_setup(lnl, 4);
}

BOOST_AUTO_TEST_SUITE_END() // SetupListNode_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetSizeGetsSize)
{
  ListNode** lnl {setup_ListNode_linked_list({5, 3, 7, 8, -23})};

  BOOST_TEST(get_size(lnl[0]) == 5);
  BOOST_TEST(get_size(lnl[1]) == 4);
  BOOST_TEST(get_size(lnl[3]) == 2);

  clean_up_ListNode_setup(lnl, 5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetTailGetsTail)
{
  ListNode** lnl {setup_ListNode_linked_list({5, 3, 7, 8, -23})};

  ListNode* tail {get_tail(lnl[0])};

  BOOST_TEST(tail->next_ == nullptr);
  BOOST_TEST(tail->value_ == -23);

  clean_up_ListNode_setup(lnl, 5);
}

BOOST_AUTO_TEST_SUITE(MergeTwoSortedLists_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnsNullPtrForBothNullPtrInputs)
{
  ListNode* l1_ptr {nullptr};
  ListNode* l2_ptr {nullptr};

  //ListNode l1 {2};
  //ListNode l2 {1};
  //ListNode* l1_ptr {&l1};
  //ListNode* l2_ptr {&l2};

  ListNode* result = merge_two_sorted_lists_by_splice(l1_ptr, l2_ptr);

  BOOST_TEST(result == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnsOtherListForNullPtr)
{
  {
    ListNode l1 {2};

    ListNode* l1_ptr {&l1};
    ListNode* l2_ptr {nullptr};

    ListNode* result = merge_two_sorted_lists_by_splice(l1_ptr, l2_ptr);

    BOOST_TEST(result->value_ == 2);
  }
  {
    ListNode l2 {5};

    ListNode* l1_ptr {nullptr};
    ListNode* l2_ptr {&l2};

    ListNode* result = merge_two_sorted_lists_by_splice(l1_ptr, l2_ptr);

    BOOST_TEST(result->value_ == 5);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SpliceNodesSplicesNodes)
{
  l11.next_ = &l12;
  l12.next_ = &l13;
  l21.next_ = &l22;
  l22.next_ = &l23;

  ListNode* l11_ptr {&l11};
  ListNode* l21_ptr {&l21};

  splice_nodes(l11_ptr, l21_ptr);

  BOOST_TEST(l11_ptr->value_ == 4);
  BOOST_TEST(l11.value_ == 1);
  BOOST_TEST(l21_ptr->value_ == 3);
  BOOST_TEST(l21.value_ == 1);
  BOOST_TEST(l21.next_->value_ == 1);
  BOOST_TEST(l21.next_->next_->value_ == 2);
  BOOST_TEST(l21.next_->next_->next_->value_ == 3);

  splice_nodes(l11_ptr, l21_ptr);
  BOOST_TEST(l11_ptr == nullptr);
  BOOST_TEST(l21_ptr != nullptr);
  BOOST_TEST(l21_ptr->value_ == 4);
  BOOST_TEST(l21.value_ == 1);
  BOOST_TEST(l21.next_->value_ == 1);
  BOOST_TEST(l21.next_->next_->value_ == 2);
  BOOST_TEST(l21.next_->next_->next_->value_ == 3);
  BOOST_TEST(l21.next_->next_->next_->next_->value_ == 4);
  BOOST_TEST(l21.next_->next_->next_->next_->next_->value_ == 4);
  BOOST_TEST(l21.next_->next_->next_->next_->next_->next_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnsSortedListWithMergedSortedListWithSplicing)
{
  l11.next_ = &l12;
  l12.next_ = &l13;
  l21.next_ = &l22;
  l22.next_ = &l23;

  // Need to "cut off" the end of the input once again because the node object
  // had been directly transformed itself.
  if (l13.next_ != nullptr)
  {
    l13.next_ = nullptr;
  }

  if (l23.next_ != nullptr)
  {
    l23.next_ = nullptr;
  }

  ListNode* l11_ptr {&l11};
  ListNode* l21_ptr {&l21};

  // Confirm that we have the original inputs.
  BOOST_TEST(l11.value_ == 1);
  BOOST_TEST(l11.next_->value_ == 2);
  BOOST_TEST(l11.next_->next_->value_ == 4);
  BOOST_TEST(l11.next_->next_->next_ == nullptr);
  BOOST_TEST(l21.value_ == 1);
  BOOST_TEST(l21.next_->value_ == 3);
  BOOST_TEST(l21.next_->next_->value_ == 4);
  BOOST_TEST(l21.next_->next_->next_ == nullptr);

  ListNode* merged_list {merge_two_sorted_lists_by_splice(l11_ptr, l21_ptr)}; 

  BOOST_TEST(merged_list->value_ == 1);
  BOOST_TEST(merged_list->next_->value_ == 1);
  BOOST_TEST(merged_list->next_->next_->value_ == 2);
  BOOST_TEST(merged_list->next_->next_->next_->value_ == 3);
  BOOST_TEST(merged_list->next_->next_->next_->next_->value_ == 4);
  BOOST_TEST(merged_list->next_->next_->next_->next_->next_->value_ == 4);
  // TODO: Remove after fixing. This is probably because the unit test case was
  // changed from above.
  BOOST_TEST(merged_list->next_->next_->next_->next_->next_->next_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReturnsSortedListWithMergedSortedListIterative)
{
  l11.next_ = &l12;
  l12.next_ = &l13;
  l21.next_ = &l22;
  l22.next_ = &l23;

  // Need to "cut off" the end of the input once again because the node object
  // had been directly transformed itself.
  if (l13.next_ != nullptr)
  {
    l13.next_ = nullptr;
  }

  if (l23.next_ != nullptr)
  {
    l23.next_ = nullptr;
  }

  ListNode* l11_ptr {&l11};
  ListNode* l21_ptr {&l21};

  // Confirm that we have the original inputs.
  BOOST_TEST(l11.value_ == 1);
  BOOST_TEST(l11.next_->value_ == 2);
  BOOST_TEST(l11.next_->next_->value_ == 4);
  BOOST_TEST(l11.next_->next_->next_ == nullptr);

  BOOST_TEST(l21.value_ == 1);
  BOOST_TEST(l21.next_->value_ == 3);
  BOOST_TEST(l21.next_->next_->value_ == 4);
  BOOST_TEST(l21.next_->next_->next_ == nullptr);
  
  ListNode* merged_list {merge_two_sorted_lists_iterative(l11_ptr, l21_ptr)}; 

  BOOST_TEST(merged_list->value_ == 1);
  BOOST_TEST(merged_list->next_->value_ == 1);
  BOOST_TEST(merged_list->next_->next_->value_ == 2);
  BOOST_TEST(merged_list->next_->next_->next_->value_ == 3);
  BOOST_TEST(merged_list->next_->next_->next_->next_->value_ == 4);
  BOOST_TEST(merged_list->next_->next_->next_->next_->next_->value_ == 4);
  // TODO: Remove after fixing. This is probably because the unit test case was
  // changed from above.
  BOOST_TEST(merged_list->next_->next_->next_->next_->next_->next_ == nullptr);
}

BOOST_AUTO_TEST_SUITE_END() // MergeTwoSortedLists_tests

BOOST_AUTO_TEST_SUITE_END() // ListNode_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SinglyLinkedListDefaultConstructs)
{
  SinglyLinkedList<int> list;

  BOOST_TEST(list.length() == 0);
  BOOST_TEST(list.front_ptr() == nullptr);
//  BOOST_TEST(list.back_ptr() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SinglyLinkedListAddsToHeadWhenEmpty)
{
  SinglyLinkedList<int> list;

  BOOST_TEST(list.length() == 0);
  BOOST_TEST(list.front_ptr() == nullptr);
//  BOOST_TEST(list.back_ptr() == nullptr);

  list.add_at_head(42);

  BOOST_TEST(list.length() == 1);
  BOOST_TEST(list.front_ptr()->value_ == 42);
//  BOOST_TEST(list.back_ptr()->value_ == 42);
  BOOST_TEST(list.front_ptr()->next_ == nullptr);
//  BOOST_TEST(list.back_ptr()->next_ == nullptr);
}

/*

=================================================================
==29==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000998 at pc 0x0000003b5680 bp 0x7ffcf54806b0 sp 0x7ffcf54806a8
WRITE of size 8 at 0x602000000998 thread T0
    #4 0x7f45bbcd40b2  (/lib/x86_64-linux-gnu/libc.so.6+0x270b2)
0x602000000998 is located 8 bytes inside of 16-byte region [0x602000000990,0x6020000009a0)
freed by thread T0 here:
    #5 0x7f45bbcd40b2  (/lib/x86_64-linux-gnu/libc.so.6+0x270b2)
previously allocated by thread T0 here:
    #5 0x7f45bbcd40b2  (/lib/x86_64-linux-gnu/libc.so.6+0x270b2)
Shadow bytes around the buggy address:
  0x0c047fff80e0: fa fa 00 00 fa fa fd fa fa fa fd fa fa fa fd fa
  0x0c047fff80f0: fa fa 00 00 fa fa fd fd fa fa 00 00 fa fa 00 00
  0x0c047fff8100: fa fa 00 00 fa fa 00 00 fa fa fd fd fa fa fd fa
  0x0c047fff8110: fa fa fd fa fa fa fd fa fa fa 00 00 fa fa 00 00
  0x0c047fff8120: fa fa 00 00 fa fa 00 00 fa fa 00 00 fa fa fd fd
=>0x0c047fff8130: fa fa fd[fd]fa fa 00 00 fa fa 00 00 fa fa fa fa
  0x0c047fff8140: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff8150: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff8160: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff8170: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff8180: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07 
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
  Shadow gap:              cc
==29==ABORTING

["MyLinkedList","addAtHead","addAtIndex","addAtTail","addAtHead","addAtIndex",
"addAtTail","addAtTail","addAtIndex","deleteAtIndex","deleteAtIndex","addAtTail"]
[[],[0],[1,4],[8],[5],[4,3],[0],[5],[6,3],[7],[5],[4]]

*/

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SinglyLinkedListCleansUpMemory)
{
  SinglyLinkedList<int> list;

  list.add_at_head(0);
  BOOST_TEST(list.length() == 1);
  BOOST_TEST(list.front_ptr()->value_ == 0);
  BOOST_TEST(list.front_ptr()->next_ == nullptr);
//  BOOST_TEST(list.back_ptr()->value_ == 0);
//  BOOST_TEST(list.back_ptr()->next_ == nullptr);

  list.add_at_index(1, 4);
  BOOST_TEST(list.length() == 2);
  BOOST_TEST(list.front_ptr()->value_ == 0);
  BOOST_TEST(list.front_ptr()->next_ != nullptr);
//  BOOST_TEST(list.front_ptr()->next_ == list.back_ptr());
//  BOOST_TEST(list.back_ptr()->value_ == 4);
//  BOOST_TEST(list.back_ptr()->next_ == nullptr);

  list.add_at_tail(8);
  BOOST_TEST(list.length() == 3);
//  BOOST_TEST(list.back_ptr()->value_ == 8);
//  BOOST_TEST(list.back_ptr()->next_ == nullptr);
  BOOST_TEST(list.front_ptr()->value_ == 0);
  BOOST_TEST(list.front_ptr()->next_ != nullptr);
  BOOST_TEST(list.front_ptr()->next_->value_ == 4);
  BOOST_TEST(list.front_ptr()->next_->next_ != nullptr);
  BOOST_TEST(list.front_ptr()->next_->next_->value_ == 8);
  BOOST_TEST(list.front_ptr()->next_->next_->next_ == nullptr);

  list.add_at_head(5);

  BOOST_TEST(list.length() == 4);

  list.add_at_index(4, 3);

  BOOST_TEST(list.length() == 5);

  list.add_at_tail(0);

  BOOST_TEST(list.length() == 6);

  list.add_at_tail(5);

  BOOST_TEST(list.length() == 7);

  list.add_at_index(6, 3);

  BOOST_TEST(list.length() == 8);

  list.add_at_head(5);

  BOOST_TEST(list.length() == 9);

  list.delete_at_index(7);

  list.delete_at_index(5);

  list.add_at_tail(4);
}

/*
Runtime Error Message: Line 177: Char 50: runtime error: member access within null pointer of type 'MyLinkedList::SinglyListNode' (solution.cpp)
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior prog_joined.cpp:182:50
Last executed input: ["MyLinkedList","addAtHead","addAtTail","addAtTail","get","get","addAtTail","addAtIndex","addAtHead","addAtHead","addAtTail","addAtTail","addAtTail","addAtTail","get","addAtHead","addAtHead","addAtIndex","addAtIndex","addAtHead","addAtTail","deleteAtIndex","addAtHead","addAtHead","addAtIndex","addAtTail","get","addAtIndex","addAtTail","addAtHead","addAtHead","addAtIndex","addAtTail","addAtHead","addAtHead","get","deleteAtIndex","addAtTail","addAtTail","addAtHead","addAtTail","get","deleteAtIndex","addAtTail","addAtHead","addAtTail","deleteAtIndex","addAtTail","deleteAtIndex","addAtIndex","deleteAtIndex","addAtTail","addAtHead","addAtIndex","addAtHead","addAtHead","get","addAtHead","get","addAtHead","deleteAtIndex","get","addAtHead","addAtTail","get","addAtHead","get","addAtTail","get","addAtTail","addAtHead","addAtIndex","addAtIndex","addAtHead","addAtHead","deleteAtIndex","get","addAtHead","addAtIndex","addAtTail","get","addAtIndex","get","addAtIndex","get","addAtIndex","addAtIndex","addAtHead","addAtHead","addAtTail","addAtIndex","get","addAtHead","addAtTail","addAtTail","addAtHead","get","addAtTail","addAtHead","addAtTail","get","addAtIndex"]
[[],[84],[2],[39],[3],[1],[42],[1,80],[14],[1],[53],[98],[19],[12],[2],[16],[33],[4,17],[6,8],[37],[43],[11],[80],[31],[13,23],[17],[4],[10,0],[21],[73],[22],[24,37],[14],[97],[8],[6],[17],[50],[28],[76],[79],[18],[30],[5],[9],[83],[3],[40],[26],[20,90],[30],[40],[56],[15,23],[51],[21],[26],[83],[30],[12],[8],[4],[20],[45],[10],[56],[18],[33],[2],[70],[57],[31,24],[16,92],[40],[23],[26],[1],[92],[3,78],[42],[18],[39,9],[13],[33,17],[51],[18,95],[18,33],[80],[21],[7],[17,46],[33],[60],[26],[4],[9],[45],[38],[95],[78],[54],[42,86]]
*/

BOOST_AUTO_TEST_SUITE_END() // LinkedLists_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures