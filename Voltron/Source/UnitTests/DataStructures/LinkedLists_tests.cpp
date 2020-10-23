//------------------------------------------------------------------------------
// \file LinkedLists_tests.cpp
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
using DataStructures::LinkedLists::ListNode;
using DataStructures::LinkedLists::Node;
using DataStructures::LinkedLists::NodeAsUniquePtr;
using DataStructures::LinkedLists::merge_two_sorted_lists_by_splice;
using DataStructures::LinkedLists::merge_two_sorted_lists_iterative;
using DataStructures::LinkedLists::merge_two_sorted_lists_simple;
using DataStructures::LinkedLists::splice_nodes;
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

  {

  }

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

BOOST_AUTO_TEST_SUITE_END() // LinkedLists_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures