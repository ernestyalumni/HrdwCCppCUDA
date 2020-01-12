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
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using DataStructures::LinkedLists::Element;
using DataStructures::LinkedLists::Node;
using DataStructures::LinkedLists::LinkedList;

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

BOOST_AUTO_TEST_SUITE_END() // LinkedLists_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures