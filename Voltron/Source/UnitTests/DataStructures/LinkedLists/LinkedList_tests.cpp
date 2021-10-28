#include "DataStructures/LinkedLists/LinkedList.h"

#include <boost/test/unit_test.hpp>

using DataStructures::LinkedLists::DWHarder::LinkedList;

template <typename T>
using Node = LinkedList<T>::Node;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(LinkedList_tests)
BOOST_AUTO_TEST_SUITE(Node_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsOnStack)
{
  Node<int> x;

  BOOST_TEST(x.get_value() == 0);
  BOOST_TEST(x.next_ptr() == nullptr);
}

BOOST_AUTO_TEST_SUITE_END() // Node_tests

//------------------------------------------------------------------------------
/// \ref 3.05.Linked_lists.ppptx, U. Waterloo, D.W. Harder, 

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsOnStack)
{
  LinkedList<int> ls;

  BOOST_TEST(x.get_value() == 0);
  BOOST_TEST(x.next_ptr() == nullptr);
}

BOOST_AUTO_TEST_SUITE_END() // LinkedList_tests
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures