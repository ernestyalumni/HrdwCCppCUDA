#include "DataStructures/LinkedLists/SentinelDoublyLinkedList.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <utility>

using DataStructures::LinkedLists::CLRS::SentinelDoublyLinkedList;

template <typename T>
using Node = SentinelDoublyLinkedList<T>::Node;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(SentinelDoublyLinkedList_tests)
BOOST_AUTO_TEST_SUITE(CLRSSentinelDoublyLinkedList_tests)

class CLRSSentinelDoublyLinkedListFixture
{
  public:

    CLRSSentinelDoublyLinkedListFixture():
      ls_{}
    {
      ls_.push_front(1);
      ls_.push_front(4);
      ls_.push_front(16);
      ls_.push_front(9);
      ls_.push_front(25);
    }

    virtual ~CLRSSentinelDoublyLinkedListFixture() = default;

    SentinelDoublyLinkedList<int> ls_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  SentinelDoublyLinkedList<int> ls {};

  BOOST_TEST(ls.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushFrontInserts)
{
  SentinelDoublyLinkedList<int> ls {};
  ls.push_front(1);
  BOOST_TEST(ls.head()->value_ == 1);
  BOOST_TEST(ls.tail()->value_ == 1);
  ls.push_front(4);
  BOOST_TEST(ls.head()->value_ == 4);
  BOOST_TEST(ls.tail()->value_ == 1);
  ls.push_front(16);
  BOOST_TEST(ls.head()->value_ == 16);
  BOOST_TEST(ls.tail()->value_ == 1);
  ls.push_front(9);
  BOOST_TEST(ls.head()->value_ == 9);
  BOOST_TEST(ls.tail()->value_ == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CanBeDynamicallyAllocated)
{
  SentinelDoublyLinkedList<int>* ls_ptr {new SentinelDoublyLinkedList<int>{}};

  delete ls_ptr;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SearchFindsValues, CLRSSentinelDoublyLinkedListFixture)
{
  Node<int>* x {ls_.search(1)};
  BOOST_TEST(x->value_ == 1);
  BOOST_TEST(x->previous_->value_ == 4);
  BOOST_TEST(ls_.is_sentinel(x->next_));

  x = ls_.search(4);
  BOOST_TEST(x->value_ == 4);
  BOOST_TEST(x->previous_->value_ == 16);
  BOOST_TEST(x->next_->value_ == 1);

  x = ls_.search(16);
  BOOST_TEST(x->value_ == 16);
  BOOST_TEST(x->previous_->value_ == 9);
  BOOST_TEST(x->next_->value_ == 4);

  x = ls_.search(9);
  BOOST_TEST(x->value_ == 9);
  BOOST_TEST(x->previous_->value_ == 25);
  BOOST_TEST(x->next_->value_ == 16);

  x = ls_.search(25);
  BOOST_TEST(x->value_ == 25);
  BOOST_TEST(ls_.is_sentinel(x->previous_));
  BOOST_TEST(x->next_->value_ == 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SearchReturnsSentinelForMissingValue,
  CLRSSentinelDoublyLinkedListFixture)
{
  Node<int>* x {ls_.search(2)};
  BOOST_TEST(ls_.is_sentinel(x));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ListDeleteRemovesElementFromTheMiddle,
  CLRSSentinelDoublyLinkedListFixture)
{
  ls_.list_delete(16);

  Node<int>* x {ls_.search(4)};
  BOOST_TEST(x->value_ == 4);
  BOOST_TEST(x->previous_->value_ == 9);
  BOOST_TEST(x->next_->value_ == 1);

  x = ls_.search(9);
  BOOST_TEST(x->value_ == 9);
  BOOST_TEST(x->previous_->value_ == 25);
  BOOST_TEST(x->next_->value_ == 4);

  x = ls_.search(16);
  BOOST_TEST(ls_.is_sentinel(x));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ListDeleteRemovesElementFromFront,
  CLRSSentinelDoublyLinkedListFixture)
{
  ls_.list_delete(25);

  Node<int>* x {ls_.search(25)};
  BOOST_TEST(ls_.is_sentinel(x));

  x = ls_.search(9);
  BOOST_TEST(x->value_ == 9);
  BOOST_TEST(ls_.is_sentinel(x->previous_));
  BOOST_TEST(x->next_->value_ == 16);
}

BOOST_AUTO_TEST_SUITE_END() // CLRSSentinelDoublyLinkedList_tests
BOOST_AUTO_TEST_SUITE_END() // SentinelDoublyLinkedList_tests
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures