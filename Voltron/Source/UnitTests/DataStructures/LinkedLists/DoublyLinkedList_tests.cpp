#include "DataStructures/LinkedLists/DoublyLinkedList.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <utility>

template <typename T>
using Node = DataStructures::LinkedLists::DoublyLinkedList<T>::Node;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(DoublyLinkedList_tests)

class DoublyLinkedListFixture
{
  public:

    DoublyLinkedListFixture():
      ls_{}
    {
      ls_.push_front(1);
      ls_.push_front(4);
      ls_.push_front(16);
      ls_.push_front(9);
      ls_.push_front(25);
    }

    virtual ~DoublyLinkedListFixture() = default;

    DoublyLinkedList<int> ls_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  DoublyLinkedList<int> ls {};

  BOOST_TEST(ls.is_empty());
  BOOST_TEST(ls.head() == nullptr);
  BOOST_TEST(ls.tail() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushFrontInserts)
{
  DoublyLinkedList<int> ls {};
  ls.push_front(1);
  BOOST_TEST(ls.head()->retrieve() == 1);
  BOOST_TEST(ls.tail()->retrieve() == 1);
  ls.push_front(4);
  BOOST_TEST(ls.head()->retrieve() == 4);
  BOOST_TEST(ls.tail()->retrieve() == 1);
  ls.push_front(16);
  BOOST_TEST(ls.head()->retrieve() == 16);
  BOOST_TEST(ls.tail()->retrieve() == 1);
  ls.push_front(9);
  BOOST_TEST(ls.head()->retrieve() == 9);
  BOOST_TEST(ls.tail()->retrieve() == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(HeadGetsHead, DoublyLinkedListFixture)
{
  BOOST_TEST(ls_.head()->retrieve() == 25);
  BOOST_TEST(ls_.head()->previous() == nullptr);
  BOOST_TEST(ls_.head()->next()->retrieve() == 9);
  BOOST_TEST(ls_.head()->next()->next()->retrieve() == 16);
  BOOST_TEST(ls_.head()->next()->next()->next()->retrieve() == 4);
  BOOST_TEST(ls_.head()->next()->next()->next()->next()->retrieve() == 1);
  BOOST_TEST(ls_.head()->next()->next()->next()->next()->next() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(TailGetsTail, DoublyLinkedListFixture)
{
  BOOST_TEST(ls_.tail()->retrieve() == 1);
  BOOST_TEST(ls_.tail()->next() == nullptr);
  BOOST_TEST(ls_.tail()->previous()->retrieve() == 4);
  BOOST_TEST(ls_.tail()->previous()->previous()->retrieve() == 16);
  BOOST_TEST(ls_.tail()->previous()->previous()->previous()->retrieve() == 9);
  BOOST_TEST(
    ls_.tail()->previous()->previous()->previous()->previous()->retrieve() ==
      25);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SearchFindsValues, DoublyLinkedListFixture)
{
  Node<int>* x {ls_.search(1)};
  BOOST_TEST(x->retrieve() == 1);
  BOOST_TEST(x->previous()->retrieve() == 4);
  BOOST_TEST(x->next() == nullptr);

  x = ls_.search(4);
  BOOST_TEST(x->retrieve() == 4);
  BOOST_TEST(x->previous()->retrieve() == 16);
  BOOST_TEST(x->next()->retrieve() == 1);

  x = ls_.search(16);
  BOOST_TEST(x->retrieve() == 16);
  BOOST_TEST(x->previous()->retrieve() == 9);
  BOOST_TEST(x->next()->retrieve() == 4);

  x = ls_.search(9);
  BOOST_TEST(x->retrieve() == 9);
  BOOST_TEST(x->previous()->retrieve() == 25);
  BOOST_TEST(x->next()->retrieve() == 16);

  x = ls_.search(25);
  BOOST_TEST(x->retrieve() == 25);
  BOOST_TEST(x->previous() == nullptr);
  BOOST_TEST(x->next()->retrieve() == 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SearchReturnsNullPtrForMissingValue,
  DoublyLinkedListFixture)
{
  Node<int>* x {ls_.search(2)};
  BOOST_TEST(x == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(PopFrontDeletes, DoublyLinkedListFixture)
{
  BOOST_TEST(ls_.pop_front() == 25);
  BOOST_TEST(ls_.head()->previous() == nullptr);
  BOOST_TEST(ls_.tail()->next() == nullptr);
  BOOST_TEST(ls_.head()->retrieve() == 9);
  BOOST_TEST(ls_.tail()->retrieve() == 1);
  BOOST_TEST(!ls_.is_empty());

  BOOST_TEST(ls_.pop_front() == 9);
  BOOST_TEST(ls_.head()->previous() == nullptr);
  BOOST_TEST(ls_.tail()->next() == nullptr);
  BOOST_TEST(ls_.head()->retrieve() == 16);
  BOOST_TEST(ls_.tail()->retrieve() == 1);
  BOOST_TEST(!ls_.is_empty());

  BOOST_TEST(ls_.pop_front() == 16);
  BOOST_TEST(ls_.head()->previous() == nullptr);
  BOOST_TEST(ls_.tail()->next() == nullptr);

  BOOST_TEST(ls_.pop_front() == 4);
  BOOST_TEST(ls_.head()->retrieve() == 1);
  BOOST_TEST(ls_.head()->previous() == nullptr);
  BOOST_TEST(ls_.head()->next() == nullptr);
  BOOST_TEST(ls_.tail()->retrieve() == 1);
  BOOST_TEST(ls_.tail()->previous() == nullptr);
  BOOST_TEST(ls_.tail()->next() == nullptr);
  BOOST_TEST(ls_.head() == ls_.tail());
  BOOST_TEST(!ls_.is_empty());

  BOOST_TEST(ls_.pop_front() == 1);
  BOOST_TEST(ls_.head() == nullptr);
  BOOST_TEST(ls_.tail() == nullptr);
  BOOST_TEST(ls_.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ListDeleteRemovesElementFromTheMiddle,
  DoublyLinkedListFixture)
{
  ls_.list_delete(16);

  Node<int>* x {ls_.search(4)};
  BOOST_TEST(x->retrieve() == 4);
  BOOST_TEST(x->previous()->retrieve() == 9);
  BOOST_TEST(x->next()->retrieve() == 1);

  x = ls_.search(9);
  BOOST_TEST(x->retrieve() == 9);
  BOOST_TEST(x->previous()->retrieve() == 25);
  BOOST_TEST(x->next()->retrieve() == 4);

  x = ls_.search(16);
  BOOST_TEST(x == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ListDeleteRemovesElementFromFront,
  DoublyLinkedListFixture)
{
  ls_.list_delete(25);

  Node<int>* x {ls_.search(25)};
  BOOST_TEST(x == nullptr);

  x = ls_.search(9);
  BOOST_TEST(x->retrieve() == 9);
  BOOST_TEST(x->previous() == nullptr);
  BOOST_TEST(x->next()->retrieve() == 16);
}

BOOST_AUTO_TEST_SUITE_END() // DoublyLinkedList_tests
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures