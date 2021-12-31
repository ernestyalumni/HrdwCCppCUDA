#include "DataStructures/LinkedLists/LinkedList.h"

#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <utility>

using DataStructures::LinkedLists::DWHarder::LinkedList;
using Tools::CaptureCoutFixture;
using std::cout;

template <typename T>
using Node = LinkedList<T>::Node;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(LinkedList_tests)

//------------------------------------------------------------------------------
/// \ref 3.05.Linked_lists.ppptx, U. Waterloo, D.W. Harder. Slide 24.
//------------------------------------------------------------------------------
int f()
{
  // ls is declared as a local variable on the stack
  LinkedList<int> ls;

  ls.push_front(3);

  cout << ls.front();

  // The return value is evaluated.
  // The compiler then calls the destructor for local variables.
  // The memory allocated for 'ls' is deallocated.
  return 0;
}

LinkedList<int>* dynamic_f(const int n)
{
  // pls is allocated memory by the OS.
  LinkedList<int>* pls {new LinkedList<int>{}};

  pls->push_front(n);
  cout << pls->front();

  // The address of the linked list is the return value.
  // After this, the 4 bytes for the pointer 'pls' is deallocated.
  // The memory allocated for the linked list is still there.

  return pls;
}

//------------------------------------------------------------------------------
/// \details Compiler will warn against this function, return address of local
/// variable.
//------------------------------------------------------------------------------
/*
LinkedList<int>* broken_f()
{
  // ls is declared as a local variable on the stack.
  LinkedList<int> ls;

  ls.push_front(3);
  cout << ls.front();

  // The return value is evaluated.
  // The compiler then calls the destructor for local variables.
  // The memory allocated for 'ls' is deallocated.

  return &ls;
}
*/

class DWHarderLinkedListFixture
{
  public:

    DWHarderLinkedListFixture():
      ls_{}
    {
      ls_.push_front(1);
      ls_.push_front(4);
      ls_.push_front(9);
      ls_.push_front(16);
    }

    virtual ~DWHarderLinkedListFixture() = default;

    LinkedList<int> ls_;
};

//------------------------------------------------------------------------------
/// \ref 3.05.Linked_lists.ppptx, U. Waterloo, D.W. Harder.
/// \details Allocation: constructor is called whenever an object is created,
/// either Statically or Dynamically.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsOnStack)
{
  // Statically. This statement defines ls to be a linked list and compiler
  // deals with memory allocation.
  LinkedList<int> ls;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsDynamically)
{
  // Dynamically created object. This statement requests sufficient memory from
  // OS to store instance of the class.
  LinkedList<int>* ls_ptr {new LinkedList<int>{}};

  delete ls_ptr;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructedListReturnsTrueForEmpty)
{
  LinkedList<int> ls;

  BOOST_TEST(ls.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructedListReturnsNullptrForBegin)
{
  LinkedList<int> ls;

  BOOST_TEST(ls.begin() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(AllocateOnStackInFunction, CaptureCoutFixture)
{
  f();

  BOOST_TEST(local_oss_.str() == "3");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(AllocateOnFreeStoreInFunction, CaptureCoutFixture)
{
  LinkedList<int>* pls {dynamic_f(42)};

  BOOST_TEST(local_oss_.str() == "42");

  delete pls;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CopyConstructs, DWHarderLinkedListFixture)
{
  LinkedList<int> ls {ls_};

  BOOST_TEST(ls.pop_front() == 16);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 9);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 4);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 1);
  BOOST_TEST(ls.empty());

  BOOST_TEST(ls_.pop_front() == 16);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 9);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 4);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 1);
  BOOST_TEST(ls_.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushFrontAddsValueToLinkedList)
{
  LinkedList<int> ls;
  ls.push_front(35);
  ls.push_front(18);
  ls.push_front(94);
  ls.push_front(72);

  BOOST_TEST(ls.front() == 72);
  BOOST_TEST(ls.pop_front() == 72);
  BOOST_TEST(ls.front() == 94);
  BOOST_TEST(ls.pop_front() == 94);
  BOOST_TEST(ls.front() == 18);
  BOOST_TEST(ls.pop_front() == 18);
  BOOST_TEST(ls.front() == 35);
  BOOST_TEST(ls.pop_front() == 35);
  BOOST_TEST(ls.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(AssignmentWithLValue, DWHarderLinkedListFixture)
{
  LinkedList<int> ls {};

  ls = ls_;

  BOOST_TEST(ls.pop_front() == 16);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 9);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 4);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 1);
  BOOST_TEST(ls.empty());

  BOOST_TEST(ls_.pop_front() == 16);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 9);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 4);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 1);
  BOOST_TEST(ls_.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveAssigns, DWHarderLinkedListFixture)
{
  LinkedList<int> ls {};

  ls = std::move(ls_);

  BOOST_TEST(ls_.empty());

  BOOST_TEST(ls.pop_front() == 16);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 9);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 4);
  BOOST_TEST(!ls.empty());
  BOOST_TEST(ls.pop_front() == 1);
  BOOST_TEST(ls.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(Reverses, DWHarderLinkedListFixture)
{
  Node<int>* head_ptr {ls_.reverse_list()};

  BOOST_TEST(head_ptr->value_ == 1);
  BOOST_TEST(head_ptr->next_->value_ == 4);
  BOOST_TEST(head_ptr->next_->next_->value_ == 9);
  BOOST_TEST(head_ptr->next_->next_->next_->value_ == 16);

  BOOST_TEST(ls_.pop_front() == 1);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 4);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 9);
  BOOST_TEST(!ls_.empty());
  BOOST_TEST(ls_.pop_front() == 16);
  BOOST_TEST(ls_.empty());
}

BOOST_AUTO_TEST_SUITE_END() // LinkedList_tests
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures