#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "DataStructures/Stacks/StacksWithArrays.h"
#include "UnitTests/DataStructures/LinkedLists/DoublyLinkedListTestValues.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t

using std::size_t;

template <typename T, size_t N>
using TwoStacksOneArray = DataStructures::Stacks::TwoStacksOneArray<T, N>;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Stacks)
BOOST_AUTO_TEST_SUITE(StacksWithArrays_tests)

BOOST_AUTO_TEST_SUITE(StackWithDynamicArray_tests)

template <typename T>
using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<T>;

template <typename T>
using StackWithDynamicArray = DataStructures::Stacks::StackWithDynamicArray<T>;

using DoublyLinkedListTestValues =
  UnitTests::DataStructures::LinkedLists::DoublyLinkedListTestValues;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  StackWithDynamicArray<DoublyLinkedList<int>> stack {};

  BOOST_TEST(stack.size() == 0);
  BOOST_TEST(stack.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WorksWithDoublyLinkedLists)
{
  DoublyLinkedListTestValues tv {};

  StackWithDynamicArray<DoublyLinkedList<int>> stack {};
  stack.push(tv.large_operand_1_);

  BOOST_TEST(stack.size() == 1);
  BOOST_TEST(!stack.is_empty());

  auto& top_reference = stack.top();
  BOOST_TEST(top_reference.size() == 8);
  BOOST_TEST(top_reference.head()->retrieve() == 1);
  BOOST_TEST(top_reference.head()->next()->retrieve() == 2);
  BOOST_TEST(top_reference.head()->next()->next()->retrieve() == 3);
  BOOST_TEST(top_reference.head()->next()->next()->next()->retrieve() == 4);

  stack.push(tv.large_operand_2_);
  BOOST_TEST(stack.size() == 2);

  stack.push(tv.large_operand_3_);
  BOOST_TEST(stack.size() == 3);

  top_reference = stack.top();
  BOOST_TEST(top_reference.size() == 9);
  BOOST_TEST(top_reference.head()->retrieve() == 0);
  BOOST_TEST(top_reference.head()->next()->retrieve() == 1);
  BOOST_TEST(top_reference.head()->next()->next()->retrieve() == 2);
  BOOST_TEST(top_reference.head()->next()->next()->next()->retrieve() == 3);

  stack.push(tv.large_operand_4_);
  BOOST_TEST(stack.size() == 4);

  top_reference = stack.top();
  BOOST_TEST(top_reference.head()->retrieve() == 9);
  BOOST_TEST(top_reference.head()->next()->retrieve() == 8);
  BOOST_TEST(top_reference.head()->next()->next()->retrieve() == 7);
  BOOST_TEST(top_reference.head()->next()->next()->next()->retrieve() == 6);

  stack.pop();
  BOOST_TEST(stack.size() == 3);
  BOOST_TEST(!stack.is_empty());

  auto& popped_reference = stack.top();

  BOOST_TEST(popped_reference.head()->retrieve() == 0);
  BOOST_TEST(popped_reference.head()->next()->retrieve() == 1);
  BOOST_TEST(popped_reference.tail()->retrieve() == 8);
  stack.pop();
  BOOST_TEST(stack.size() == 2);
  BOOST_TEST(!stack.is_empty());

  stack.pop();
  BOOST_TEST(stack.size() == 1);
  BOOST_TEST(!stack.is_empty());

  //stack.pop();
  //BOOST_TEST(stack.size() == 0);
  //BOOST_TEST(stack.is_empty());


  {
    auto& popped_reference = stack.top();
    // BOOST_TEST(popped_reference.head()->retrieve() == 9);
    //BOOST_TEST(popped_reference.head()->next()->retrieve() == 8);
    //BOOST_TEST(popped_reference.tail()->retrieve() == 6);

  }

  // BOOST_TEST(popped_reference.head()->retrieve() == 9);
  //BOOST_TEST(popped_reference.head()->next()->retrieve() == 8);
  // BOOST_TEST(popped_reference.tail()->retrieve() == 6);
  // stack.pop();
  //BOOST_TEST(stack.size() == 2);
  //BOOST_TEST(!stack.is_empty());

  //popped_reference = stack.top();

  //BOOST_TEST(popped_reference.head()->retrieve() == 9);
  //BOOST_TEST(popped_reference.head()->next()->retrieve() == 9);
  //BOOST_TEST(popped_reference.tail()->retrieve() == 1);
}

BOOST_AUTO_TEST_SUITE_END() // StackWithDynamicArray_tests

BOOST_AUTO_TEST_SUITE(TwoStacksOneArray_tests)

template <typename T, size_t N>
class TestTwoStacksOneArray : public TwoStacksOneArray<T, N>
{
  public:

    using TwoStacksOneArray<T, N>::TwoStacksOneArray;
    using TwoStacksOneArray<T, N>::top_index_1;
    using TwoStacksOneArray<T, N>::top_index_2;    
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  TwoStacksOneArray<int, 12> a;

  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.is_empty_1());
  BOOST_TEST(a.is_empty_2());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Stack1NoLongerEmptyAfterFirstPush)
{
  TestTwoStacksOneArray<int, 12> a;

  BOOST_TEST_REQUIRE(a.is_empty_1());

  a.push_1(4);
  BOOST_TEST(!a.is_empty_1());
  BOOST_TEST(*a.top_index_1() == 0);
}

//------------------------------------------------------------------------------
/// \ref Cormen, Leiserson, Rivest, and Stein (2009), pp. 235. Exercises 10.1-2
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Queue1PushesAndPops)
{
  TwoStacksOneArray<int, 12> a;

  a.push_1(4);
  BOOST_TEST(a.top_1() == 4);
  a.push_1(1);
  BOOST_TEST(a.top_1() == 1);
  a.push_1(3);
  BOOST_TEST(a.top_1() == 3);
  BOOST_TEST(a.pop_1() == 3);
  BOOST_TEST(a.top_1() == 1);
  a.push_1(8);
  BOOST_TEST(a.top_1() == 8);
  BOOST_TEST(a.pop_1() == 8);
  BOOST_TEST(a.top_1() == 1);

  BOOST_TEST(a.pop_1() == 1);
  BOOST_TEST(a.top_1() == 4);

  BOOST_TEST(a.pop_1() == 4);
  BOOST_TEST(a.is_empty_1());
  BOOST_TEST(a.is_empty_2());
  BOOST_TEST(a.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Queue2PushesAndPops)
{
  TwoStacksOneArray<int, 12> a;

  a.push_2(4);
  BOOST_TEST(a.top_2() == 4);
  a.push_2(1);
  BOOST_TEST(a.top_2() == 1);
  a.push_2(3);
  BOOST_TEST(a.top_2() == 3);
  BOOST_TEST(a.pop_2() == 3);
  BOOST_TEST(a.top_2() == 1);
  a.push_2(8);
  BOOST_TEST(a.top_2() == 8);
  BOOST_TEST(a.pop_2() == 8);
  BOOST_TEST(a.top_2() == 1);

  BOOST_TEST(a.pop_2() == 1);
  BOOST_TEST(a.top_2() == 4);

  BOOST_TEST(a.pop_2() == 4);
  BOOST_TEST(a.is_empty_1());
  BOOST_TEST(a.is_empty_2());
  BOOST_TEST(a.size() == 0);
}

BOOST_AUTO_TEST_SUITE_END() // TwoStacksOneArray_tests

BOOST_AUTO_TEST_SUITE_END() // StacksWithArrays_tests
BOOST_AUTO_TEST_SUITE_END() // Stacks
BOOST_AUTO_TEST_SUITE_END() // DataStructures