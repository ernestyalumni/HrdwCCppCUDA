#include "DataStructures/Stacks/StacksImplementations.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t

template <typename T>
using StackAsResizeableArray =
  DataStructures::Stacks::CRTP::StackAsResizeableArray<T>;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Stacks)
BOOST_AUTO_TEST_SUITE(StacksImplementations_tests)

BOOST_AUTO_TEST_SUITE(StackAsResizeableArray_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  StackAsResizeableArray<int> stack;

  BOOST_TEST(stack.is_empty());
  BOOST_TEST(stack.size() == 0);
}

//------------------------------------------------------------------------------
/// \ref Cormen, Leiserson, Rivest, and Stein (2009), pp. 235. Exercises 10.1-1
/// Use Fig. 10.1 as a model, illustrate each operation.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushesAndPops)
{
  StackAsResizeableArray<int> stack;

  // Initially an empty stack.
  BOOST_TEST_REQUIRE(stack.is_empty());
  BOOST_TEST_REQUIRE(stack.size() == 0);

  stack.push(4);
  BOOST_TEST(!stack.is_empty());
  BOOST_TEST(stack.size() == 1);

  stack.push(1);
  BOOST_TEST(!stack.is_empty());
  BOOST_TEST(stack.size() == 2);

  stack.push(3);
  BOOST_TEST(!stack.is_empty());
  BOOST_TEST(stack.size() == 3);

  BOOST_TEST(stack.pop() == 3);
  BOOST_TEST(!stack.is_empty());
  BOOST_TEST(stack.size() == 2);

  stack.push(8);
  BOOST_TEST(!stack.is_empty());
  BOOST_TEST(stack.size() == 3);

  BOOST_TEST(stack.pop() == 8);
  BOOST_TEST(!stack.is_empty());
  BOOST_TEST(stack.size() == 2);
}

BOOST_AUTO_TEST_SUITE_END() // StackAsArray_tests

BOOST_AUTO_TEST_SUITE_END() // StacksImplementations_tests
BOOST_AUTO_TEST_SUITE_END() // Stacks
BOOST_AUTO_TEST_SUITE_END() // DataStructures