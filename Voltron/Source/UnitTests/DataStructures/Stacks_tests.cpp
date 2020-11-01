//------------------------------------------------------------------------------
/// \file Stack_tests.cpp
/// \date 20201101 07:37
//------------------------------------------------------------------------------
#include "DataStructures/Stacks.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Stacks::StackWithVector;
using DataStructures::Stacks::Stack;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Stack_tests)


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StackWithVectorIsAStack)
{
  StackWithVector<int> stack;

  stack.push(1);
  stack.push(2);
  stack.push(3);

  for (int i {0}; i < 3; ++i)
  {
    if (!stack.is_empty())
    {
      BOOST_TEST(stack.top() == 3 - i);
    }

    BOOST_TEST(stack.pop());
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StackIsAStack)
{
  Stack<int> stack;

  stack.push(1);
  stack.push(2);
  stack.push(3);

  for (int i {0}; i < 3; ++i)
  {
    if (!stack.is_empty())
    {
      BOOST_TEST(stack.top() == 3 - i);
    }

    BOOST_TEST(stack.pop());
  }
}

// https://leetcode.com/explore/learn/card/queue-stack/230/usage-stack/1358/

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StackWorksWithFurtherUsageExamples)
{
  // 1. Initialize a stack.
  Stack<int> s;
  // 2. Push new element.
  s.push(5);
  s.push(13);
  s.push(8);
  s.push(6);
  // 3. Check if stack is empty.
  BOOST_TEST(!s.is_empty());

  // 4. Pop an element.
  BOOST_TEST(s.pop());

  // 5. Get the top element.
  BOOST_TEST(s.top() == 8);

  // 6. Get the size of the stack
  BOOST_TEST(s.size() == 3);
}

BOOST_AUTO_TEST_SUITE_END() // Stack_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures