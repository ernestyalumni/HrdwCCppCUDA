//------------------------------------------------------------------------------
/// \file Queue_tests.cpp
/// \date 20201101 04:56
//------------------------------------------------------------------------------
#include "DataStructures/Queues/Queues.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Queues::CircularQueue;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queue_tests)

// cf. https://leetcode.com/explore/learn/card/queue-stack/228/first-in-first-out-data-structure/1337/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CircularQueueIsAQueue)
{
  CircularQueue<int> queue {3};

  BOOST_TEST(queue.enqueue(1));
  BOOST_TEST(queue.enqueue(2));
  BOOST_TEST(queue.enqueue(3));
  BOOST_TEST(!queue.enqueue(4)); // Return false, queue is full.
  BOOST_TEST(queue.rear() == 3);
  BOOST_TEST(queue.is_full());
  BOOST_TEST(queue.dequeue()); // return true
  BOOST_TEST(queue.enqueue(4)); // return true
  BOOST_TEST(queue.rear() == 4);
}

BOOST_AUTO_TEST_SUITE_END() // Queue_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures