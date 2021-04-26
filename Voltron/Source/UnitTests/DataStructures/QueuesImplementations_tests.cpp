#include "DataStructures/QueuesImplementations.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Queues::CRTP::QueueAsArray;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queues)
BOOST_AUTO_TEST_SUITE(QueuesImplementations_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(QueueAsArrayDefaultConstructs)
{
  QueueAsArray<int> queue;
  BOOST_TEST(queue.size() == 0);
  BOOST_TEST(queue.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(QueueAsArrayEnqueues)
{
  QueueAsArray<int> queue;

  for (int i {0}; i < 3; ++i)
  {
    queue.enqueue(42 + i);
    BOOST_TEST(queue.size() == (i + 1));
    BOOST_TEST(!queue.is_empty()); 
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(QueueAsArrayDequeues)
{
  QueueAsArray<int> queue;

  for (int i {0}; i < 3; ++i)
  {
    queue.enqueue(42 + i);
  }

  for (int i {0}; i < 3; ++i)
  {
    BOOST_TEST(!queue.is_empty());
    const int result {queue.dequeue()};
    BOOST_TEST(result == (42 + i));
    BOOST_TEST(queue.size() == (2 - i));
  }
}

BOOST_AUTO_TEST_SUITE_END() // QueuesImplementations_tests
BOOST_AUTO_TEST_SUITE_END() // Queue
BOOST_AUTO_TEST_SUITE_END() // DataStructures