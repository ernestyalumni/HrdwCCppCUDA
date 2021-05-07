#include "DataStructures/Queues/QueuesWithArrays.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queues)
BOOST_AUTO_TEST_SUITE(QueuesWithArrays_tests)

BOOST_AUTO_TEST_SUITE(QueuesWithHeadTailFixedSizeArrayOnStack_tests)

template <typename T, std::size_t N>
using Queue =
  DataStructures::Queues::CRTP::QueueWithHeadTailFixedSizeArrayOnStack<T, N>;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  Queue<int, 8> queue;

  BOOST_TEST(queue.head() == 0);
  BOOST_TEST(queue.tail() == 0);
  BOOST_TEST(queue.is_empty());
  BOOST_TEST(queue.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(QueueEnqueues)
{
  Queue<int, 12> queue;

  queue.enqueue(1);

  BOOST_TEST(queue.head() == 0);
  BOOST_TEST(queue.tail() == 1);
  BOOST_TEST(!queue.is_empty());
  BOOST_TEST(queue.size() == 1);

  queue.enqueue(2);

  BOOST_TEST(queue.head() == 0);
  BOOST_TEST(queue.tail() == 2);
  BOOST_TEST(!queue.is_empty());
  BOOST_TEST(queue.size() == 2);

  queue.enqueue(3);

  BOOST_TEST(queue.head() == 0);
  BOOST_TEST(queue.tail() == 3);
  BOOST_TEST(!queue.is_empty());
  BOOST_TEST(queue.size() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(QueueDequeues)
{
  Queue<int, 12> queue;

  for (int i {0}; i < 6; ++i)
  {
    queue.enqueue(i + 1);

    BOOST_TEST(queue.head() == 0);
    BOOST_TEST(queue.tail() == i + 1);
    BOOST_TEST(!queue.is_empty());
    BOOST_TEST(queue.size() == i+1);
  }

  queue.enqueue(15);
  queue.enqueue(6);
  queue.enqueue(9);
  queue.enqueue(8);
  queue.enqueue(4);

  BOOST_TEST(queue.head() == 0);
  BOOST_TEST(queue.tail() == 11);

  for (int i {0}; i < 6; ++i)
  {
    const int item {queue.dequeue()};
    BOOST_TEST(item == i + 1);
    BOOST_TEST(queue.head() == i + 1);
  }

  BOOST_TEST(queue.tail() == 11);
  BOOST_TEST(queue.head() == 6);
  BOOST_TEST(queue.size() == 5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(EnqueueWrapsAround)
{
  Queue<int, 12> queue;

  for (int i {0}; i < 6; ++i)
  {
    queue.enqueue(i + 1);

    BOOST_TEST(queue.head() == 0);
    BOOST_TEST(queue.tail() == i + 1);
    BOOST_TEST(!queue.is_empty());
    BOOST_TEST(queue.size() == i+1);
  }

  queue.enqueue(15);
  queue.enqueue(6);
  queue.enqueue(9);
  queue.enqueue(8);
  queue.enqueue(4);

  for (int i {0}; i < 6; ++i)
  {
    const int item {queue.dequeue()};
    BOOST_TEST(item == i + 1);
    BOOST_TEST(queue.head() == i + 1);
  }

  queue.enqueue(17);

  BOOST_TEST(queue.head() == 6);
  BOOST_TEST(queue.tail() == 0);
  BOOST_TEST(!queue.is_empty());
  BOOST_TEST(queue.size() == 6);

  queue.enqueue(3);

  BOOST_TEST(queue.head() == 6);
  BOOST_TEST(queue.tail() == 1);
  BOOST_TEST(!queue.is_empty());
  BOOST_TEST(queue.size() == 7);

  queue.enqueue(5);

  BOOST_TEST(queue.head() == 6);
  BOOST_TEST(queue.tail() == 2);
  BOOST_TEST(!queue.is_empty());
  BOOST_TEST(queue.size() == 8);
}

BOOST_AUTO_TEST_SUITE_END() // QueuesWithHeadTailFixedSizeArrayOnStack_tests

BOOST_AUTO_TEST_SUITE_END() // QueuesWithArrays_tests
BOOST_AUTO_TEST_SUITE_END() // Queue
BOOST_AUTO_TEST_SUITE_END() // DataStructures