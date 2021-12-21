//------------------------------------------------------------------------------
/// \file DynamicQueue_tests.cpp
/// \date 20211101 23:08
//------------------------------------------------------------------------------
#include "DataStructures/Queues/DynamicQueue.h"

#include <boost/test/unit_test.hpp>

using namespace DataStructures::Queues;

template <typename T>
class TestDynamicQueueAsHierarchy : public AsHierarchy::DynamicQueue<T>
{
  public:

    // Derived class needs a constructor.
    using AsHierarchy::DynamicQueue<T>::DynamicQueue;

    using AsHierarchy::DynamicQueue<T>::is_full;
    using AsHierarchy::DynamicQueue<T>::back;
    using AsHierarchy::DynamicQueue<T>::get_front_index;
    using AsHierarchy::DynamicQueue<T>::get_back_index;
    using AsHierarchy::DynamicQueue<T>::get_size;
    using AsHierarchy::DynamicQueue<T>::get_array_capacity;
};

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queues)
BOOST_AUTO_TEST_SUITE(DynamicQueue_tests)
BOOST_AUTO_TEST_SUITE(AsHierarchy_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  AsHierarchy::DynamicQueue<int> q;

  BOOST_TEST(q.is_empty());
  BOOST_TEST(q.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  AsHierarchy::DynamicQueue<int> q {32};

  BOOST_TEST(q.is_empty());
  BOOST_TEST(q.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Pushes)
{
  AsHierarchy::DynamicQueue<int> q {32};

  BOOST_TEST(q.is_empty());
  BOOST_TEST(q.size() == 0);

  q.push(3);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 1);
  BOOST_TEST(q.front() == 3);

  q.push(5);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 2);
  BOOST_TEST(q.front() == 3);

  q.push(2);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 3);
  BOOST_TEST(q.front() == 3);

  q.push(15);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 4);
  BOOST_TEST(q.front() == 3);

  q.push(42);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(q.front() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Pops)
{
  AsHierarchy::DynamicQueue<int> q {32};

  q.push(3);
  q.push(5);
  q.push(2);
  q.push(15);
  q.push(42);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(q.front() == 3);

  BOOST_TEST(q.pop() == 3);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 4);
  BOOST_TEST(q.front() == 5);

  BOOST_TEST(q.pop() == 5);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 3);
  BOOST_TEST(q.front() == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushesAndPops)
{
  AsHierarchy::DynamicQueue<int> q {32};

  q.push(3);
  q.push(5);
  q.push(2);
  q.push(15);
  q.push(42);
  q.pop();
  q.pop();
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 3);
  BOOST_TEST(q.front() == 2);

  q.push(14);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 4);
  BOOST_TEST(q.front() == 2);

  q.push(7);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(q.front() == 2);

  BOOST_TEST(q.pop() == 2);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 4);
  BOOST_TEST(q.front() == 15);

  q.push(9);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(q.front() == 15);

  BOOST_TEST(q.pop() == 15);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 4);
  BOOST_TEST(q.front() == 42);

  BOOST_TEST(q.pop() == 42);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 3);
  BOOST_TEST(q.front() == 14);

  q.push(51);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 4);
  BOOST_TEST(q.front() == 14);

  BOOST_TEST(q.pop() == 14);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 3);
  BOOST_TEST(q.front() == 7);

  BOOST_TEST(q.pop() == 7);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(q.size() == 2);
  BOOST_TEST(q.front() == 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FillsToCapacityAndBack)
{
  AsHierarchy::DynamicQueue<int> q {16};

  for (int i {0}; i < 16; ++i)
  {
    q.push(i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == i + 1);
  }

  for (int i {0}; i < 5; ++i)
  {
    BOOST_TEST(q.pop() == i);
    BOOST_TEST(q.size() == 15 - i);
    BOOST_TEST(q.front() == i + 1);    
  }

  for (int i {0}; i < 5; ++i)
  {
    q.push(i + 16);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == 12 + i);
    BOOST_TEST(q.front() == 5);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FillsToCapacityAndDoubles)
{
  AsHierarchy::DynamicQueue<int> q {16};

  for (int i {0}; i < 16; ++i)
  {
    q.push(i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == i + 1);
    BOOST_TEST(q.front() == 0);
  }

  for (int i {0}; i < 16; ++i)
  {
    q.push(i + 16);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == i + 17);
    BOOST_TEST(q.front() == 0);
  }

  for (int i {0}; i < 31; ++i)
  {
    BOOST_TEST(q.pop() == i);
    BOOST_TEST(q.size() == 31 - i);
    BOOST_TEST(q.front() == i + 1);
  }

  BOOST_TEST(q.pop() == 31);
  BOOST_TEST(q.size() == 0);
  BOOST_TEST(q.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DoublesCapacityAfterPops)
{
  AsHierarchy::DynamicQueue<int> q {16};

  for (int i {0}; i < 16; ++i)
  {
    q.push(i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == i + 1);
    BOOST_TEST(q.front() == 0);
  }

  for (int i {0}; i < 6; ++i)
  {
    BOOST_TEST(q.pop() == i);
    BOOST_TEST(q.size() == 15 - i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.front() == i + 1);
  }

  for (int i {0}; i < 6; ++i)
  {
    q.push(i + 16);
    BOOST_TEST(q.size() == i + 11);
    BOOST_TEST(q.front() == 6);
    BOOST_TEST(!q.is_empty());
  }

  for (int i {0}; i < 3; ++i)
  {
    q.push(22 + i);
    BOOST_TEST(q.size() == 17 + i);
    BOOST_TEST(q.front() == 6);
    BOOST_TEST(!q.is_empty());
  }

  for (int i {0}; i < 18; ++i)
  {
    BOOST_TEST(q.pop() == i + 6);
    BOOST_TEST(q.size() == 18 - i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.front() == i + 7);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DoubleCapacityCopiesOverPreviousArrayFromArrayStart)
{
  TestDynamicQueueAsHierarchy<int> q {16};

  for (int i {0}; i < 16; ++i)
  {
    q.push(i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == i + 1);
    BOOST_TEST(q.front() == 0);
  }

  BOOST_TEST(q.is_full());
  BOOST_TEST(q.back() == 15);
  BOOST_TEST(q.get_front_index() == 0);
  BOOST_TEST(q.get_back_index() == 15);
  BOOST_TEST(q.get_size() == 16);
  BOOST_TEST(q.get_array_capacity() == 16);

  for (int i {0}; i < 6; ++i)
  {
    BOOST_TEST(q.pop() == i);
    BOOST_TEST(q.size() == 15 - i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.front() == i + 1);
  }

  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.back() == 15);
  BOOST_TEST(q.get_front_index() == 6);
  BOOST_TEST(q.get_back_index() == 15);
  BOOST_TEST(q.get_size() == 10);
  BOOST_TEST(q.get_array_capacity() == 16);

  for (int i {0}; i < 6; ++i)
  {
    q.push(i + 16);
    BOOST_TEST(q.size() == i + 11);
    BOOST_TEST(q.front() == 6);
    BOOST_TEST(!q.is_empty());
  }

  BOOST_TEST(q.is_full());
  BOOST_TEST(q.back() == 21);
  BOOST_TEST(q.get_front_index() == 6);
  BOOST_TEST(q.get_back_index() == 5);
  BOOST_TEST(q.get_size() == 16);
  BOOST_TEST(q.get_array_capacity() == 16);

  q.push(22);
  BOOST_TEST(q.size() == 17);
  BOOST_TEST(q.front() == 6);
  BOOST_TEST(!q.is_empty());
  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.back() == 22);
  BOOST_TEST(q.get_front_index() == 0);
  BOOST_TEST(q.get_back_index() == 16);
  BOOST_TEST(q.get_size() == 17);
  BOOST_TEST(q.get_array_capacity() == 32);

  for (int i {1}; i < 3; ++i)
  {
    q.push(22 + i);
    BOOST_TEST(q.size() == 17 + i);
    BOOST_TEST(q.front() == 6);
    BOOST_TEST(!q.is_empty());
  }

  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.back() == 24);
  BOOST_TEST(q.get_front_index() == 0);
  BOOST_TEST(q.get_back_index() == 18);
  BOOST_TEST(q.get_size() == 19);
  BOOST_TEST(q.get_array_capacity() == 32);

  for (int i {0}; i < 18; ++i)
  {
    BOOST_TEST(q.pop() == i + 6);
    BOOST_TEST(q.size() == 18 - i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.front() == i + 7);
  }
}

//------------------------------------------------------------------------------
/// \ref pp. 234, Ch. 10 Elementary Data Structures, Introduction to Algorithms,
/// Cormen, Leiserson, Rivest, Stein.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HeadPointsToFirstElement)
{
  TestDynamicQueueAsHierarchy<int> q {12};

  for (int i {1}; i < 7; ++i)
  {
    q.enqueue(i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == i);
    BOOST_TEST(q.front() == 1);
    BOOST_TEST(q.get_front_index() == 0);
    BOOST_TEST(q.get_back_index() == i - 1);
    BOOST_TEST(q.get_array_capacity() == 12);
  }

  q.enqueue(15);
  q.enqueue(6);
  q.enqueue(9);
  q.enqueue(8);
  q.enqueue(4);
  BOOST_TEST(q.size() == 11);
  BOOST_TEST(q.get_back_index() == 10);
  BOOST_TEST(q.get_array_capacity() == 12);

  for (int i {1}; i < 7; ++i)
  {
    BOOST_TEST(q.dequeue() == i);
    BOOST_TEST(q.get_front_index() == i);
  }

  BOOST_TEST(q.head() == 15);
  BOOST_TEST(q.get_front_index() == 6);
  BOOST_TEST(q.get_back_index() == 10);
  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.get_array_capacity() == 12);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CircularlyWrapsAroundIfNotFull)
{
  TestDynamicQueueAsHierarchy<int> q {12};

  for (int i {1}; i < 7; ++i)
  {
    q.enqueue(i);
    BOOST_TEST(!q.is_full());
  }

  q.enqueue(15);
  q.enqueue(6);
  q.enqueue(9);
  q.enqueue(8);
  q.enqueue(4);
  BOOST_TEST(!q.is_full());
  q.enqueue(17);
  BOOST_TEST(q.is_full());
  BOOST_TEST(q.get_array_capacity() == 12);

  for (int i {1}; i < 7; ++i)
  {
    BOOST_TEST(q.dequeue() == i);
    BOOST_TEST(q.get_front_index() == i);
    BOOST_TEST(!q.is_full());
  }

  
}

BOOST_AUTO_TEST_SUITE_END() // AsHierarchy_tests
BOOST_AUTO_TEST_SUITE_END() // DynamicQueue_tests
BOOST_AUTO_TEST_SUITE_END() // Queues
BOOST_AUTO_TEST_SUITE_END() // DataStructures