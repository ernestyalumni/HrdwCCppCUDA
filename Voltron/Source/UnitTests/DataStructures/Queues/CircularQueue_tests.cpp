#include "DataStructures/Queues/CircularQueue.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using namespace DataStructures::Queues;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queues)
BOOST_AUTO_TEST_SUITE(CircularQueue_tests)
BOOST_AUTO_TEST_SUITE(AsHierarchy_tests)

template <typename T>
class TestCircularQueueAsHierarchy :
  public AsHierarchy::CLRS::CircularQueue<T>
{
  public:

    // Derived class needs a constructor.
    using AsHierarchy::CLRS::CircularQueue<T>::CircularQueue;

    using AsHierarchy::CLRS::CircularQueue<T>::get_head;
    using AsHierarchy::CLRS::CircularQueue<T>::get_tail;
    using AsHierarchy::CLRS::CircularQueue<T>::get_array_capacity;
    using AsHierarchy::CLRS::CircularQueue<T>::is_null_array;    
};

class CircularQueueAsHierarchyFixture
{
  public:

    CircularQueueAsHierarchyFixture():
      rhs_q_{12}
    {
      for (int i {1}; i < 7; ++i)
      {
        rhs_q_.enqueue(i);
      }

      rhs_q_.enqueue(15);
      rhs_q_.enqueue(6);
      rhs_q_.enqueue(9);
      rhs_q_.enqueue(8);
      rhs_q_.enqueue(4);

      for (int i {1}; i < 7; ++i)
      {
        BOOST_TEST_REQUIRE(rhs_q_.dequeue() == i);
      }
    }

    virtual ~CircularQueueAsHierarchyFixture() = default;

    TestCircularQueueAsHierarchy<int> rhs_q_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  TestCircularQueueAsHierarchy<int> q {};
  BOOST_TEST(q.get_array_capacity() == 10);
  BOOST_TEST(q.size() == 0);
  BOOST_TEST(q.get_head() == 0);
  BOOST_TEST(q.get_tail() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CopyConstructs, CircularQueueAsHierarchyFixture)
{
  TestCircularQueueAsHierarchy q {rhs_q_};

  BOOST_TEST(q.get_array_capacity() == 12);
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.get_head() == 6);
  BOOST_TEST(q.get_tail() == 11);

  BOOST_TEST(rhs_q_.get_array_capacity() == 12);
  BOOST_TEST(rhs_q_.size() == 5);
  BOOST_TEST(!rhs_q_.is_full());
  BOOST_TEST(rhs_q_.get_head() == 6);
  BOOST_TEST(rhs_q_.get_tail() == 11);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CopyAssignmentCopies, CircularQueueAsHierarchyFixture)
{
  TestCircularQueueAsHierarchy<int> q {};

  q = rhs_q_;

  BOOST_TEST(q.get_array_capacity() == 12);
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.get_head() == 6);
  BOOST_TEST(q.get_tail() == 11);
  BOOST_TEST(!q.is_null_array());

  BOOST_TEST(rhs_q_.get_array_capacity() == 12);
  BOOST_TEST(rhs_q_.size() == 5);
  BOOST_TEST(!rhs_q_.is_full());
  BOOST_TEST(rhs_q_.get_head() == 6);
  BOOST_TEST(rhs_q_.get_tail() == 11);
  BOOST_TEST(!rhs_q_.is_null_array());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveConstructs, CircularQueueAsHierarchyFixture)
{
  TestCircularQueueAsHierarchy<int> q {std::move(rhs_q_)};

  BOOST_TEST(q.get_array_capacity() == 12);
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.get_head() == 6);
  BOOST_TEST(q.get_tail() == 11);
  BOOST_TEST(!q.is_null_array());

  BOOST_TEST(rhs_q_.get_array_capacity() == 12);
  BOOST_TEST(rhs_q_.size() == 5);
  BOOST_TEST(!rhs_q_.is_full());
  BOOST_TEST(rhs_q_.get_head() == 6);
  BOOST_TEST(rhs_q_.get_tail() == 11);    
  BOOST_TEST(rhs_q_.is_null_array());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveAssigns, CircularQueueAsHierarchyFixture)
{
  TestCircularQueueAsHierarchy<int> q {};

  q = std::move(rhs_q_);

  BOOST_TEST(q.get_array_capacity() == 12);
  BOOST_TEST(q.size() == 5);
  BOOST_TEST(!q.is_full());
  BOOST_TEST(q.get_head() == 6);
  BOOST_TEST(q.get_tail() == 11);
  BOOST_TEST(!q.is_null_array());

  BOOST_TEST(rhs_q_.get_array_capacity() == 12);
  BOOST_TEST(rhs_q_.size() == 5);
  BOOST_TEST(!rhs_q_.is_full());
  BOOST_TEST(rhs_q_.get_head() == 6);
  BOOST_TEST(rhs_q_.get_tail() == 11);    
  BOOST_TEST(rhs_q_.is_null_array());
}

//------------------------------------------------------------------------------
/// \ref pp. 235 10.1-3 Ch. 10 "Elementary Data Structures" of Cormen,
/// Leiserson, Rivest, Stein.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(EnqueuesAndDequeues)
{
  AsHierarchy::CLRS::CircularQueue<int> q {6};

  q.enqueue(4);
  q.enqueue(1);
  q.enqueue(3);
  BOOST_TEST(q.dequeue() == 4);
  q.enqueue(8);
  BOOST_TEST(q.dequeue() == 1);
  BOOST_TEST(q.dequeue() == 3);
  BOOST_TEST(q.dequeue() == 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FillsToCapacityAndBack)
{
  // Remember, full capacity is considered to be when there are N - 1 elements
  // in a N-sized array, see
  // https://stackoverflow.com/questions/16395354/why-q-head-q-tail-1-represents-the-queue-is-full-in-clrs
  AsHierarchy::CLRS::CircularQueue<int> q {17};

  for (int i {0}; i < 16; ++i)
  {
    q.enqueue(i);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == i + 1);
  }

  BOOST_TEST(q.is_full());

  for (int i {0}; i < 5; ++i)
  {
    BOOST_TEST(q.dequeue() == i);
    BOOST_TEST(q.size() == 15 - i);
  }

  for (int i {0}; i < 5; ++i)
  {
    q.enqueue(i + 16);
    BOOST_TEST(!q.is_empty());
    BOOST_TEST(q.size() == 12 + i);
  }

  BOOST_TEST(q.is_full());
}

BOOST_AUTO_TEST_SUITE_END() // AsHierarchy_tests
BOOST_AUTO_TEST_SUITE_END() // CircularQueue_tests
BOOST_AUTO_TEST_SUITE_END() // Queues
BOOST_AUTO_TEST_SUITE_END() // DataStructures