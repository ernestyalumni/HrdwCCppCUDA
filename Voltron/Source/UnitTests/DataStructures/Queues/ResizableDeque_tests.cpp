#include "DataStructures/Queues/ResizableDeque.h"

#include <boost/test/unit_test.hpp>
#include <cstddef>

using namespace DataStructures::Deques;

template <typename T>
class TestResizableDequeAsHierarchy : public AsHierarchy::ResizableDeque<T>
{
  public:

    // Derived class needs a constructor.
    using AsHierarchy::ResizableDeque<T>::ResizableDeque;

    using AsHierarchy::ResizableDeque<T>::is_full;
    using AsHierarchy::ResizableDeque<T>::get_front_index;
    using AsHierarchy::ResizableDeque<T>::get_back_index;
    using AsHierarchy::ResizableDeque<T>::get_array_capacity;
};

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queues)
BOOST_AUTO_TEST_SUITE(ResizableDeque_tests)
BOOST_AUTO_TEST_SUITE(AsHierarchy_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  AsHierarchy::ResizableDeque<int> d;

  BOOST_TEST(d.is_empty());
  BOOST_TEST(d.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithSizeValue)
{
  AsHierarchy::ResizableDeque<int> d {16};

  BOOST_TEST(d.is_empty());
  BOOST_TEST(d.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushFrontAppendsFirstValueToFront)
{
  AsHierarchy::ResizableDeque<int> d {16};

  d.push_front(3);
  BOOST_TEST(d.front() == 3);
  BOOST_TEST(d.back() == 3);
  BOOST_TEST(d.size() == 1);
  BOOST_TEST(!d.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushBackAppendsFirstValueToBack)
{
  AsHierarchy::ResizableDeque<int> d {16};

  d.push_back(5);
  BOOST_TEST(d.front() == 5);
  BOOST_TEST(d.back() == 5);
  BOOST_TEST(d.size() == 1);
  BOOST_TEST(!d.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushFrontAndPushBackWorks)
{
  AsHierarchy::ResizableDeque<int> d {16};

  d.push_front(3);
  BOOST_TEST(d.front() == 3);
  BOOST_TEST(d.back() == 3);
  BOOST_TEST(d.size() == 1);
  BOOST_TEST(!d.is_empty());

  d.push_back(5);
  BOOST_TEST(d.front() == 3);
  BOOST_TEST(d.back() == 5);
  BOOST_TEST(d.size() == 2);
  BOOST_TEST(!d.is_empty());

  d.push_front(2);
  BOOST_TEST(d.front() == 2);
  BOOST_TEST(d.back() == 5);
  BOOST_TEST(d.size() == 3);
  BOOST_TEST(!d.is_empty());

  d.push_front(15);
  BOOST_TEST(d.front() == 15);
  BOOST_TEST(d.back() == 5);
  BOOST_TEST(d.size() == 4);
  BOOST_TEST(!d.is_empty());

  d.push_back(42);
  BOOST_TEST(d.front() == 15);
  BOOST_TEST(d.back() == 42);
  BOOST_TEST(d.size() == 5);
  BOOST_TEST(!d.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PopFrontWorksWithPushFrontToArrayCapacity)
{
  constexpr std::size_t N {8};

  TestResizableDequeAsHierarchy<int> d {N};

  d.push_front(0);
  BOOST_TEST_REQUIRE(!d.is_empty());
  BOOST_TEST_REQUIRE(d.size() == 1);
  BOOST_TEST(d.front() == 0);
  BOOST_TEST(d.back() == 0);

  for (int i {1}; i < N; ++i)
  {
    d.push_front(N - i);
    BOOST_TEST(d.front() == N - i);
    BOOST_TEST(d.back() == 0);
    BOOST_TEST(d.size() == 1 + i);
    BOOST_TEST(!d.is_empty());
  }  

  BOOST_TEST(d.is_full());

  for (int i {1}; i < N; ++i)
  {
    BOOST_TEST(d.pop_front() == i);
    BOOST_TEST(d.back() == 0);
    BOOST_TEST(d.size() == N - i);
  }

  BOOST_TEST(d.pop_front() == 0);
  BOOST_TEST(d.is_empty());
  BOOST_TEST(d.size() == 0);   
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PopBackWorksWithPushBackToArrayCapacity)
{
  constexpr std::size_t N {8};

  TestResizableDequeAsHierarchy<int> d {N};

  d.push_back(0);
  BOOST_TEST_REQUIRE(!d.is_empty());
  BOOST_TEST_REQUIRE(d.size() == 1);
  BOOST_TEST(d.front() == 0);
  BOOST_TEST(d.back() == 0);

  for (int i {1}; i < N; ++i)
  {
    d.push_back(i);
    BOOST_TEST(d.front() == 0);
    BOOST_TEST(d.back() == i);
    BOOST_TEST(d.size() == 1 + i);
    BOOST_TEST(!d.is_empty());
  }  

  BOOST_TEST(d.is_full());

  for (int i {1}; i < N; ++i)
  {
    BOOST_TEST(d.pop_back() == N - i);
    BOOST_TEST(d.back() == N - 1 - i);
    BOOST_TEST(d.size() == N - i);
  }

  BOOST_TEST(d.pop_back() == 0);
  BOOST_TEST(d.is_empty());
  BOOST_TEST(d.size() == 0);   
}

BOOST_AUTO_TEST_SUITE_END() // AsHierarchy_tests
BOOST_AUTO_TEST_SUITE_END() // ResizableDeque_tests
BOOST_AUTO_TEST_SUITE_END() // Deques
BOOST_AUTO_TEST_SUITE_END() // DataStructures