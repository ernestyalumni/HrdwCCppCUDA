//------------------------------------------------------------------------------
/// \file Vector_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/Std/Containers/SpanView.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using Cpp::Std::Containers::SpanView;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Std)
BOOST_AUTO_TEST_SUITE(Containers)
BOOST_AUTO_TEST_SUITE(Vector_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructVector)
{
  vector<char> a {};

  BOOST_TEST(a.empty());
  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.max_size() > 0);
  BOOST_TEST(a.capacity() == 0);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReserveIncreasesCapacity)
{
  vector<char> a {};
  // 2**10 = 1024
  constexpr int N_max {1024};
  a.reserve(N_max);
  BOOST_TEST(a.empty());
  // Size, number of elements, stays the same.
  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.capacity() == N_max);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertAssignElementsAndIncreaseSize)
{
  vector<char> a {};
  constexpr int N_max {1024};
  a.reserve(N_max);
  string test_string {"differential"};
  SpanView<char> span_view {test_string.data(), test_string.size()};
  a.insert(a.end(), span_view.begin(), span_view.end());

  for (int i {0}; i < test_string.size(); ++i)
  {
    BOOST_TEST(a[i] == test_string[i]);
    BOOST_TEST(a[i] == span_view[i]);
  }
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == test_string.size());
  BOOST_TEST(a.capacity() == N_max);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClearClearsContentAndSize)
{
  vector<char> a {};
  constexpr int N_max {1024};
  a.reserve(N_max);
  string test_string {"differential"};
  a.insert(a.end(), test_string.begin(), test_string.end());
  for (int i {0}; i < test_string.size(); ++i)
  {
    BOOST_TEST_REQUIRE(a[i] == test_string[i]);
  }

  // void clear();
  // cf. https://en.cppreference.com/w/cpp/container/vector/clear
  a.clear();
  BOOST_TEST(a.empty());
  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.capacity() == N_max);
}

BOOST_AUTO_TEST_SUITE_END() // Vector_tests
BOOST_AUTO_TEST_SUITE_END() // Containers
BOOST_AUTO_TEST_SUITE_END() // Std
BOOST_AUTO_TEST_SUITE_END() // Cpp