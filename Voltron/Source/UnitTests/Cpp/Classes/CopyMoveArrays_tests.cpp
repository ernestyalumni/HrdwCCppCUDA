#include "Cpp/Classes/CopyMoveArrays.h"

// For Performance testing.

#include "Utilities/Time/GetElapsedTime.h"
#include "Utilities/Time/TimeSpec.h"

#include <iostream>
#include <vector>

using Utilities::Time::GetElapsedTime;
using Utilities::Time::TimeSpec;
using std::cout;
using std::vector;

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <initializer_list>
#include <utility>

using Cpp::Classes::DefaultWithArray;
using Cpp::Classes::WithArray;
using std::initializer_list;
using std::size_t;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Classes)
BOOST_AUTO_TEST_SUITE(CopyMoveArrays_tests)

BOOST_AUTO_TEST_SUITE(WithArray_tests)

template <typename T, std::size_t N>
WithArray<T, N> create_rvalue_with_array(const initializer_list<T>& input)
{
  WithArray<T, N> wa {input};
  return wa;
}

template <typename T, std::size_t N>
WithArray<T, N> addition_with_array_rvalue(
  const WithArray<T, N>& a,
  const WithArray<T, N>& b)
{
  WithArray<T, N> wa {a + b};
  return wa;
}

template <typename T, std::size_t N>
DefaultWithArray<T, N> addition_with_array_rvalue(
  const DefaultWithArray<T, N>& a,
  const DefaultWithArray<T, N>& b)
{
  DefaultWithArray<T, N> wa {a + b};
  return wa;
}

template <typename T, std::size_t N>
WithArray<T, N> create_copy_elision_with_array(const initializer_list<T>& input)
{
  return WithArray<T, N>{input};
}

template <typename T, std::size_t N>
WithArray<T, N> addition_with_array_copy_elision(
  const WithArray<T, N>& a,
  const WithArray<T, N>& b)
{
  return a + b;
}

template <typename T, std::size_t N>
DefaultWithArray<T, N> addition_with_array_copy_elision(
  const DefaultWithArray<T, N>& a,
  const DefaultWithArray<T, N>& b)
{
  return a + b;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  WithArray<double, 3> a {};
  BOOST_TEST(a[0] == 0);
  BOOST_TEST(a[1] == 0);
  BOOST_TEST(a[2] == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInitializerList)
{
  WithArray<double, 3> a {1, 2, 3};
  BOOST_TEST(a[0] == 1);
  BOOST_TEST(a[1] == 2);
  BOOST_TEST(a[2] == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyConstructorCopies)
{
  WithArray<double, 3> a {1, 2, 3};
  BOOST_TEST(a[0] == 1);
  BOOST_TEST(a[1] == 2);
  BOOST_TEST(a[2] == 3);

  WithArray<double, 3> b {a};
  BOOST_TEST(b[0] == 1);
  BOOST_TEST(b[1] == 2);
  BOOST_TEST(b[2] == 3);

  // Source unchanged.
  BOOST_TEST(a[0] == 1);
  BOOST_TEST(a[1] == 2);
  BOOST_TEST(a[2] == 3);

  a.set_value(0, 42);
  BOOST_TEST(a[0] == 42);
  BOOST_TEST(b[0] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyAssignmentCopies)
{
  WithArray<double, 3> a {1, 2, 3};
  BOOST_TEST(a[0] == 1);
  BOOST_TEST(a[1] == 2);
  BOOST_TEST(a[2] == 3);

  WithArray<double, 3> b = a;
  BOOST_TEST(b[0] == 1);
  BOOST_TEST(b[1] == 2);
  BOOST_TEST(b[2] == 3);

  // Source unchanged.
  BOOST_TEST(a[0] == 1);
  BOOST_TEST(a[1] == 2);
  BOOST_TEST(a[2] == 3);

  a.set_value(0, 42);
  BOOST_TEST(a[0] == 42);
  BOOST_TEST(b[0] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveConstructorSwaps)
{
  WithArray<double, 3> a {1, 2, 3};
  BOOST_TEST(a[0] == 1);
  BOOST_TEST(a[1] == 2);
  BOOST_TEST(a[2] == 3);

  WithArray<double, 3> b {std::move(a)};
  BOOST_TEST(b[0] == 1);
  BOOST_TEST(b[1] == 2);
  BOOST_TEST(b[2] == 3);

  // Source swapped.
  BOOST_TEST(a[0] == 0);
  BOOST_TEST(a[1] == 0);
  BOOST_TEST(a[2] == 0);

  a.set_value(0, 42);
  BOOST_TEST(a[0] == 42);
  BOOST_TEST(b[0] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveAssignmentSwaps)
{
  WithArray<double, 3> a {1, 2, 3};
  BOOST_TEST(a[0] == 1);
  BOOST_TEST(a[1] == 2);
  BOOST_TEST(a[2] == 3);

  WithArray<double, 3> b = std::move(a);
  BOOST_TEST(b[0] == 1);
  BOOST_TEST(b[1] == 2);
  BOOST_TEST(b[2] == 3);

  // Source swapped.
  BOOST_TEST(a[0] == 0);
  BOOST_TEST(a[1] == 0);
  BOOST_TEST(a[2] == 0);

  a.set_value(0, 42);
  BOOST_TEST(a[0] == 42);
  BOOST_TEST(b[0] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MoveAssignmentFromFunctionReturn)
{
  {
    WithArray<double, 3> a {1, 2, 3};
    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 2);
    BOOST_TEST(a[2] == 3);

    a = create_rvalue_with_array<double, 3>({4, 5, 6});
    BOOST_TEST(a[0] == 4);
    BOOST_TEST(a[1] == 5);
    BOOST_TEST(a[2] == 6);
  }

  {
    WithArray<double, 3> a {1, 2, 3};
    BOOST_TEST(a[0] == 1);
    BOOST_TEST(a[1] == 2);
    BOOST_TEST(a[2] == 3);

    a = create_copy_elision_with_array<double, 3>({4, 5, 6});
    BOOST_TEST(a[0] == 4);
    BOOST_TEST(a[1] == 5);
    BOOST_TEST(a[2] == 6);
  }
}

// Uncomment out to run this.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/*
BOOST_AUTO_TEST_CASE(PerformanceComparison)
{
  cout << "\n Performance Comparsion For WithArray" << "\n";

  GetElapsedTime elapsed_time {};

  vector<WithArray<double, 9>> wav {};
  vector<DefaultWithArray<double, 9>> dwav {};

  constexpr size_t N {60'000'000};

  elapsed_time.start();

  for (size_t i {0}; i < N; ++i)
  {
    WithArray<double, 9> wai {
      0.1 + 0.01 * static_cast<double>(i),
      0.2 + 0.01 * static_cast<double>(i),
      0.3 + 0.01 * static_cast<double>(i),
      0.4 + 0.01 * static_cast<double>(i),
      0.5 + 0.01 * static_cast<double>(i),
      0.6 + 0.01 * static_cast<double>(i),
      0.7 + 0.01 * static_cast<double>(i),
      0.8 + 0.01 * static_cast<double>(i),
      0.9 + 0.01 * static_cast<double>(i)};

    wav.emplace_back(wai);
  }

  TimeSpec resulting_t {elapsed_time()};

  cout << "\n WithArray init : " << resulting_t << "\n";

  BOOST_TEST(wav.size() == N);

  elapsed_time.start();

  for (size_t i {0}; i < N; ++i)
  {
    DefaultWithArray<double, 9> dwai {
      0.1 + 0.01 * static_cast<double>(i),
      0.2 + 0.01 * static_cast<double>(i),
      0.3 + 0.01 * static_cast<double>(i),
      0.4 + 0.01 * static_cast<double>(i),
      0.5 + 0.01 * static_cast<double>(i),
      0.6 + 0.01 * static_cast<double>(i),
      0.7 + 0.01 * static_cast<double>(i),
      0.8 + 0.01 * static_cast<double>(i),
      0.9 + 0.01 * static_cast<double>(i)};

    dwav.emplace_back(dwai);
  }

  resulting_t = elapsed_time();

  cout << "\n DefaultWithArray init : " << resulting_t << "\n";

  BOOST_TEST(dwav.size() == N);

  elapsed_time.start();

  for (size_t i {0}; i < N - 1; ++i)
  {
    WithArray<double, 9> waf {
      addition_with_array_rvalue(wav[i], wav[i + 1])};
  }

  resulting_t = elapsed_time();

  cout << "\n WithArray r value add : " << resulting_t << "\n";

  elapsed_time.start();

  for (size_t i {0}; i < N - 1; ++i)
  {
    DefaultWithArray<double, 9> dwaf {
      addition_with_array_rvalue(dwav[i], dwav[i + 1])};
  }

  resulting_t = elapsed_time();

  cout << "\n DefaultWithArray r value add : " << resulting_t << "\n";

  elapsed_time.start();

  for (size_t i {0}; i < N - 1; ++i)
  {
    WithArray<double, 9> waf {
      addition_with_array_copy_elision(wav[i], wav[i + 1])};
  }

  resulting_t = elapsed_time();

  cout << "\n WithArray copy elision add : " << resulting_t << "\n";

  elapsed_time.start();

  for (size_t i {0}; i < N - 1; ++i)
  {
    DefaultWithArray<double, 9> dwaf {
      addition_with_array_copy_elision(dwav[i], dwav[i + 1])};
  }

  resulting_t = elapsed_time();

  cout << "\n DefaultWithArray copy elision add : " << resulting_t << "\n";

}
*/
BOOST_AUTO_TEST_SUITE_END() // WithArray_tests

BOOST_AUTO_TEST_SUITE_END() // CopyMoveArrays_tests
BOOST_AUTO_TEST_SUITE_END() // Classes
BOOST_AUTO_TEST_SUITE_END() // Cpp
