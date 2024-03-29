//------------------------------------------------------------------------------
/// \file Recursion_tests.cpp
///
/// \ref cf. https://www.bogotobogo.com/cplusplus/functors.php
//------------------------------------------------------------------------------
#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <algorithm> // std::for_each
#include <iostream>
#include <string>
#include <vector>

using Tools::CaptureCoutFixture;
using std::string;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Functions_tests)

// https://gitlab.com/manning-fpcpp-book/code-examples/blob/master/chapter-06/fibonacci/main.cpp

// O(2^n) complexity
unsigned int fibonacci_recursion(unsigned int n)
{
  return n == 0 ? 0 :
    n == 1 ? 1 :
      fibonacci_recursion(n - 1) + fibonacci_recursion(n - 2);  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FibonacciOrdinaryRecursion)
{
  BOOST_TEST(fibonacci_recursion(0) == 0);
  BOOST_TEST(fibonacci_recursion(1) == 1);
  BOOST_TEST(fibonacci_recursion(2) == 1);
  BOOST_TEST(fibonacci_recursion(3) == 2);
  BOOST_TEST(fibonacci_recursion(4) == 3);
  BOOST_TEST(fibonacci_recursion(5) == 5);
  BOOST_TEST(fibonacci_recursion(6) == 8);
  BOOST_TEST(fibonacci_recursion(7) == 13);
}

// Cached Fibonacci
std::vector<unsigned int> cache {0, 1};

// Implementation of the Fibonacci function which caches all previously
// calculated results.
unsigned int fibonacci_cached(unsigned int n)
{
  if (cache.size() > n)
  {
    std::cout << "using cache " << n << "\n";
    return cache[n];
  }
  else
  {
    const auto result = fibonacci_cached(n - 1) + fibonacci_cached(n - 2);
    cache.push_back(result);
    return result;
  }
}

// USE functors.
// cf. https://www.bogotobogo.com/cplusplus/functors.php
template <typename T>
class PrintElements
{
  public:
    void operator()(const T& element)
    {
      std::cout << element << ' ';
    }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(FibonacciCachedRecursion, CaptureCoutFixture)
{
  std::for_each(cache.begin(), cache.end(), PrintElements<unsigned int>{});

  BOOST_TEST(fibonacci_cached(0) == 0);
  BOOST_TEST(fibonacci_cached(1) == 1);
  BOOST_TEST(fibonacci_cached(2) == 1);
  BOOST_TEST(fibonacci_cached(3) == 2);
  BOOST_TEST(fibonacci_cached(4) == 3);
  BOOST_TEST(fibonacci_cached(5) == 5);
  BOOST_TEST(fibonacci_cached(6) == 8);

  const string expected {
    string{"0 1 "} +
    string{"using cache 0\n"} +
    string{"using cache 1\n"} +
    string{"using cache 1\n"} + 
    string{"using cache 0\n"} +
    string{"using cache 2\n"} +
    string{"using cache 1\n"} +
    string{"using cache 3\n"} +
    string{"using cache 2\n"} +
    string{"using cache 4\n"} +
    string{"using cache 3\n"} +
    string{"using cache 5\n"} +
    string{"using cache 4\n"}};

  BOOST_TEST(local_oss_.str() == expected);
}

BOOST_AUTO_TEST_SUITE_END() // Functions_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp