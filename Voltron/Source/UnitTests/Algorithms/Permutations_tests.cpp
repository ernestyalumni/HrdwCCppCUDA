//------------------------------------------------------------------------------
// \file Permutations_tests.cpp
//------------------------------------------------------------------------------
#include "Algorithms/Permutations.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Algorithms::Permutations::Details::single_swap;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Permutations_tests)

// https://www.oreilly.com/library/view/optimized-c/9781491922057/ch04.html
// String problems:
// strings are dynamically allocated
// amoritizes cost of reallocating storage for character buffer as string grows
// allocated dynamically once any operation makes string longer
// Cost is unused space.
// Strings are values (in expressions)
// must copy
// lots of call to memory manager (to dynamically allocate to temp string)
// alot of copying
// in C++11 no copy-on-write; must copy.
// Can use r-value move semantics in C++11 string, at least.
BOOST_AUTO_TEST_SUITE(StringExamples)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateString)
{
  // https://en.cppreference.com/w/cpp/string/basic_string/replace
  // Replace
  {
    std::string a_string {"The quick brown fox jumps over the lazy dog."};

    a_string.replace(10, 5, "red"); // (5)

    a_string.replace(a_string.begin(), a_string.begin() + 3, 1, 'A'); // (6)

    std::string target {"A quick red fox jumps over the lazy dog."};
    BOOST_TEST(a_string == target);
  }

}

BOOST_AUTO_TEST_SUITE_END() // StringExamples

BOOST_AUTO_TEST_SUITE(Details_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateSingleSwap)
{
  {
    std::string a {"Quick"};

    single_swap(a, 2, 4);

    BOOST_TEST(a == "Qukci");
  }

}

BOOST_AUTO_TEST_SUITE_END() // Details_tests

BOOST_AUTO_TEST_SUITE_END() // Permutations_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms