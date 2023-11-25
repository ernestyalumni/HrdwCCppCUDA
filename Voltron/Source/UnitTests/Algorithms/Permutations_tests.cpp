//------------------------------------------------------------------------------
// \file Permutations_tests.cpp
//------------------------------------------------------------------------------
#include "Algorithms/Permutations.h"

#include "Tools/CaptureCout.h"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <string>
#include <sstream>
#include <vector>

using Algorithms::Permutations::Details::single_swap;
using Algorithms::Permutations::Details::print_permutations;
using Tools::CaptureCoutFixture;
using std::string;

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

// cf. https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateSingleSwap)
{
  {
    std::string a {"Quick"};

    single_swap(a, 2, 4);

    BOOST_TEST(a == "Qukci");
  }
  {
    std::string a {"abcdefghijklmnopqrstuvwxyz"};

    single_swap(a, 4, 2);

    BOOST_TEST(a == "abedcfghijklmnopqrstuvwxyz");
  }
  {
    std::string a {"abcdefghijklmnopqrstuvwxyz"};

    single_swap(a, 5, 2);

    BOOST_TEST(a == "abfdecghijklmnopqrstuvwxyz");

    single_swap(a, 3, 8);

    BOOST_TEST(a == "abfiecghdjklmnopqrstuvwxyz");
  }
}

// cf. https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(DemonstratePrintPermutations, CaptureCoutFixture)
{
  std::string a = "ABC";
  print_permutations(a, 0, a.length() - 1);

  const string expected {
    string{" Second swap : ABC "} +
    string{" Second swap : ABC "} +
    string{" Second swap : ACB "} +
    string{" Second swap : ABC "} + 
    string{" Second swap : ABC "} +
    string{" Second swap : BAC "} +
    string{" Second swap : BAC "} +
    string{" Second swap : BCA "} +
    string{" Second swap : BAC "} +
    string{" Second swap : ABC "} +
    string{" Second swap : CBA "} +
    string{" Second swap : CBA "} +
    string{" Second swap : CAB "} +
    string{" Second swap : CBA "} +
    string{" Second swap : ABC "}};

  BOOST_TEST(local_oss_.str() == expected);  

  BOOST_TEST(true);
}

// cf. https://en.cppreference.com/w/cpp/algorithm/rotate
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateStdRotate)
{
  {
    std::stringstream string_stream;
    // cf. https://en.cppreference.com/w/cpp/algorithm/rotate
    std::vector<int> v {2, 4, 2, 0, 5, 10, 7, 3, 7, 1};

    // Before sort
    for (int n : v)
    {
      string_stream << n << " ";
    }

    BOOST_TEST(string_stream.str() == "2 4 2 0 5 10 7 3 7 1 ");

    string_stream.str(std::string{});
  }
}

BOOST_AUTO_TEST_SUITE_END() // Details_tests

BOOST_AUTO_TEST_SUITE_END() // Permutations_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms