//------------------------------------------------------------------------------
/// \file EqualAndMismatch_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/algorithm/equal
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>

#include <algorithm> // std::equal
#include <string>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Algorithms_tests)

// cf. https://en.cppreference.com/w/cpp/algorithm/equal
bool is_palindrome(const std::string& s)
{
  return std::equal(s.begin(), s.begin() + s.size() / 2, s.rbegin());
}

std::string palindrome_test(const std::string& s)
{
  return s + " " + (is_palindrome(s) ? "is" : "is not") + " a palindrome";
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdEqualComparesStrings)
{
  BOOST_TEST(true);

}

BOOST_AUTO_TEST_SUITE_END() // Algorithms_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms
BOOST_AUTO_TEST_SUITE_END() // Cpp
