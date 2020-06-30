//------------------------------------------------------------------------------
/// \file BitManipulationLonelyInteger_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \details Span is a class template and belongs in Containers library.
/// \ref https://www.hackerrank.com/challenges/ctci-lonely-integer/problem
/// \details To run only these unit tests, do this:
/// ./Check --run_test="Algorithms/HackerRank"
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <functional>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(HackerRank)
BOOST_AUTO_TEST_SUITE(BitManipulation)
BOOST_AUTO_TEST_SUITE(LonelyInteger_tests)

std::string ltrim(const std::string&);
std::string rtrim(const std::string&);
std::vector<std::string> split(const std::string&);

// Complete the findLonely function below.
int findLonely(std::vector<int> arr)
{
  return 0;
}

//------------------------------------------------------------------------------

std::string ltrim(const std::string& str)
{
  std::string s(str);

  s.erase(
    s.begin(),
    std::find_if(
      s.begin(),
      s.end(),
      std::not1(ptr_fun<int, int>(std::isspace))));

  return s;
}

std::string rtrim(const std::string& str)
{
  std::string s(str);

  s.erase(
    std::find_if(
      s.rbegin(),
      s.rend(),
      not1(ptr_fun<int, int>(std::isspace))).base(),
    s.end());

  return s;
}

std::vector<std::string> split(const std::string& str)
{
  std::vector<std::string> tokens;

  std::string::size_type start = 0;
  std::string::size_type end = 0;

  while ((end = str.find(" ", start)) != std::string::npos)
  {
    tokens.push_back(str.substr(start, end - start));

    start = end + 1;
  }

  tokens.push_back(str.substr(start));

  return tokens;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitwiseLeftShiftOfNegativeNumbers)
{
  BOOST_TEST(true);
}


BOOST_AUTO_TEST_SUITE_END() // LonelyInteger_tests
BOOST_AUTO_TEST_SUITE_END() // HackerRank
BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms