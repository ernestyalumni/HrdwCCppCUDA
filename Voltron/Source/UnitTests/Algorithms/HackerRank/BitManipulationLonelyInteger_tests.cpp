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
#include <functional> // std::not_fn
#include <numeric> // std::iota
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

std::string ltrim(const std::string& str)
{
  std::string s(str);

  s.erase(
    s.begin(),
    std::find_if(
      s.begin(),
      s.end(),
      //std::not1(ptr_fun<int, int>(std::isspace))));
      //------------------------------------------------------------------------
      /// \url https://en.cppreference.com/w/cpp/utility/functional/not_fn
      /// \brief Creates a forwarding call wrapper that returns the negation of
      /// the callable object it holds.
      //------------------------------------------------------------------------
      std::not_fn([](int ch){ return std::isspace(ch); })));

  return s;
}

std::string rtrim(const std::string& str)
{
  std::string s(str);

  s.erase(
    std::find_if(
      s.rbegin(),
      s.rend(),
      //not1(ptr_fun<int, int>(std::isspace))).base(),
      std::not_fn([](int ch){ return std::isspace(ch); })).base(),
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

// cf. https://en.cppreference.com/w/cpp/utility/functional/not1
// https://en.cppreference.com/w/cpp/utility/functional/not1
// template <class F>
// not_fn(F&& f);
// where f - object from which Callable object held by wrapper is constructed.

// template <typename ArgumentType, typename ResultType>
// struct unary_function;
// unary_function removedin C++17
//struct LessThan7 : std::unary_function<int, bool>
struct LessThan7
{
  bool operator()(int i) const
  {
    return i < 7;
  }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdNotfnCreatesCallWrapperToNegationOfCallableObject)
{
  // std::iota fills range [first, last) with sequentially increasing values
  // starting with value, and repetitively evaluating ++value
  // template <class ForwardIt, class T>
  // void iota(ForwardIt first, ForwardIt last, T value);
  std::vector<int> v(10);
  std::iota(begin(v), end(v), 0);
  // std::count_f counts number of elements for which predicate p returns true
  BOOST_TEST(std::count_if(begin(v), end(v), std::not_fn(LessThan7())) == 3);
}

BOOST_AUTO_TEST_SUITE_END() // LonelyInteger_tests
BOOST_AUTO_TEST_SUITE_END() // BitManipulation
BOOST_AUTO_TEST_SUITE_END() // HackerRank
BOOST_AUTO_TEST_SUITE_END() // Algorithms