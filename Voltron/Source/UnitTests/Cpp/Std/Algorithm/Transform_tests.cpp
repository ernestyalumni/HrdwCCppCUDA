//------------------------------------------------------------------------------
// \file Transform_tests.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>

#include <algorithm> // std::transform
#include <array>
#include <cctype> // std::toupper
#include <iostream>
#include <list>
#include <optional>
#include <string>

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monoidal)
BOOST_AUTO_TEST_SUITE(Monoid_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdTransformExamples)
{
  std::string s {"hello"};

  // \ref https://en.cppreference.com/w/cpp/algorithm/transform
  // <algorithm>
  // 1.
  // template <class InputIt, class OutputIt, class UnaryOperation>
  // OutputIt transform(InputIt first1, InputIt last1, OutputIt d_first,
  //  UnaryOperation unary_op);
  //
  // template <class InputIt, class OutputIt, class UnaryOperation>
  // constexpr OutputIt transform(InputIt first1, InputIt last1,
  //  OutputIt d_first, UnaryOperation unary_op);
  //
  // 3.
  // template <class InputIt1, class InputIt2, class OutputIt,
  //  class BinaryOperation>
  // OutputIt transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
  //  OutputIt d_first, BinaryOperation binary_op);
  //
  // 1. unary operation unary_op is applied to range defined by [first1, last1)
  // 3. Binary operation binary_op applied to pairs of elements from 2 ranges:
  //  1 defined by [first1, last2), other beginning at first2

  std::transform(s.begin(),
    s.end(),
    s.begin(),
    [](unsigned char c) -> unsigned char {return std::toupper(c); });
  // Converts given character to uppercase according to character conversion
  // rules defined by currently installed C locale.


  std::vector<std::size_t> ordinals;
  std::transform(s.begin(),
    s.end(),
    std::back_inserter(ordinals),
    [](unsigned char c) -> std::size_t { return c;});

  //std::cout << s << ':';

  //BOOST_TEST(s == "HELLO");

  for (auto ord : ordinals)
  {
    std::cout << ' ' << ord; // 72 69 76 76 79
  }

  BOOST_TEST(true);
}

// cf. https://nalaginrut.com/archives/2019/10/31/8%20essential%20patterns%20you%20should%20know%20about%20functional%20programming%20in%20c%2B%2B14

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdTransformAsMapAsHighOrderFunction)
{
  std::list<int> c {1, 2, 3};
  std::transform(
    c.begin(),
    c.end(),
    c.begin(),
    [](int i)
      {
        return '0' + i;
      });

  for (char i : c)
  {
    std::cout << i << std::endl;
  }
  std::array<char, 3> c_expected {'1', '2', '3'};
  int i {0};
  for (const auto& ele : c)
  {
    BOOST_TEST(ele == c_expected[i]);
    i++;
  }
}

BOOST_AUTO_TEST_SUITE_END() // Monoid_tests
BOOST_AUTO_TEST_SUITE_END() // Monoidal
BOOST_AUTO_TEST_SUITE_END() // Categories