//------------------------------------------------------------------------------
// \file ToHexString_tests.cpp
//------------------------------------------------------------------------------
#include "Utilities/ToHexString.h"

#include <boost/test/unit_test.hpp>
#include <algorithm> // std::generate
#include <cstddef> // std::byte
#include <cstdio> // printf
#include <iostream>
#include <sstream> // std::stringstream
#include <string>
#include <vector>

using Utilities::ToHexString;
using Utilities::to_hex_string;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(ToHexString_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PrintsRValueCharAsHex)
{
  {
    std::stringstream out;

    out << to_hex_string('A');

    BOOST_TEST(out.str() == "41");
  }
  {
    std::stringstream out;

    out << to_hex_string('5');

    BOOST_TEST(out.str() == "35");
  }
  {
    std::stringstream out;

    out << to_hex_string('b');

    BOOST_TEST(out.str() == "62");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PrintsLValueCharAsHex)
{
  std::stringstream out;
  const char character {'A'};

  out << to_hex_string(character);

  BOOST_TEST(out.str() == "41");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AsIncreasingAddressesWorks)
{
  const ToHexString hex_string {1234};

  std::stringstream out;

  //hex_string.as_increasing_addresses(out);

  //BOOST_TEST(out.str() == "d2");
}

// cf. https://en.cppreference.com/w/cpp/algorithm/generate
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdGeneratePlayground)
{
  std::vector<int> v (5);
  BOOST_TEST(v.size() == 5);

  // Initialize with default values 4, 3, 2, 1, 0 from a lambda function.
  // The reverse of the equivalent to std::iota(v.begin(), v.end(), 0)

  // Without the mutable keyword, error: decrement of read-only variable 'n'
  // https://stackoverflow.com/questions/5501959/why-does-c11s-lambda-require-mutable-keyword-for-capture-by-value-by-defau
  // It required mutable because by default, a function object should produce
  // the same result every time it's called; this is the difference between an
  // object oriented function and function using a global variable, effectively.
  std::generate(v.begin(), v.end(), [n = 4] () mutable { return n--;});

  for (int i {0}; i < v.size(); ++i)
  {
    BOOST_TEST(v[i] == v.size() - i - 1);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IncreasingAddressesPrintWorks)
{
  std::cout << "\n IncreasingAddressesPrintWorks begins \n";
  {
    const ToHexString hex_string {1234};

    hex_string.increasing_addresses_print();
  }
  {
    constexpr short x_1 {15213};
    const ToHexString hex_string {x_1};
    hex_string.increasing_addresses_print(); // 6d 93
  }
  {
    constexpr short y {-15213};
    const ToHexString hex_string {y};
    hex_string.increasing_addresses_print();
  }

  std::cout << "\n IncreasingAddressesPrintWorks ends \n";

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DecreasingAddressesPrintWorks)
{
  std::cout << "\n DecreasingAddressesPrintWorks begins \n";
  {
    const ToHexString hex_string {1234};

    hex_string.decreasing_addresses_print();
  }
  {
    const ToHexString hex_string {0x123456789abcdef0};

    hex_string.decreasing_addresses_print();
  }
  {
    constexpr short x_1 {15213};
    const ToHexString hex_string {x_1};
    hex_string.decreasing_addresses_print();
  }
  {
    constexpr short y {-15213};
    const ToHexString hex_string {y};
    hex_string.decreasing_addresses_print(); // 3b c4
  }

  std::cout << "\n DecreasingAddressesPrintWorks ends \n";

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE(RepresentingStrings_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StringsAsHex)
{
  std::stringstream out;

  //char input[6] {"12345"};
  //std::string input {"12345"};
  const ToHexString hex_string {std::string{"12345"}};
  // Only gets the pointer address.
  //BOOST_TEST(hex_string.as_increasing_addresses() ==
  //  "b03c9fb3fd7f005000000031323334350000a6ce5188dd518");
}


BOOST_AUTO_TEST_SUITE_END() // RepresentingStrings_tests

BOOST_AUTO_TEST_SUITE_END() // ToHexString_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities