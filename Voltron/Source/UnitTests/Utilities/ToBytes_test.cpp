//------------------------------------------------------------------------------
// \file ToBytes_test.cpp
//------------------------------------------------------------------------------
#include "Utilities/ToBytes.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::byte
#include <cstdio> // printf
#include <iostream>
#include <limits>
#include <string>

using Utilities::ToBytes;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(ToBytes_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReinterpretCastAndHexadecimalPrintExamples)
{
  std::cout << "\n ReinterpretCastAndHexadecimalPrintExamples begins \n";


  constexpr unsigned short x_0 {15213};
  constexpr short x_1 {15213};
  constexpr short y {-15213};

  auto x = reinterpret_cast<const std::byte*>(&x_0);
  auto xa = reinterpret_cast<const unsigned char*>(&x_0);

  //std::cout << std::to_integer(x[0]) << '\n'; // didn't work
  //std::cout << std::to_integer(*x) << '\n'; // didn't work
  // std::cout << x << '\n'; // worked

  //std::cout << xa[0] << '\n'; // m
  //std::cout << xa[1] << '\n'; // ;
  //std::cout << xa[2] << '\n'; 
  //std::cout << xa[3] << '\n';
  //printf("%x \n", xa); // 178892be
  //printf("%01x \n", xa[0]);
  //printf("%01x \n", xa[1]);
  //printf("%01x \n", xa[2]);
  //printf("%001x \n", xa[3]);
  //printf("%001x \n", xa[4]);

  BOOST_TEST_REQUIRE(sizeof(unsigned short) == 2);
  BOOST_TEST_REQUIRE(sizeof(short) == 2);

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ToBytesWorks)
{
  constexpr unsigned short x_0 {15213};
  constexpr short x_1 {15213};
  constexpr short y {-15213};

  const ToBytes to_bytes_x_0 {x_0};

  std::cout << "\n Print 15213 with increasing addresses \n";
  to_bytes_x_0.increasing_addresses_print(); // WORKS
  //to_bytes_x_0.decreasing_addresses_print(); // WORKS
  std::cout << "\n END OF Print 15213 with increasing addresses \n";

  ToBytes to_bytes_x_1 {x_1};

  to_bytes_x_1.increasing_addresses_print(); // WORKS 93
  to_bytes_x_1.decreasing_addresses_print(); // WORKS c4

  const ToBytes to_bytes_y {y};

  to_bytes_y.increasing_addresses_print(); // WORKS c4
  to_bytes_y.decreasing_addresses_print(); // WORKS 93
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IncreasingAddressesHexWorks)
{
  constexpr unsigned short x_0 {15213};
  constexpr short x_1 {15213};
  constexpr short y {-15213};

  const ToBytes to_bytes_x_0 {x_0};

  ToBytes to_bytes_x_1 {x_1};

  const ToBytes to_bytes_y {y};

  const std::string x_0_str {to_bytes_x_0.increasing_addresses_hex()};
  BOOST_TEST(x_0_str == "6d3b");

  const std::string x_1_str {to_bytes_x_1.increasing_addresses_hex()};
  BOOST_TEST(x_1_str == "6d3b");

  const std::string y_str {to_bytes_y.increasing_addresses_hex()};
  BOOST_TEST(y_str == "93c4");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IncreasingAddressesHexWorksByInduction)
{
  {
    BOOST_TEST(sizeof(unsigned int) == 4);
    const unsigned int x {0};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "0000");
  }
  {
    const unsigned int x {1};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "1000");
  }
  {
    const unsigned int x {2};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "2000");
  }
  {
    const unsigned int x {15};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "f000");
  }
  {
    const unsigned int x {16};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "10000");
  }
  {
    const unsigned int x {17};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "11000");
  }
  {
    const unsigned int x {std::numeric_limits<unsigned int>::max() - 1};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "feffffff");
  }
  {
    const unsigned int x {std::numeric_limits<unsigned int>::max()};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "ffffffff");
  }
  {
    const unsigned int x {std::numeric_limits<unsigned int>::min()};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.increasing_addresses_hex() == "0000");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DecreasingAddressesHexWorksByInduction)
{
  {
    BOOST_TEST(sizeof(unsigned int) == 4);
    const unsigned int x {0};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "0000");
  }
  {
    const unsigned int x {1};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "0001");
  }
  {
    const unsigned int x {2};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "0002");
  }
  {
    const unsigned int x {15};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "000f");
  }
  {
    const unsigned int x {16};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "00010");
  }
  {
    const unsigned int x {17};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "00011");
  }
  {
    const unsigned int x {std::numeric_limits<unsigned int>::max() - 1};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "fffffffe");
  }
  {
    const unsigned int x {std::numeric_limits<unsigned int>::max()};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "ffffffff");
  }
  {
    const unsigned int x {std::numeric_limits<unsigned int>::min()};
    const ToBytes to_bytes_x {x};
    BOOST_TEST(to_bytes_x.decreasing_addresses_hex() == "0000");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DecreasingAddressesHexWorks)
{
  constexpr unsigned short x_0 {15213};
  constexpr short x_1 {15213};
  constexpr short y {-15213};

  const ToBytes to_bytes_x_0 {x_0};

  ToBytes to_bytes_x_1 {x_1};

  const ToBytes to_bytes_y {y};

  const std::string x_0_str {to_bytes_x_0.decreasing_addresses_hex()};
  BOOST_TEST(x_0_str == "3b6d");

  const std::string x_1_str {to_bytes_x_1.decreasing_addresses_hex()};
  BOOST_TEST(x_1_str == "3b6d");

  const std::string y_str {to_bytes_y.decreasing_addresses_hex()};
  BOOST_TEST(y_str == "c493");
}

BOOST_AUTO_TEST_SUITE(RepresentingStrings_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StringToByteGetsPointer)
{
  const std::string input {"12345"};
  const ToBytes input_in_bytes {input};

  //BOOST_TEST(input_in_bytes.decreasing_addresses_hex() ==
  //  "0000040ad14000353433323100000005007ffdc68e6e10 ");

  const ToBytes input_as_char_in_bytes {input.data()};

  //BOOST_TEST(input_as_char_in_bytes.decreasing_addresses_hex() ==
  //  "007ffd70befdb0 ");

  //const ToBytes r_input {"12345"};

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // RepresentingStrings_tests

BOOST_AUTO_TEST_SUITE_END() // ToBytes_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities