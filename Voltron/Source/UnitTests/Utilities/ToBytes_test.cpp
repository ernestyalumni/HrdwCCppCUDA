//------------------------------------------------------------------------------
// \file ToBytes_test.cpp
//------------------------------------------------------------------------------
#include "Utilities/ToBytes.h"

#include <boost/test/unit_test.hpp>
#include <cstddef> // std::byte
#include <cstdio> // printf
#include <iostream>

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

  //to_bytes_x_0.increasing_addresses_print(); // WORKS
  //to_bytes_x_0.decreasing_addresses_print(); // WORKS

  ToBytes to_bytes_x_1 {x_1};

  to_bytes_x_1.increasing_addresses_print(); // WORKS 93
  to_bytes_x_1.decreasing_addresses_print(); // WORKS c4

  const ToBytes to_bytes_y {y};

  to_bytes_y.increasing_addresses_print(); // WORKS c4
  to_bytes_y.decreasing_addresses_print(); // WORKS 93
}

BOOST_AUTO_TEST_SUITE_END() // ToBytes_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities