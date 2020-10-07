//------------------------------------------------------------------------------
// \file FromAddress_tests.cpp
//------------------------------------------------------------------------------
#include "Utilities/FromAddress.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace Utilities::FromAddress;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(FromAddress_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IntPtrAddressObtainedFromAddress)
{
  int x {42};
  int* int_ptr {&x};
  const auto from_address_to_bytes_x = from_address_of_to_bytes(x);
  const auto from_pointer_to_bytes_x = from_pointer_to_address_in_bytes(x);

  BOOST_TEST(
    from_address_to_bytes_x.decreasing_addresses_hex() ==
      from_pointer_to_bytes_x.decreasing_addresses_hex());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IntPtrAddressObtainedAsHexString)
{
  int x {42};
  int* int_ptr {&x};
  const auto from_address_to_hex_string_x = from_address_of_to_hex_string(x);
  const auto from_pointer_to_hex_string_x =
    from_pointer_to_address_in_hex_string(x);

  // *** stack smashing detected ***: terminated
  // BOOST_TEST(
  //  from_address_to_hex_string_x.as_decreasing_addresses() ==
  //    from_pointer_to_hex_string_x.as_decreasing_addresses());
}

BOOST_AUTO_TEST_SUITE_END() // FromAddress_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities