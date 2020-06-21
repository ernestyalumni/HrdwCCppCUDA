//------------------------------------------------------------------------------
/// \file ToAddress_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/memory/to_address
/// https://en.cppreference.com/w/cpp/utility/feature_test
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <memory>
#include <iostream>

#ifdef __cpp_lib_to_address

#define  has_to_address 1

#else

#define has_to_address 0

#endif

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Memory)
BOOST_AUTO_TEST_SUITE(ToAddress_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasToAddress)
{
  std::cout << "\n std::to_address?" << has_to_address << "\n";

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // ToAddress_tests
BOOST_AUTO_TEST_SUITE_END() // Memory
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp