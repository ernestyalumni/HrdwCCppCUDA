#include "DataStructures/HashTables/HashFunctions.h"

#include <boost/test/unit_test.hpp>

using DataStructures::HashTables::HashFunctions::Details::string_to_radix_128;
using DataStructures::HashTables::HashFunctions::DivisionMethod;
using DataStructures::HashTables::HashFunctions::UnitIntervalToHash;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(HashTables)
BOOST_AUTO_TEST_SUITE(HashFunctions)

BOOST_AUTO_TEST_SUITE(UnitIntervalToHash_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithNumberOfSlots)
{
  UnitIntervalToHash hf {10};

  BOOST_TEST(hf.get_m() == 10);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CalculatesSlotsForKeysFrom0ToBefore1)
{
  UnitIntervalToHash hf {10};

  BOOST_TEST(hf(0.01) == 0);
  BOOST_TEST(hf(0.05) == 0);
  BOOST_TEST(hf(0.09) == 0);
  BOOST_TEST(hf(0.1) == 1);
  BOOST_TEST(hf(0.11) == 1);
  BOOST_TEST(hf(0.15) == 1);
  BOOST_TEST(hf(0.19) == 1);
  BOOST_TEST(hf(0.2) == 2);
  BOOST_TEST(hf(0.21) == 2);
  BOOST_TEST(hf(0.25) == 2);
  BOOST_TEST(hf(0.29) == 2);
  BOOST_TEST(hf(0.91) == 9);
  BOOST_TEST(hf(0.95) == 9);
  BOOST_TEST(hf(0.99) == 9);
}

BOOST_AUTO_TEST_SUITE_END() // UnitIntervalToHash_tests

BOOST_AUTO_TEST_SUITE(DivisionMethod_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithNumberOfSlots)
{
  DivisionMethod hf {11};

  BOOST_TEST(hf.get_m() == 11);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Computes)
{
  DivisionMethod hf {5};

  BOOST_TEST(hf(0) == 0);
  BOOST_TEST(hf(1) == 1);
  BOOST_TEST(hf(2) == 2);
  BOOST_TEST(hf(3) == 3);
  BOOST_TEST(hf(4) == 4);
  BOOST_TEST(hf(5) == 0);
  BOOST_TEST(hf(6) == 1);
  BOOST_TEST(hf(7) == 2);
}

BOOST_AUTO_TEST_SUITE_END() // DivisionMethod_tests

BOOST_AUTO_TEST_SUITE(StringToRadix128_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Computes)
{
  BOOST_TEST(string_to_radix_128("pt") == 14452);
}

BOOST_AUTO_TEST_SUITE_END() // StringToRadix128_tests

BOOST_AUTO_TEST_SUITE_END() // HashFunctions
BOOST_AUTO_TEST_SUITE_END() // HashTables
BOOST_AUTO_TEST_SUITE_END() // DataStructures