#include "Utilities/Memory/HashTable.h"

#include <cstdint>
#include <boost/test/unit_test.hpp>

using Utilities::Memory::to_int;

template <typename T>
T wrap_around(const T x)
{
  return x + (1 << (sizeof(T) - 1));
}

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Memory)

BOOST_AUTO_TEST_SUITE(HashTable_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ToIntIntermediarySteps)
{
  {
    int16_t result {static_cast<int16_t>(-65537)};
    result = result + (1 << (sizeof(int16_t) - 1));

    BOOST_TEST(result == 1);

    BOOST_TEST(wrap_around<int16_t>(static_cast<int16_t>(-65536)) == 2);
    BOOST_TEST(wrap_around<int16_t>(static_cast<int16_t>(-65535)) == 3);
  }
  {
    int x {-69};
    int* int_ptr {&x};

    BOOST_TEST(to_int(int_ptr) == -8);

    x = -42;
    BOOST_TEST(to_int(int_ptr) == -5);

    x = 42;
    BOOST_TEST(to_int(int_ptr) == 5);

    x = 69;
    BOOST_TEST(to_int(int_ptr) == 8);

    x = 9;
    BOOST_TEST(to_int(int_ptr) == 1);
    x = 8;
    BOOST_TEST(to_int(int_ptr) == 1);
    x = 7;
    BOOST_TEST(to_int(int_ptr) == 0);
    x = 6;
    BOOST_TEST(to_int(int_ptr) == 0);
  }
}

BOOST_AUTO_TEST_SUITE_END() // HashTable_tests
BOOST_AUTO_TEST_SUITE_END() // Memory
BOOST_AUTO_TEST_SUITE_END() // Utilities
