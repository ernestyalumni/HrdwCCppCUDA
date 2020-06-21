//------------------------------------------------------------------------------
/// \file StdBitCast_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \details Function template that obtains value of type by reinterpreting
/// object representation of From.
/// \ref https://en.cppreference.com/w/cpp/numeric/bit_cast
//------------------------------------------------------------------------------
#include "Cpp/Numerics/BitCast.h"

#include <boost/test/unit_test.hpp>
#include <cstdint> // uint32_t
#include <iostream>

// cf. https://docs.microsoft.com/en-us/cpp/preprocessor/hash-ifdef-and-hash-ifndef-directives-c-cpp?view=vs-2019
// https://en.cppreference.com/w/cpp/feature_test
#ifdef __cpp_lib_bit_cast

#include <bit>

#define has_bit_cast 1

#else

#define has_bit_cast 0

#endif

// https://en.cppreference.com/w/cpp/feature_test
#ifdef __has_include
#if __has_include(<version>)
#include <version>
#endif
#endif

// https://en.cppreference.com/w/cpp/preprocessor/include
// Since C++17, use __has_include, preprocessor constant expression that
// evaluates to 1 if filename found, 0 if not. Program is ill-formed if argument
// would not be a valid argument to #include directive.
//
// More refs:
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
// https://en.cppreference.com/w/cpp/utility/feature_test

#if __has_include(<bit>)

#include <bit>
#define have_bit 1

#else
#define have_bit 0

//std::cout << "\n Does not have <span> \n;"

#endif 

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Numerics)
BOOST_AUTO_TEST_SUITE(BitCast_tests)

#if has_bit_cast == 1

using Cpp::Numerics::bit_cast;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdbitConstructsWithDouble)
{
  std::cout << "\n Has bit_cast: have_bit: " << have_bit << "\n";

  std::cout << "\n #ifdef __cpp_lib_bit true, has_bit_cast: " <<
    has_bit_cast << "\n";

  std::bit_cast<uint64_t>(19880124.0);
}

#else

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DoesNotHaveBit)
{
  std::cout << "\n Does not have bit; have_bit: " << have_bit << "\n";

  std::cout << "\n #ifdef __cpp_lib_bit false, has_bit_cast: " <<
    has_bit_cast << "\n";

  BOOST_TEST(true);  
}

#endif

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitCastFromFloatToInt32)
{
  {
    const int32_t value {bit_cast<int32_t>(1.0f)};
    BOOST_TEST(value == 1065353216);
  }

  {
    const uint32_t value {bit_cast<uint32_t>(1.0f)};
    BOOST_TEST(value == 1065353216);
  }
}

BOOST_AUTO_TEST_SUITE_END() // BitCast_tests
BOOST_AUTO_TEST_SUITE_END() // Numerics
BOOST_AUTO_TEST_SUITE_END() // Cpp
