//------------------------------------------------------------------------------
// \file NarrowCast_test.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <stdexcept>

#include "Utilities/NarrowCast.h"

using Utilities::narrow_cast;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(NarrowCast_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// cf. Stroustrup. C++ Programming Language, 4th. Ed. pp. 299 Sec. 11.5
BOOST_AUTO_TEST_CASE(NarrowCastWorks)
{
  const auto c1 = narrow_cast<char>(64);

  // Will throw if chars are unsigned.
  //BOOST_CHECK_THROW(narrow_cast<char>(-64), std::runtime_error); 
  const auto c2 = narrow_cast<char>(-64); // Will throw if chars are unsigned

  //const auto c3 = narrow_cast<char>(264); // will throw if chars are 8-bit and signed
  BOOST_CHECK_THROW(narrow_cast<char>(264), std::runtime_error);

  const auto d1 = narrow_cast<double>(1/3.0F); // OK
  
  //const auto f1 = narrow_cast<float>(1/3.0); // will probably throw
  BOOST_CHECK_THROW(narrow_cast<float>(1/3.0), std::runtime_error);

  const auto c4 = narrow_cast<char>(42); // may throw
  const auto f2 = narrow_cast<float>(42.0); // may throw

  //const auto p1 = narrow_cast<char*>(42); // compile-time error, invalid static_cast
  //const auto i1 = narrow_cast<int>("chararray"); // compile-time error, invalid static_Cast

  const auto d2 = narrow_cast<double>(42); // may throw (but probably will not)
  const auto i2 = narrow_cast<int>(42.0); // may throw

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // NarrowCast_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
