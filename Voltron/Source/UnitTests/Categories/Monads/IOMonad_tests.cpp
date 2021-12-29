//------------------------------------------------------------------------------
/// \file IOMonad_tests.cpp
/// \author Ernest Yeung
//------------------------------------------------------------------------------
#include "Categories/Monads/IOMonad.h"

#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>

using Categories::Monads::IOMonad::IO;
using Tools::CaptureCoutFixture;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(IOMonad_tests)

// Test morphisms.

IO print_hello()
{
  std::cout << "IO Hello ";
  return IO();
}

IO print_world()
{
  std::cout << "World IO\n";
  return IO();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(IOMonadAsClassDoesOperations, CaptureCoutFixture)
{
  IO my_io = IO();

  my_io = my_io.do_operation(print_hello).do_operation(print_world);
  my_io.do_operation(print_hello).do_operation(print_world);

  const std::string expected {"IO Hello World IO\nIO Hello World IO\n"};

  BOOST_TEST(local_oss_.str() == expected);
}

BOOST_AUTO_TEST_SUITE_END() // IOMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories