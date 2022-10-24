//------------------------------------------------------------------------------
// \file Monad_tests.cpp
//------------------------------------------------------------------------------
#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <optional>

using Tools::CaptureCoutFixture;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monoidal)
BOOST_AUTO_TEST_SUITE(Monad_tests)

// optional can be used as the return type of a factory that may fail
std::optional<std::string> create(bool b)
{
  if (b)
  {
    return "Godzilla";
  }
  return {};
}

// std::nullopt can be used to create any (empty) std::optional
auto create2(bool b)
{
  return b ? std::optional<std::string>{"Godzilla"} : std::nullopt;
}

// std::reference_wrapper may be used to return a reference
auto create_ref(bool b)
{
  static std::string value = "Godzilla";
  return b ? std::optional<std::reference_wrapper<std::string>>(value) :
    std::nullopt;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// \ref https://en.cppreference.com/w/cpp/utility/optional
BOOST_AUTO_TEST_CASE(StdOptionalExamples)
{
  CaptureCoutFixture capture_cout {};

  std::cout << "create(false) returned "
    << create(false).value_or("empty") << '\n';

  BOOST_TEST(capture_cout.local_oss_.str() == "create(false) returned empty\n");
}

BOOST_AUTO_TEST_SUITE_END() // Monad_tests
BOOST_AUTO_TEST_SUITE_END() // Monoidal
BOOST_AUTO_TEST_SUITE_END() // Categories