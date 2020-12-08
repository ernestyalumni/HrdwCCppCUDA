//------------------------------------------------------------------------------
/// \file Invoke_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/utility/functional/invoke
/// https://en.cppreference.com/w/cpp/types/is_invocable
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <type_traits>

using std::is_invocable;
using std::is_invocable_r;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(FunctionObjects)
BOOST_AUTO_TEST_SUITE(Invoke_tests)

BOOST_AUTO_TEST_SUITE_END() // Invoke_tests

BOOST_AUTO_TEST_SUITE(Invocable_tests)

auto func2(char) -> int (*)()
{
  return nullptr;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsInvocableDeterminesIfFunctionTypeCanBeInvoked)
{
  BOOST_TEST(is_invocable<int()>::value);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
  IsInvocableDeterminesIfFunctionTypeCanBeInvokedToYieldReturnType)
{
  BOOST_TEST((is_invocable_r<int, int()>::value));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
  IsInvocableDeterminesIfFunctionTypeCanBeInvokedWithArgumentTypes)
{
  BOOST_TEST((is_invocable_r<void, void(int), int>::value));
  BOOST_TEST((is_invocable_r<int(*)(), decltype(func2), char>::value));
}


BOOST_AUTO_TEST_SUITE_END() // Invocable_tests

BOOST_AUTO_TEST_SUITE_END() // FunctionObjects
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp