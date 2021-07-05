#include "Cpp/CallByValueCallByReference.h"
#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Cpp::CallByReference::call_by_ref;
using Cpp::CallByValue::call_by_value;
using Tools::CaptureCoutFixture;
using std::string;

BOOST_AUTO_TEST_SUITE(Cpp)

BOOST_AUTO_TEST_SUITE(CallByValue)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MyFunctionChangesValueOfCopy,
  CaptureCoutFixture)
{
  int y {25};

  Cpp::CallByValue::my_function(y);

  const string expected {"Value of x from my_function: 50\n"};

  BOOST_TEST(local_oss_.str() == expected);

  BOOST_TEST(y == 25);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CallByValueChangesValueOfCopy,
  CaptureCoutFixture)
{
  int y {60};

  const int result {call_by_value(y)};

  BOOST_TEST(result == 50);

  const string expected {"Value of x from call_by_value: 50\n"};

  BOOST_TEST(local_oss_.str() == expected);

  BOOST_TEST(y == 60);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CallByValueAcceptsRValues,
  CaptureCoutFixture)
{
  const int result {call_by_value(100)};

  BOOST_TEST(result == 50);

  const string expected {"Value of x from call_by_value: 50\n"};

  BOOST_TEST(local_oss_.str() == expected);
}

BOOST_AUTO_TEST_SUITE_END() // CallByValue


//------------------------------------------------------------------------------
/// In call by reference, actual value that's passed as argument is changed
/// after performing some operation on it.
///
/// When call by reference is used, it creates a copy of the reference of that
/// variable into stack section in memory. When value is changed using reference
/// it changes value of actual variable.
///
/// Call by reference mainly used when we want to change value of passed
/// argument into invoker function.
///
/// 1 function can return only 1 value. When we need more than 1 value from a
/// function, we can pass them as output argument in this manner.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE(CallByReference)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MyFunctionChangesValueOfVariable,
  CaptureCoutFixture)
{
  int y {25};

  Cpp::CallByReference::my_function(y);

  const string expected {"Value of x from my_function: 50\n"};

  BOOST_TEST(local_oss_.str() == expected);

  BOOST_TEST(y == 50);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CallByRefChangesValueOfVariable,
  CaptureCoutFixture)
{
  int y {60};

  const int result {call_by_ref(y)};

  BOOST_TEST(result == 50);

  const string expected {"Value of x from call_by_ref: 50\n"};

  BOOST_TEST(local_oss_.str() == expected);

  BOOST_TEST(y == 50);
}

#ifdef FORCE_COMPILE_ERRORS

// \ref https://www.cs.fsu.edu/~myers/c++/notes/references.html
// When function expects strict reference types in parameter list, an L-value
// (i.e. variable or storage location) must be passed in.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CallByRefDoesNotAcceptRValues,
  CaptureCoutFixture)
{
  const int result {call_by_ref(60)};
}

#endif // FORCE_COMPILE_ERRORS

BOOST_AUTO_TEST_SUITE_END() // CallByReference

BOOST_AUTO_TEST_SUITE_END() // Cpp