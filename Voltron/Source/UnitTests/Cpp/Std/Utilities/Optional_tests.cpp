//------------------------------------------------------------------------------
/// \file Optional_tests.cpp
/// \ref https://en.cppreference.com/w/cpp/utility/optional
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <memory>
#include <optional> // std::nullopt
#include <utility> // std::move

using std::nullopt;
using std::optional;

BOOST_AUTO_TEST_SUITE(Cpp) // The C++ Language
BOOST_AUTO_TEST_SUITE(Std)
BOOST_AUTO_TEST_SUITE(Optional_tests)

// \ref https://en.cppreference.com/w/cpp/utility/optional/optional
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsToObjectThatDoesNotContainAValue)
{
  optional<int> a;

  BOOST_TEST(!a.has_value());
  BOOST_TEST(!static_cast<bool>(a));
  BOOST_TEST((a == nullopt));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithValue)
{
  optional<int> a {42};

  BOOST_TEST(a.has_value());
  BOOST_TEST(static_cast<bool>(a));
  BOOST_TEST((a != nullopt));
  BOOST_TEST(a.value() == 42);
  BOOST_TEST(*a == 42);

}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyConstructs)
{
  const optional<int> a {42};

  const optional<int> b {a};

  // Original copy source doesn't get mutated.
  BOOST_TEST(a.has_value());
  BOOST_TEST(static_cast<bool>(a));
  BOOST_TEST((a != nullopt));
  BOOST_TEST(a.value() == 42);
  BOOST_TEST(*a == 42);

  BOOST_TEST(b.has_value());
  BOOST_TEST(static_cast<bool>(b));
  BOOST_TEST((b != nullopt));
  BOOST_TEST(b.value() == 42);
  BOOST_TEST(*b == 42);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateHasValue)
{
  std::optional<int> optional_1 {42};

  BOOST_TEST(optional_1.has_value());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NulloptAssignmentRemovesContents)
{
  optional<int> a {42};

  BOOST_TEST(a.value() == 42);

  a = nullopt;

  BOOST_TEST(!a.has_value());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ResetRemovesContents)
{
  optional<int> a {42};

  BOOST_TEST(a.value() == 42);

  a.reset();

  BOOST_TEST(!a.has_value());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DereferenceOperatorCanChangeValue)
{
  optional<int> a {42};

  BOOST_TEST(a.value() == 42);

  *a = 43;

  BOOST_TEST(a.value() == 43);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IncrementCanChangeValueWithDereferencing)
{
  optional<int> a {42};

  BOOST_TEST(a.value() == 42);

  *a += 1;

  BOOST_TEST(a.value() == 43);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DecrementCanChangeValueWithDereferencing)
{
  optional<int> a {42};

  BOOST_TEST(a.value() == 42);

  *a -= 2;

  BOOST_TEST(a.value() == 40);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ValueCanChangeValue)
{
  optional<int> a {42};

  BOOST_TEST(a.value() == 42);

  a.value() = 43;

  BOOST_TEST(a.value() == 43);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(EmplaceCanChangeValue)
{
  optional<int> a {42};

  BOOST_TEST(a.value() == 42);

  a.emplace(43);

  BOOST_TEST(a.value() == 43);
}


BOOST_AUTO_TEST_SUITE_END() // Optional_tests
BOOST_AUTO_TEST_SUITE_END() // Std
BOOST_AUTO_TEST_SUITE_END() // Cpp