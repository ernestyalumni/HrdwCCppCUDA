//------------------------------------------------------------------------------
// \file RuleOf5_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/RuleOf5.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Cpp::RuleOf5::RuleOf5Object;
using std::string;

BOOST_AUTO_TEST_SUITE(Hierarchy)
BOOST_AUTO_TEST_SUITE(RuleOf5_tests)
BOOST_AUTO_TEST_SUITE(RuleOf5Object_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstruction)
{
  RuleOf5Object a;
  BOOST_TEST(a.s_data() == "");
  BOOST_TEST(a.int_data() == 0);
  BOOST_TEST(a.copy_constructor_counter() == 0);
  BOOST_TEST(a.copy_assign_counter() == 0);
  BOOST_TEST(a.move_constructor_counter() == 0);
  BOOST_TEST(a.move_assign_counter() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopyConstructor)
{
  const string input_s {"Sixty-Nine"};
  const int input_int {42};

  const RuleOf5Object a {input_s, input_int};

  const RuleOf5Object b {a};

  BOOST_TEST(a.s_data() == input_s);
  BOOST_TEST(a.int_data() == input_int);
  BOOST_TEST(a.copy_constructor_counter() == 0);
  BOOST_TEST(a.copy_assign_counter() == 0);
  BOOST_TEST(a.move_constructor_counter() == 0);
  BOOST_TEST(a.move_assign_counter() == 0);

  BOOST_TEST(b.s_data() == input_s);
  BOOST_TEST(b.int_data() == input_int);
  BOOST_TEST(b.copy_constructor_counter() == 1);
  BOOST_TEST(b.copy_assign_counter() == 0);
  BOOST_TEST(b.move_constructor_counter() == 0);
  BOOST_TEST(b.move_assign_counter() == 0);
}

BOOST_AUTO_TEST_SUITE_END() // RuleOf5Object_tests
BOOST_AUTO_TEST_SUITE_END() // RuleOf5_tests
BOOST_AUTO_TEST_SUITE_END() // Hierarchy
