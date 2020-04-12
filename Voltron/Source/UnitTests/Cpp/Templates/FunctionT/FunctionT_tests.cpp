//------------------------------------------------------------------------------
/// \file FunctionT_tests.cpp
/// \ref Vandevoorde, Josuttis, Gregor. C++ Templates: The Complete Guide. 2nd
/// Ed. Addison-Wesley Professional. 2017.
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <string>

using std::cos;
using std::sin;
using std::sqrt;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Templates)
BOOST_AUTO_TEST_SUITE(FunctionTemplates)
BOOST_AUTO_TEST_SUITE(FunctionT_tests)

template <typename X, typename FX, FX ObjectMap(X)>
FX object_map(const X& x)
{
  return ObjectMap(x);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MapAsTemplateParameter)
{
  const auto result = object_map<double, double, &sin>(M_PI_4);
  BOOST_TEST(result == 1.0 / sqrt(2.0));
}

BOOST_AUTO_TEST_SUITE_END() // FunctionT_tests

BOOST_AUTO_TEST_SUITE_END() // FunctionTemplates
BOOST_AUTO_TEST_SUITE_END() // Templates
BOOST_AUTO_TEST_SUITE_END() // Cpp