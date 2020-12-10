//------------------------------------------------------------------------------
/// \file FunctionObject_test.cpp
/// \ref Vandevoorde, Josuttis, Gregor. C++ Templates: The Complete Guide.
/// Addison-Wesley Professional; 2nd edition. 2017
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
//#include <functional> // std::functional
#include <vector>
#include <sstream> // std::ostringstream;

using std::ostringstream;
using std::vector;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(FunctionObject_tests)

//------------------------------------------------------------------------------
/// \ref Vandevoorde, Josuttis, Gregor (2017). pp. 517, 22.1
/// \brief Demonstrate Function template that enumerates integer values from 0
/// up to some value, providing each value to given function object f.
/// bridge/forupto3.hpp from Vandevoorde, Josuttis, Gregor (2017)
//------------------------------------------------------------------------------
template <typename F>
void for_up_to(const int n, F f)
{
  for (int i {0}; i < n; ++i)
  {
    f(i); // call passed function f for i
  }
}

void for_up_to_3(const int n, std::function<void(int)> f)
{
  for (int i {0}; i < n; ++i)
  {
    f(i); // call passed function f for i
  }
}

class TestOutputStringStream
{
  private:


};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FunctionTemplateForStaticPolymorphism)
{
  constexpr int n {5};

  vector<int> values;
  for_up_to(n, [&values](int i) { values.push_back(i); });

  for (int i {0}; i < n; ++i)
  {
    BOOST_TEST(values.at(i) == i);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdFunctionForTypeErasure)
{
  constexpr int n {5};

  vector<int> values;
  for_up_to_3(n, [&values](int i) { values.push_back(i); });

  for (int i {0}; i < n; ++i)
  {
    BOOST_TEST(values.at(i) == i);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdFunctionTypesMatchOnArgumentsNotOnReturnType)
{
  std::function<void(double)> f = [](double x) -> double { return x*x; };
  f(2);
  // Void has incomplete type.
  //auto result = f(2);
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // FunctionObject_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp
