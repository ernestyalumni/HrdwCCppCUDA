//------------------------------------------------------------------------------
/// \file MainCommandLine_tests.cpp
/// \ref Bjarne Stroustrup. The C++ Programming Language, 4th Edition.
/// Addison-Wesley Professional. May 19, 2013. ISBN-13: 978-0321563842
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <sstream>
#include <vector>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Main_Command_Line_Arguments_tests)

// cf. http://www.cplusplus.com/reference/sstream/istringstream/istringstream/
// https://en.cppreference.com/w/cpp/io/basic_istringstream
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IStringStreamExtractsFormattedData)
{
  std::string string_values {"125 320 512 750 333"};
  std::istringstream iss {string_values};

  std::vector<int> expected_values {250, 640, 1024, 1500, 666};
  for (int n {0}; n < 5; ++n)
  {
    int val;
    iss >> val;
    BOOST_TEST(val * 2 == expected_values[n]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DeclareAndInitializeArgvAsArrayOfCharPointers)
{
  const int argc {5};

  const char* lions[] = {"Black", "Red", "Green", "Blue", "Yellow", 0};

  const char* argv[argc + 1];

  for (int i {0}; i <= argc; ++i)
  {
    argv[i] = lions[i];
  }

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // Main_Command_Line_Arguments_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp