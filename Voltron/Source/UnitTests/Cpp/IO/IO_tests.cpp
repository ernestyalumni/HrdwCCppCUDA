//------------------------------------------------------------------------------
/// \file IO_tests.cpp
//------------------------------------------------------------------------------
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iostream>
#include <string>

using Tools::TemporaryDirectory;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(IO)
BOOST_AUTO_TEST_SUITE(IO_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IfstreamExamples)
{
  std::cout << "\n IfstreamExamples \n";

  // cf. https://en.cppreference.com/w/cpp/io/basic_ifstream
  TemporaryDirectory temp_dir {"/temp"};  
  std::string filename {"Test.b"};

  // prepare a file to read
  double d {3.14};

  const std::string full_filepath {temp_dir.path() + "/" + filename};

  std::cout << full_filepath << '\n';

  std::ofstream(full_filepath, std::ios::binary).write(
    reinterpret_cast<char*>(&d), sizeof(d)) << 123 << "abc";

  // open file for reading
  std::ifstream istrm(full_filepath, std::ios::binary);

  if (!istrm.is_open())
  {
    std::cout << "failed to open " << full_filepath << '\n';
  }
  else
  {
    double d;
    istrm.read(reinterpret_cast<char*>(&d), sizeof(d)); // binary input
    int n;
    std::string s;
    if (istrm >> n >> s) // text input
    {
      std::cout << "read back from file: " << d << ' ' << n << ' ' << s << '\n';

      BOOST_TEST(d == 3.14);
      BOOST_TEST(n == 123);
      BOOST_TEST(s == "abc");
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // IO_tests 

BOOST_AUTO_TEST_SUITE_END() // IO
BOOST_AUTO_TEST_SUITE_END() // Cpp