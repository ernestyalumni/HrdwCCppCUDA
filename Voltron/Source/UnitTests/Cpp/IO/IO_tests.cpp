//------------------------------------------------------------------------------
/// \file IO_tests.cpp
//------------------------------------------------------------------------------
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <cctype> // std::isprint
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

using Tools::TemporaryDirectory;
using std::snprintf;

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

BOOST_AUTO_TEST_SUITE(CStyleIO_tests)

BOOST_AUTO_TEST_SUITE(CStyleStringReview)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CStringSizeIncludesNullTerminationCharacter)
{
  const char* hello_world = "Hello World!";
  BOOST_TEST(hello_world[12] == '\0');
}

BOOST_AUTO_TEST_SUITE_END() // CStyleStringReview 

//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/io/c/fprintf
/// https://youtu.be/rUA3IkQNe5I
/// \details
/// Instead of printing to terminal, output, print to another char buffer.
///
/// int snprintf(char* buffer, std::size_t buf_size, const char* format, ...);
///
/// 4) Writes results to char string buffer. At most buf_size - 1 characters
/// written. Resulting char string terminated with null char, unless buf_size is
/// 0. If buf_size is 0, nothing is written, and buffer may be null ptr,
/// however, return value (number of bytes that would've been written not
/// including null terminator) still calculated and returned.
///
/// buffer - ptr to char string to write to,
/// buf_size - buf_size - 1 
/// format - ptr to null-terminated multibyte string specifying how to interpret
/// the data.
/// Format string consists of ordinary multibyte chars (except %), which are
/// copied unchanged into output stream, and conversion specifications.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SnPrintfWritesResultsToCharacterStringBuffer)
{
}

BOOST_AUTO_TEST_SUITE_END() // CStyleIO_tests 

BOOST_AUTO_TEST_SUITE_END() // IO
BOOST_AUTO_TEST_SUITE_END() // Cpp