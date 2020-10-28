//------------------------------------------------------------------------------
/// \file CStyleFileIO_tests.cpp
/// \date 20201028 03:44
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <cstdio>

using std::FILE;
using std::fopen;
using std::remove;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(FileIO)
BOOST_AUTO_TEST_SUITE(CStyleFileIO_tests)


// https://en.cppreference.com/w/cpp/io/c/fopen

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FopenReturnsNullPtrIfFileOpeningFailed)
{
  // "r", read, open a file for reading, if file doesn't exist, failure to open.

  FILE* fp {fopen("test_cstylefileio.txt", "r")};

  if (fp == nullptr)
  {
    // File opening failed.
    BOOST_TEST(true);
  }

}

BOOST_AUTO_TEST_SUITE_END() // CStyleFileIO_tests
BOOST_AUTO_TEST_SUITE_END() // FileIO
BOOST_AUTO_TEST_SUITE_END() // Cpp