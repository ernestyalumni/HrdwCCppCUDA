//------------------------------------------------------------------------------
/// \file CStyleFileIO_tests.cpp
/// \date 20201028 03:44
//------------------------------------------------------------------------------

#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <cstdio>
#include <string>

using Tools::TemporaryDirectory;
using std::FILE;
using std::fopen;
using std::remove;
using std::string;

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FopenOpensFileForWriting)
{
  TemporaryDirectory temp_dir {"Temp"};

  string test_full_path {temp_dir.path() + "test.txt"};

  // "w", write, Create file for writing, if already exists, destroy contents,
  // if file doesn't exist, create new.
  FILE* fp {fopen(test_full_path.data(), "w")};

  if (fp == nullptr)
  {
    // File opening failed.
    BOOST_TEST(true);
  }

  int c; // note: int, not car, required to handle EOF
  //while ((c = ))

  // closes a file.
  fclose(fp);
}

BOOST_AUTO_TEST_SUITE_END() // CStyleFileIO_tests
BOOST_AUTO_TEST_SUITE_END() // FileIO
BOOST_AUTO_TEST_SUITE_END() // Cpp