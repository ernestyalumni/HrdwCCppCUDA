//------------------------------------------------------------------------------
/// \file GetCurrentDirectory_tests.cpp
/// \date 20201028 03:44
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(FileIO)
BOOST_AUTO_TEST_SUITE(GetCurrentDirectory_tests)
BOOST_AUTO_TEST_SUITE(StdFilesystemCurrent_Path_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdFilesystemCurrentPathGetsCurrentPath)
{
  const fs::path current_path_object {fs::current_path()};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CanBeUsedToInitializeString)
{
  // Returns absolute path of current working directory, obtained as if (in
  // native format) by POSIX getcwd.
  // cf. https://en.cppreference.com/w/cpp/filesystem/current_path
  const std::string current_path {fs::current_path()};

  BOOST_TEST(true);
}


BOOST_AUTO_TEST_SUITE_END() // StdFilesystemCurrent_Path_tests

BOOST_AUTO_TEST_SUITE_END() // CStyleFileIO_tests
BOOST_AUTO_TEST_SUITE_END() // FileIO
BOOST_AUTO_TEST_SUITE_END() // Cpp