//------------------------------------------------------------------------------
/// \file Filepaths_tests.cpp
//------------------------------------------------------------------------------
#include "Tools/Filepaths.h"
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <filesystem>

namespace fs = std::filesystem;

using Tools::TemporaryDirectory;
using Tools::get_source_directory;
using Tools::get_data_directory;

BOOST_AUTO_TEST_SUITE(Tools)
BOOST_AUTO_TEST_SUITE(Filepaths_tests)

BOOST_AUTO_TEST_SUITE(GetSourceDirectory_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetSourceDirectoryGetsSourceDirectory)
{
  BOOST_TEST(
    get_source_directory().lexically_relative(
      get_source_directory().parent_path()) == "Source");

  BOOST_TEST(fs::is_directory(get_source_directory()));
}

BOOST_AUTO_TEST_SUITE_END() // GetSourceDirectory_tests

BOOST_AUTO_TEST_SUITE(GetDataDirectory_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetDataDirectoryGetsDataDirectory)
{
  BOOST_TEST(
    get_data_directory().lexically_relative(
      get_data_directory().parent_path()) == "data");

  BOOST_TEST(fs::is_directory(get_data_directory()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PathCanBeUsedToCheckFileExistence)
{
  TemporaryDirectory temp_dir {"/temp"};
  

}

BOOST_AUTO_TEST_SUITE_END() // GetDataDirectory_tests

BOOST_AUTO_TEST_SUITE_END() // Filepaths_tests
BOOST_AUTO_TEST_SUITE_END() // Tools