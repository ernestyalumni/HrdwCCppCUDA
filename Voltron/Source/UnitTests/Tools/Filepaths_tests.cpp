//------------------------------------------------------------------------------
/// \file Filepaths_tests.cpp
//------------------------------------------------------------------------------
#include "Tools/Filepaths.h"
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using Tools::TemporaryDirectory;
using Tools::get_source_directory;
using Tools::get_data_directory;
using std::ofstream;
using std::ifstream;

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
  std::string filename {"Test.a"};

  const std::string full_filepath {temp_dir.path() + "/" + filename};
  // Create a file.
  std::ofstream(full_filepath, std::ios::binary).write(
    full_filepath.data(), full_filepath.size());

  // Check if directory exists.
  BOOST_TEST(fs::exists(fs::path(temp_dir.path())));

  // Check if file exists.
  BOOST_TEST(fs::exists(fs::path(full_filepath)));
}

BOOST_AUTO_TEST_SUITE_END() // GetDataDirectory_tests

BOOST_AUTO_TEST_SUITE_END() // Filepaths_tests
BOOST_AUTO_TEST_SUITE_END() // Tools