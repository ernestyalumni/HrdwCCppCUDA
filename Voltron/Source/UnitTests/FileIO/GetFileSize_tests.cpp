#include "FileIO/GetFileSize.h"
#include "ExampleTestValues.h"

#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <fstream> // std::ifstream
#include <ios> // std::ios, std::streamsize
#include <limits> // std::numeric_limits
#include <string>

using FileIO::GetFileStatus;
using FileIO::get_file_size;
using UnitTests::FileIO::example_file_path_1;

BOOST_AUTO_TEST_SUITE(FileIO)

BOOST_AUTO_TEST_SUITE(GetFileSize_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StepsToConstructFromWorkingPathWorks)
{
  const std::string current_path_1 {example_file_path_1()};

  // ifstream = basic_ifstream<char>, implements high-level input operations on
  // file-based streams.
  std::ifstream file {};
  // std::ios = std::basic_ios<char>. Type, openmode, in, binary, open in binary
  // mode, open for reading
  file.open(current_path_1, std::ios::in | std::ios::binary);

  if (file.is_open())
  {
    file.ignore(std::numeric_limits<std::streamsize>::max());
    std::streamsize length {file.gcount()};
    BOOST_TEST(length == 784);

    // Since ignore will have set EOF.
    file.clear();
    // Type seekdir, seeking direction type; following constants also defined:
    // beg - the beginning of a stream.
    file.seekg(0, std::ios_base::beg);
  }
  else
  {
    BOOST_TEST(false);
  }

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetFileSizeGetsFileSize)
{
  const std::streamsize result {get_file_size(example_file_path_1())};

  BOOST_TEST(result == 784);
}

BOOST_AUTO_TEST_SUITE_END() // GetFileSize_tests

BOOST_AUTO_TEST_SUITE(GetFileStatus_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  GetFileStatus get_file_status {example_file_path_1()};
}

BOOST_AUTO_TEST_SUITE_END() // GetFileStatus_tests

BOOST_AUTO_TEST_SUITE_END() // FileIO