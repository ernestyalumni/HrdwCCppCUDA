#include "FileIO/FileFlagsModes.h"
#include "FileIO/OpenFile.h"
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <fcntl.h>
#include <filesystem>

using FileIO::AccessMode;
using FileIO::OpenExistingFile;
using Tools::TemporaryDirectory;

BOOST_AUTO_TEST_SUITE(FileIO)
BOOST_AUTO_TEST_SUITE(OpenFile_tests)

BOOST_AUTO_TEST_SUITE(OpenExistingFile_tests)

namespace fs = std::filesystem;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OpensFile)
{
  TemporaryDirectory temp_dir {"Temp"};

  OpenExistingFile open_file {AccessMode::write_only};
  open_file.add_additional_flag(FileFlag::nonblocking_open);
  open_file.add_additional_flag(FileFlag::synchronize_writes);

  open_file(temp_dir.path() + "/TempFile");

  // Don't need to do this - TemporaryDirectory destructor will remove it.
  //::close(open_file.fd());

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // OpenExistingFile_tests

BOOST_AUTO_TEST_SUITE_END() // OpenFile_tests
BOOST_AUTO_TEST_SUITE_END() // FileIO