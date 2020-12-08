//------------------------------------------------------------------------------
/// \file PrintBytes_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \details To run only the Bits unit tests, do this:
/// ./Check --run_test="Algorithms/Bits"
/// 
//------------------------------------------------------------------------------
#include "Algorithms/Bits/PrintBytes.h"
#include "Tools/TemporaryDirectory.h"
#include "Tools/TemporaryFileDescriptor.h"

#include <boost/test/unit_test.hpp>
#include <cstdio> // std::fflush
#include <fcntl.h> // ::open
#include <fstream>
#include <sstream>

using Algorithms::Bits::CVersion::show_bytes;
using Algorithms::Bits::CVersion::show_float;
using Algorithms::Bits::CVersion::show_int;
using Algorithms::Bits::CVersion::show_pointer;
using Tools::TemporaryDirectory;
using Tools::create_temporary_file_and_file_descriptor;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Bits)
BOOST_AUTO_TEST_SUITE(PrintBytes_tests)
BOOST_AUTO_TEST_SUITE(CVersion)

// cf. https://en.cppreference.com/w/cpp/io/c/fflush
//
// int fflush(std::FILE* stream);
//
// For output streams (and for update streams on which last operation was
// output), writes any unwritten data from stream's buffer to associated output
// device.
//
// For input streams (and for update streams on which last operation was input),
// behavior undefined.
//
// Returns 0 on success. Otherwise EOF returned and error indicator of file
// stream is set.
//
// cf. https://man7.org/linux/man-pages/man3/fflush.3.html
// 
// For output streams, fflush() forces write of all user-space buffered data for
// given output or update stream via stream's underlying write function.
//
// cf. https://man7.org/linux/man-pages/man3/stdout.3.html
//
// extern FILE *stdout;
//
// Under normal circumstances, every UNIX program has 3 streams opened for it
// when it starts up, one for input, one for output, one for printing diagnostic
// or error messages. These are typically attached to user's terminal (see tty)
// but might instead refer to files or other devices, depending on what parent
// process chose to set up.
//
// Output stream referred to as "standard output."
//
// Since FILEs are buffering wrapper around UNIX fds, same underlying files may
// also be accessed using raw UNIX file interface, i.e. functions like read and
// lseek.
//
// On program startup, integer fds associated with streams stdin, stdout, and
// stderr are 0, 1, 2, respectively.
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ShowIntPrintsBytesToOutput)
{
  TemporaryDirectory temp_dir {"Temp"};

  auto filename_and_fd =
    create_temporary_file_and_file_descriptor(temp_dir.path(), "temp");

  // cf. https://stackoverflow.com/questions/11110300/how-to-capture-output-of-printf
  std::fflush(stdout);
  int stdout_fd {::dup(STDOUT_FILENO)};
  int redirected_output_fd {
    ::open(filename_and_fd.first.data(), O_WRONLY)};
  ::dup2(redirected_output_fd, STDOUT_FILENO);
  ::close(redirected_output_fd);
  show_int(42);
  ::fflush(stdout);
  ::dup2(stdout_fd, STDOUT_FILENO);
  ::close(stdout_fd);

  // https://stackoverflow.com/questions/132358/how-to-read-file-content-into-istringstream
  std::ifstream file {filename_and_fd.first};

  std::stringstream buffer;

  if (file)
  {
    buffer << file.rdbuf();

    file.close();

    BOOST_TEST(buffer.str().at(0) == ' ');
    BOOST_TEST(buffer.str().at(1) == '2');
    BOOST_TEST(buffer.str().at(2) == 'a');
    
    BOOST_TEST(buffer.str() == " 2a 00 00 00\n");
  }

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitwiseAndToGetIthBit)
{

}

BOOST_AUTO_TEST_SUITE_END() // CVersion
BOOST_AUTO_TEST_SUITE_END() // PrintBytes_tests
BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms