//------------------------------------------------------------------------------
/// \file TemporaryDirectoriesAndFiles_tests.cpp
//------------------------------------------------------------------------------
#include "Tools/TemporaryDirectory.h"
#include "Tools/TemporaryFileDescriptor.h"

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <algorithm> // std::copy_if
#include <array>
#include <climits>
#include <unistd.h> // ::getcwd
#include <iostream>
#include <string>
#include <utility> // std::pair

using Tools::TemporaryDirectory;
using Tools::create_temporary_file_and_file_descriptor;

BOOST_AUTO_TEST_SUITE(Tools)
BOOST_AUTO_TEST_SUITE(TemporaryDirectoriesAndFiles_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetCwdAndMkdtempWorks)
{
  std::array<char, PATH_MAX> buffer_array;
  std::string buffer;

  // #include <unistd.h>
  // char *getcwd(char *buf, size_t size);
  // char *get_current_dir_name(void);
  // getcwd() function copies an absolute pathname of the current working
  // directory to the array pointed to by buf, which is of length size.
  //
  // RETURN VALUE:
  // On success, these functions return ptr to string containing pathname of the
  // current working directory. In case, ::getcwd(), ::getwd() this is same
  // value as buf.
  // On failure, these functions return NULL, errno set.

  std::cout << "\n\n GetCwdAndMkdtempWorks \n";

  std::cout << " Path MAX : " << PATH_MAX << '\n';

  ::getcwd(buffer_array.data(), PATH_MAX);

  //std::string current_pathname {::getcwd(buffer.data(), 0)};
  std::cout << "Buffer : " << buffer_array[0] << '\n';
  //std::cout << current_pathname << '\n';

  auto it = std::find(buffer_array.cbegin(), buffer_array.cend(), '\0');

  std::string return_string {buffer_array.cbegin(), it};
  std::cout << return_string << ' ' << return_string.size() << '\n';

  //std::cout << buffer_array[46] << ' ' << buffer_array[47] << ' ' <<
    //buffer_array[48] << ' ' << buffer_array[49] << '\n';

  //if (buffer_array[48] == '\0')
  //{
  //  std::cout << " null terminated char found " << '\n';
  //}

  std::string current_dir_name_str {::get_current_dir_name()};
  std::cout << current_dir_name_str << '\n';
  //auto output_it =
    //std::copy_if(
      //buffer_array.cbegin(),
      //buffer_array.cend(),
      //buffer.data(),
      //[](v)
      //{
       // v 
      //})

  const std::string temporary_directory_prefix {"Temp"};

//  std::string template_string {temporary_directory_prefix + "XXXXXX"};
  std::string template_string {
    current_dir_name_str + "/" + temporary_directory_prefix + "XXXXXX"};

  std::string resulting_template {::mkdtemp(template_string.data())};
  std::cout << resulting_template << '\n';

  // cf. https://www.boost.org/doc/libs/1_45_0/libs/filesystem/v3/doc/reference.html
  boost::filesystem::remove_all(resulting_template);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TemporaryDirectoryConstructs)
{
  TemporaryDirectory temp_dir {"Temp"};

  // /home/topolo/PropD/HrdwCCppCUDA/Voltron/BuildGcc/TemprAkaZg
  std::cout << temp_dir.path() << '\n';

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TemporaryFileAndFileDescriptorWorks)
{
  TemporaryDirectory temp_dir {"Temp"};

  auto filename_and_fd =
    create_temporary_file_and_file_descriptor(temp_dir.path(), "temp");

  // /home/topolo/PropD/HrdwCCppCUDA/Voltron/BuildGcc/Temp2qFHCe/tempxHk3cf 4
  std::cout << filename_and_fd.first << ' ' << filename_and_fd.second << '\n';
}

BOOST_AUTO_TEST_SUITE_END() // TemporaryDirectoriesAndFiles_tests
BOOST_AUTO_TEST_SUITE_END() // Tools