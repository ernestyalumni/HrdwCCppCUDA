//------------------------------------------------------------------------------
/// \file TemporaryFileDescriptor.h
/// \brief Temporary file (with file name) and file descriptor.
///-----------------------------------------------------------------------------
#ifndef _TOOLS_TEMPORARY_FILE_DESCRIPTOR_H_
#define _TOOLS_TEMPORARY_FILE_DESCRIPTOR_H_

#include <stdlib.h>
#include <string>
#include <utility> // std::pair

namespace Tools
{

//decltype(auto) create_temporary_file_and_file_descriptor(
//  const std::string& directory_path)

//------------------------------------------------------------------------------
/// \fn create_temporary_file_and_file_descriptor
/// \param directory_path e.g. "base/username/tempdir"
/// \return std::pair of the template string and the file descriptor as an
/// integer. Example of the template string:
/// /Voltron/BuildGcc/Temp0vsqP1/tempVkNj51
//------------------------------------------------------------------------------ 
decltype(auto) create_temporary_file_and_file_descriptor =
  [](
    const std::string& directory_path,
    const std::string& filename_prefix
    ) -> std::pair<std::string, int>
  {
    std::string template_string {
      directory_path + "/" + filename_prefix + "XXXXXX"};

    const int fd {::mkstemp(template_string.data())};

    return std::pair<std::string, int>{template_string, fd};
  };



} // namespace Tools

#endif // _TOOLS_TEMPORARY_FILE_DESCRIPTOR_H_