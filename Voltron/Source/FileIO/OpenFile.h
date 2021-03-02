//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man2/open.2.html
/// \brief Wrapper for ::open, which opens the file specified.
//------------------------------------------------------------------------------
#ifndef FILE_IO_OPEN_FILE_H
#define FILE_IO_OPEN_FILE_H

#include "FileFlagsModes.h"

#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <string>

namespace FileIO
{

namespace Details
{

class HandleOpenErrors
{

};

} // namespace Details

class OpenExistingFile
{
  public:

    OpenExistingFile(const AccessMode mode);

    void operator()(const std::string& pathname);

    int add_additional_flag(const FileFlag flag);

    // Accessors.

    AccessMode access_mode() const
    {
      return access_mode_;
    }

    std::string pathname() const
    {
      return pathname_;
    }

    int flags() const
    {
      return flags_;
    }

    int fd() const
    {
      return fd_;
    }

  private:

    AccessMode access_mode_;
    std::string pathname_;
    int flags_;
    int fd_;
};

} // namespace FileIO

#endif // FILE_IO_OPEN_FILE_H
