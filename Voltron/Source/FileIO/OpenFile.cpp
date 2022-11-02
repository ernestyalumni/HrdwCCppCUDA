#include "OpenFile.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "FileFlagsModes.h"

#include <fcntl.h>
#include <string>
#include <unistd.h> // ::close

using Cpp::Utilities::TypeSupport::get_underlying_value;
using std::string;

namespace FileIO
{

OpenExistingFile::OpenExistingFile(const AccessMode mode):
  access_mode_{mode},
  pathname_{},
  flags_{get_underlying_value(mode)},
  fd_{}
{}

OpenExistingFile::~OpenExistingFile()
{
  if (fd_ >= 0)
  {
    ::close(fd_);
  }
}

void OpenExistingFile::operator()(const string& pathname)
{
  pathname_ = pathname;
  int return_value {::open(pathname_.c_str(), flags_)};

  // TODO: call HandleOpenErrors instance to handle return_value.

  fd_ = return_value;
}

int OpenExistingFile::add_additional_flag(const FileFlag flag)
{
  // Bitwise or'd flags.
  flags_ = flags_ | get_underlying_value(flag);

  return flags_;
}

} // namespace FileIO
