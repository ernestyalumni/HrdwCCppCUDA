#include "FileFlagsModes.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

using Cpp::Utilities::TypeSupport::get_underlying_value;

namespace FileIO
{

int to_access_mode_value(const AccessMode access_mode)
{
  return get_underlying_value<AccessMode>(access_mode);
}

int to_file_flag_value(const FileFlag file_flag)
{
  return get_underlying_value<FileFlag>(file_flag);
}


} // namespace FileIO
