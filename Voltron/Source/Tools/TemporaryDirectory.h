//------------------------------------------------------------------------------
/// \file TemporaryDirectory.h
/// \brief Temporary directory.
///-----------------------------------------------------------------------------
#ifndef _TOOLS_TEMPORARY_DIRECTORY_H_
#define _TOOLS_TEMPORARY_DIRECTORY_H_

#include <string>

namespace Tools
{

class TemporaryDirectory
{
  public:
    TemporaryDirectory(
      const std::string& directory_name_prefix,
      const std::string& base_directory_path);

    //--------------------------------------------------------------------------
    /// \param directory_name_prefix
    /// e.g. "Temp" or i.e. /Temp
    //--------------------------------------------------------------------------
    TemporaryDirectory(const std::string& directory_name_prefix = "Temp");

    ~TemporaryDirectory();

    std::string temporary_directory_path() const
    {
      return temporary_directory_path_;
    }

  protected:

    std::string make_temporary_directory(
      const std::string& directory_name_prefix,
      const std::string& base_directory_path);

  private:

    std::string temporary_directory_path_;
}; // class TemporaryDirectory

} // namespace Tools

#endif // _TOOLS_TEMPORARY_DIRECTORY_H_