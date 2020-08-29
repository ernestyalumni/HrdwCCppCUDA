//------------------------------------------------------------------------------
/// \file TemporaryDirectory.h
/// \brief Temporary directory.
///-----------------------------------------------------------------------------
#ifndef TOOLS_TEMPORARY_DIRECTORY_H
#define TOOLS_TEMPORARY_DIRECTORY_H

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
    /// \brief Create a temporary directory starting from the current directory
    /// the function is called from.
    /// \param directory_name_prefix
    /// e.g. "Temp" or i.e. /Temp
    //--------------------------------------------------------------------------
    explicit TemporaryDirectory(
      const std::string& directory_name_prefix = "Temp");

    ~TemporaryDirectory();

    std::string path() const
    {
      return path_;
    }

  protected:

    std::string make_temporary_directory(
      const std::string& directory_name_prefix,
      const std::string& base_directory_path);

  private:

    std::string path_;
}; // class TemporaryDirectory

} // namespace Tools

#endif // TOOLS_TEMPORARY_DIRECTORY_H