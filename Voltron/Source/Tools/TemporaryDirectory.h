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
    ///
    /// \details Current directory obtained with ::get_current_dir_name
    //--------------------------------------------------------------------------
    explicit TemporaryDirectory(
      const std::string& directory_name_prefix = "Temp");

    ~TemporaryDirectory();

    std::string path() const
    {
      return path_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \details Wraps system call ::mkdtemp
    //--------------------------------------------------------------------------

    std::string make_temporary_directory(
      const std::string& directory_name_prefix,
      const std::string& base_directory_path);

  private:

    std::string path_;
}; // class TemporaryDirectory

//------------------------------------------------------------------------------
/// \fn create_temporary_filename
/// \brief Convenience function to help create a filename existing within a
/// TemporaryDirectory instance.
///
/// \details Help to avoid this mistake by users:
///
/// TemporaryDirectory temp_dir {"Temp"};
/// string test_full_path {temp_dir.path() + "test.txt"};
///
/// in which this results:
/// /home/topolo/PropD/HrdwCCppCUDA/Voltron/BuildGcc/TempOC4Glatest.txt
/// and not this:
/// /home/topolo/PropD/HrdwCCppCUDA/Voltron/BuildGcc/TempOC4Gla/test.txt
/// because the "/" is forgotten between filename and directory.
//------------------------------------------------------------------------------
std::string create_temporary_filename(
  const TemporaryDirectory& temp_dir,
  const std::string& filename);

} // namespace Tools

#endif // TOOLS_TEMPORARY_DIRECTORY_H