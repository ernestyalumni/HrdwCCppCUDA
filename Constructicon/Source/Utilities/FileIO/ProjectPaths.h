#ifndef UTILITIES_FILE_IO_PROJECT_PATHS
#define UTILITIES_FILE_IO_PROJECT_PATHS

#include <filesystem>
#include <string>

namespace Utilities
{

namespace FileIO
{

class ProjectPaths
{
  public:

    static constexpr char source_subdirectory_name[] {"Source"};

    static constexpr char unit_test_subdirectory_name[] {"UnitTests"};

    inline static std::filesystem::path get_project_path()
    {
      return std::filesystem::path(__FILE__).parent_path().parent_path()
        .parent_path().parent_path();
    }

    inline static std::filesystem::path get_source_path()
    {
      return get_project_path() / std::string{source_subdirectory_name};
    }

    inline static std::filesystem::path get_unit_test_path()
    {
      return get_source_path() / std::string{unit_test_subdirectory_name};
    }
};

} // namespace FileIO

} // namespace Utilities

#endif // UTILITIES_FILE_IO_PROJECT_PATHS