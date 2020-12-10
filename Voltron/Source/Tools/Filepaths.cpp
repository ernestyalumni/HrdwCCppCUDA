//------------------------------------------------------------------------------
/// \file Filepaths.cpp
/// \brief Useful directory paths.
///-----------------------------------------------------------------------------
#include <filesystem>

namespace fs = std::filesystem;

namespace Tools
{

fs::path get_source_directory()
{
  return fs::path(__FILE__).parent_path().parent_path();
}

fs::path get_data_directory()
{
  return get_source_directory().parent_path().concat("/data");
}

} // namespace Tools
