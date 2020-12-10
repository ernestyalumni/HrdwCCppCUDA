//------------------------------------------------------------------------------
/// \file Filepaths.h
/// \brief Useful directory paths.
///-----------------------------------------------------------------------------
#ifndef TOOLS_FILE_PATHS_H
#define TOOLS_FILE_PATHS_H

#include <filesystem>

namespace Tools
{

std::filesystem::path get_source_directory();

std::filesystem::path get_data_directory();

} // namespace Tools

#endif // TOOLS_FILE_PATHS_H