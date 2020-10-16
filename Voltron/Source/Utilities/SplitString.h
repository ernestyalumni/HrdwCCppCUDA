//------------------------------------------------------------------------------
/// \file SplitString.h
/// \author
/// \brief .
/// \ref https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
///-----------------------------------------------------------------------------
#ifndef UTILITIES_SPLIT_STRING_H
#define UTILITIES_SPLIT_STRING_H

#include <string>
#include <vector>

namespace Utilities
{

std::vector<std::string> split_string(
  std::string& s,
  const std::string& delimiter);

std::vector<std::string> split_string(
  const std::string& s,
  const std::string& delimiter);

} // namespace Utilities

#endif // UTILITIES_SPLIT_STRING_H