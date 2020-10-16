//------------------------------------------------------------------------------
/// \file SplitString.cpp
/// \author
/// \brief .
/// \ref https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
///-----------------------------------------------------------------------------
#include "SplitString.h"

#include <cstddef> // std::size_t
#include <string>
#include <vector>

using std::size_t;
using std::string;
using std::vector;

namespace Utilities
{

vector<string> split_string(string& s, const string& delimiter)
{
  size_t position {0};
  vector<string> split_strings;

  // npos is a special value equal to maximum value, end of string indicator for
  // functions expecting a string index.
  // find characters in the string. Returns position of first character of found
  // substring or npos if no such substring is found.
  while ((position = s.find(delimiter)) != string::npos)
  {
    // returns a substring [pos, pos+count)
    split_strings.emplace_back(s.substr(0, position));

    // Removes count characeters starting at index, if erase(index, count).
    s.erase(0, position + delimiter.length());
  }
  split_strings.emplace_back(s);

  return split_strings;
}

vector<string> split_string(const string& s, const string& delimiter)
{
  string modifiable_s {s};

  return split_string(modifiable_s, delimiter);
}

} // namespace Utilities

