//------------------------------------------------------------------------------
/// \file SplitString.h
/// \author
/// \brief .
/// \ref https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
///-----------------------------------------------------------------------------
#ifndef UTILITIES_SPLIT_STRING_H
#define UTILITIES_SPLIT_STRING_H

#include <istream>
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

//------------------------------------------------------------------------------
/// \ref https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
/// \details Advantages: works on any stream, not just strings.
/// Cons: Can't split on anything else than spaces,
/// it can be improved in performance,
/// arguably a lot of code for just splitting a string!
//------------------------------------------------------------------------------
std::vector<std::string> split_string_by_iterator(std::string& text);


//------------------------------------------------------------------------------
/// \ref https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
/// \details Solution 1.2, Need to have a string disguised into another type.
/// There are 2 solutions: inheriting from std::string, and wrapping a string
/// with implicit conversion. Here, choose inheritance.
//------------------------------------------------------------------------------

class WordsDelimitedByCommas : public std::string
{
  public:

    WordsDelimitedByCommas() = default;


    //--------------------------------------------------------------------------
    /// \details istream provides support for high level input operations on
    /// character streams.
    //--------------------------------------------------------------------------
    friend std::istream& operator>>(
      std::istream& is,
      WordsDelimitedByCommas& output);
};

std::vector<std::string> split_string_by_overload(std::string& text);

} // namespace Utilities

#endif // UTILITIES_SPLIT_STRING_H