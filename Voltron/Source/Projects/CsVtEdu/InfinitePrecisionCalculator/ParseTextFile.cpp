#include "ParseTextFile.h"

#include <string>

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

ParseTextFile::ParseTextFile(const std::string& file_path):
  read_in_{},
  read_in_stream_{file_path}
{
  parse_stream();
}

void ParseTextFile::parse_stream()
{
  bool expression_started {false};
  std::string current_string {};

  // https://en.cppreference.com/w/cpp/string/basic_string/getline
  // getline(std::basic_istream<>& input, std::basic_string<>& str), extracts
  // characters from input and appends them to str until endline character
  // as delimiter.
  for (std::string line {}; std::getline(read_in_stream_, line);)
  {
    bool is_empty_line {true};
    for (std::size_t i {0}; i < line.size(); ++i)
    {
      // https://en.cppreference.com/w/cpp/string/byte/isspace
      // Checks white space characters: ' ', '\f', '\n', '\r', '\t', '\v'
      if (!std::isspace(line[i]))
      {
        is_empty_line = false;
      }
    }

    if (!is_empty_line)
    {
      // We have the continuation of an expression.
      if (expression_started)
      {
        current_string += line;
      }
      // We have the start of an expression.
      else
      {
        current_string = line;
        expression_started = true;
      }
    }
    else
    {
      // We have the end of an expression; otherwise, no expression has come
      // up yet.
      if (expression_started)
      {
        expression_started = false;
        read_in_.append(current_string);
        // Remember to clear the current string since we have the end of an
        // expression.
        current_string.clear();
      }
    }
  }
}

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu
