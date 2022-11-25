#include "ParseTextLine.h"

#include <cctype>
#include <string>

using std::string;

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

ParseTextLine::ParseTextLine(const string& line):
  read_in_{}
{
  parse_text_line(line);
}

void ParseTextLine::parse_text_line(const string& line)
{
  bool is_operand {false};
  bool not_in_leading_zeros {false};

  string parsed {};

  for (char c : line)
  {
    if (!is_operand && !not_in_leading_zeros)
    {
      parsed.clear();
    }

    // cf. https://en.cppreference.com/w/cpp/string/byte/isspace
    // Return value: non-zero if character is a whitespace character, 0
    // otherwise.
    if (std::isspace(c) != 0)
    {
      if (!parsed.empty())
      {
        read_in_.append(parsed);
      }

      is_operand = false;
      not_in_leading_zeros = false;
    }
    else if (std::isdigit(c))
    {
      if (c == '0')
      {
        if (not_in_leading_zeros)
        {
          parsed += c;
        }
      }
      else
      {
        not_in_leading_zeros = true;
        parsed += c;
      }

      is_operand = true;
    }
    else if (c == '+' || c =='*' || c == '^')
    {
      parsed += c;
      read_in_.append(parsed);
      parsed.clear();
      is_operand = false;
      not_in_leading_zeros = false;
    }
  }
}

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu
