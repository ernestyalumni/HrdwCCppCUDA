#ifndef CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_LINE_H
#define CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_LINE_H

#include "DataStructures/Arrays/DynamicArray.h"

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

class ParseTextLine
{
  public:

    ParseTextLine(const std::string& line);

    ~ParseTextLine() = default;

    DataStructures::Arrays::DynamicArray<std::string> read_in_;

  protected:

    void parse_text_line(const std::string& line);
};

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_LINE_H