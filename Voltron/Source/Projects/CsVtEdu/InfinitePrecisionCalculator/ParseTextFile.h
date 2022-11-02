#ifndef CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_FILE_H
#define CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_FILE_H

#include "DataStructures/Arrays/DynamicArray.h"

#include <fstream>
#include <string>

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

class ParseTextFile
{
  public:

    ParseTextFile(const std::string& file_path);

    DataStructures::Arrays::DynamicArray<std::string> read_in_;

  private:

    void parse_stream();

    std::ifstream read_in_stream_;
};

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_FILE_H