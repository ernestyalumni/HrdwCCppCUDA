#ifndef CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_FILE_H
#define CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_FILE_H

#include "DataStructures/Arrays/DynamicArray.h"
#include "DataStructures/LinkedLists/DoublyLinkedList.h"

#include <cstddef>
#include <fstream>
#include <string>
#include <type_traits>

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

template <typename T>
using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<T>;

class ParseTextFile
{
  public:

    ParseTextFile(const std::string& file_path);

    DataStructures::Arrays::DynamicArray<std::string> read_in_;

  private:

    void parse_stream();

    std::ifstream read_in_stream_;
};

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void strip_leading_zeros(DoublyLinkedList<T>& ll)
{
  bool reached_leading_nonzero_value_or_last_value {false};

  while (!reached_leading_nonzero_value_or_last_value)
  {
    if (ll.size() <= 1 || ll.tail()->retrieve() != static_cast<T>(0))
    {
      reached_leading_nonzero_value_or_last_value = true;
    }
    else
    {
      ll.pop_back();
      ll.get_tail();
    }
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
int convert_to_int(const DoublyLinkedList<T>& ll)
{
  int result {0};
  int digit_position {1};

  auto iterator = ll.begin();

  while (iterator != ll.end())
  {
    result += static_cast<int>(*iterator) * digit_position;
    ++iterator;
    digit_position *= 10;
  }

  return result;
}

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_PARSE_TEXT_FILE_H