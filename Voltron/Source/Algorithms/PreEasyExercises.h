#ifndef ALGORITHMS_PRE_EASY_EXERCISES_H
#define ALGORITHMS_PRE_EASY_EXERCISES_H

#include <concepts>
#include <iostream>
#include <vector>

namespace Algorithms
{
namespace PreEasyExercises
{

//------------------------------------------------------------------------------
/// https://blog.faangshui.com/p/before-leetcode
/// 1. Array Indexing
/// Understanding how to navigate arrays is essential. Here are ten exercises,
//  sorted in increasing difficulty, that build upon each other:
//------------------------------------------------------------------------------
class ArrayIndexing
{
  public:

    template <std::ranges::range Container>
    static auto iterate_over_array(
      const Container& array,
      const bool is_print = false)
    {
      Container result {};
      for (const auto& element : array)
      {
        result.push_back(element);
        if (is_print)
        {
          std::cout << element << ' ';
        }
      }
      return result;
    }
};

} // namespace PreEasyExercises
} // namespace Algorithms

#endif // ALGORITHMS_PRE_EASY_EXERCISES_H