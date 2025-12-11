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

    //--------------------------------------------------------------------------
    /// 1. Iterate Over an Array
    /// Write a function that prints each element in an array in order from the
    /// first to the last.
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    /// 2. Iterate Over an Array in Reverse 
    /// Modify the previous function to print the elements in reverse order,
    /// from the last to the first. 
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static auto iterate_over_array_in_reverse(
      const Container& array,
      const bool is_print = false)
    {
      Container result {};
      for (
        auto it = std::ranges::rbegin(array);
        it != std::ranges::rend(array);
        ++it)
      {
        const auto& element = *it;
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