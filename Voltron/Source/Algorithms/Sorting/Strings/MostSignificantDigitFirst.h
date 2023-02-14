#ifndef ALGORITHMS_SORTING_STRINGS_MOST_SIGNIFICANT_DIGIT_FIRST_H
#define ALGORITHMS_SORTING_STRINGS_MOST_SIGNIFICANT_DIGIT_FIRST_H

#include <algorithm> // std::fill
#include <cstddef>
#include <string>

namespace Algorithms
{
namespace Sorting
{
namespace Strings
{

namespace MostSignificantDigitFirst
{

int char_at(const std::string& s, const std::size_t d)
{
  return d < s.length() ? static_cast<int>(s[d]) : -1;
}

template <typename ContainerT>
void recursive_step(
  ContainerT& a,
  const std::size_t low,
  const std::size_t high,
  const std::size_t d)
{
  // static means value is maintained between function calls.
  // See https://stackoverflow.com/questions/177437/what-does-const-static-mean-in-c-and-c  
  // This is the smallest subarray size before we use insertion sort.
  static constexpr std::size_t cutoff {15};

  if (high <= low + cutoff)
  {
    insertion_sort(a, low, high, d);
    return;
  }
}



} // namespace MostSignificantDigitFirst

} // namespace Strings
} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_STRINGS_LEAST_SIGNIFICANT_DIGIT_FIRST_H