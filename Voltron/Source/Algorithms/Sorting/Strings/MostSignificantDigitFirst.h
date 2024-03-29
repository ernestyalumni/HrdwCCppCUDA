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

//------------------------------------------------------------------------------
/// \brief Obtain integer value of the character at position d (0-based
/// counting) for the given string.
/// \return -1 if specified character position is past the end of the string.
//------------------------------------------------------------------------------
int char_at(const std::string& s, const std::size_t d);

//------------------------------------------------------------------------------
/// \details Start comparing from the dth letter or, i.e. dth character.
/// Strings that have the same characters in the same order, but for the string
/// that is of shorter length (so it's "truncated"), then it's considered to be
/// "less than" the other string.
//------------------------------------------------------------------------------
bool is_less(const std::string& v, const std::string& w, const std::size_t d);

//------------------------------------------------------------------------------
/// \details Avoids the cost of reexamining characters that we know to be equal,
/// so extra argument d so that we assume the first d characters of all strings
/// to be sorted and known to be equal (though this doesn't depend on this fact
/// to be true or not).
//------------------------------------------------------------------------------
template <typename ContainerT>
void insertion_sort(
  ContainerT& a,
  const std::size_t low,
  const std::size_t high,
  const std::size_t d)
{
  for (std::size_t i {low}; i <= high; ++i)
  {
    for (std::size_t j {i}; j > low && is_less(a[j], a[j - 1], d); --j)
    {
      std::string temp {a[j]};
      a[j] = a[j - 1];
      a[j - 1] = temp;
    }
  }
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

  // Radix.
  static constexpr std::size_t R {256};

  if (high <= low + cutoff)
  {
    insertion_sort(a, low, high, d);
    return;
  }

  // 0 already signifies "end of string".
  // Initialize array of the alphabet size, + 2.
  int count[R + 2] {};

  // Compute frequency counts.
  for (std::size_t i {low}; i <= high; ++i)
  {
    count[static_cast<std::size_t>(char_at(a[i], d) + 2)] += 1;
  }

  // Transform count to indices.
  for (std::size_t r {0}; r < R + 1; ++r)
  {
    count[r + 1] += count[r];
  }

  ContainerT auxiliary_array (a.size());

  // Distribute.
  for (std::size_t i {low}; i <= high; ++i)
  {
    const std::size_t key_index {static_cast<std::size_t>(char_at(a[i], d))};

    auxiliary_array[count[key_index + 1]] = a[i];

    count[key_index + 1] += 1;
  }

  // Copy back.
  for (std::size_t i {low}; i <= high; ++i)
  {
    a[i] = auxiliary_array[i - low];
  }

  // Recursively sort for each character value.
  for (std::size_t r {0}; r < R; ++r)
  {
    recursive_step(a, low + count[r], low + count[r + 1] - 1, d + 1);
  }
}

template <typename ContainerT>
void most_significant_digit_first_sort(ContainerT& a)
{
  recursive_step(a, 0, a.size() - 1, 0);
}

} // namespace MostSignificantDigitFirst

} // namespace Strings
} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_STRINGS_LEAST_SIGNIFICANT_DIGIT_FIRST_H