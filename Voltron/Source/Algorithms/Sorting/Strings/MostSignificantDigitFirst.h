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

static constexpr

} // namespace MostSignificantDigitFirst

//------------------------------------------------------------------------------
/// See https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/MSD.java
/// \brief Sort an array of strings, on the leading "word_length" characters.
/// See pp. 706, Ch. 5, Algorithms, 4th Ed. Robert Sedgewick, Kevin Wayne.
/// See pp. 709. O(WN) time complexity. An input array of N strings that each
/// have W characters has a total of WN characters, so running time of Least
/// Significant Digit string sort is linear in size of input.
//------------------------------------------------------------------------------
template <typename ContainerT>
void most_significant_digit_first_sort(
    ContainerT& a,
    const std::size_t word_length)
{
  constexpr std::size_t extended_ascii_alphabet_size {256};

  ContainerT auxiliary_array (a.size());

  // Sort by key-indexed counting on dth character.
  for (int d {static_cast<int>(word_length) - 1}; d >= 0; --d)
  {
    // Compute frequency counts.
    int count[extended_ascii_alphabet_size + 1] {};
    std::fill(count, count + extended_ascii_alphabet_size + 1, 0);

    for (auto& entry : a)
    {
      count[static_cast<std::size_t>(entry[d]) + 1] += 1;
    }

    // Compute cumulates.
    // This gives the index, for each "key" or "letter", to start "adding from"
    // on the auxiliary array.
    for (std::size_t r {0}; r <= extended_ascii_alphabet_size; ++r)
    {
      count[r + 1] += count[r];
    }

    // Move data, i.e. distribute the data according to how it should be sorted.
    for (std::size_t i {0}; i < a.size(); ++i)
    {
      const std::size_t key_index {static_cast<std::size_t>(a[i][d])};

      auxiliary_array[count[key_index]] = a[i];

      // Increment count entry for where next item with key value r should be
      // placed.
      count[key_index] += 1;
    }

    // Copy back.
    // TODO: Why doesn't .end() work with std::copy?
    std::copy(
      auxiliary_array.begin(),
      auxiliary_array.begin() + a.size(),
      a.begin());
    // This works as well.
    /*
    for (std::size_t i {0}; i < a.size(); ++i)
    {
      a[i] = auxiliary_array[i];
    }
    */
  }
}

} // namespace Strings
} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_STRINGS_LEAST_SIGNIFICANT_DIGIT_FIRST_H