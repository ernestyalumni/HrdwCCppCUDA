#ifndef ALGORITHMS_SORTING_KEY_INDEXED_COUNTING_H
#define ALGORITHMS_SORTING_KEY_INDEXED_COUNTING_H

#include <algorithm> // std::fill
#include <cstddef>
#include <utility> // std::move

namespace Algorithms
{
namespace Sorting
{

//------------------------------------------------------------------------------
/// See Sec. 5.1, String Sorts, Ch. 5 STrings, pp. 705 of Algorithms, 4th Ed.
/// Robert Sedgewick, Kevin Wayne.
//------------------------------------------------------------------------------
template <int MaximumKeyValue, typename ContainerT>
ContainerT key_indexed_counting_sort(ContainerT& a)
{
  // Compute frequency counts.
  // We add 2 more entries than the maximum value that a key can take (e.g. use
  // a Section number to be a key, and if the maximum section value is 4, then
  // the size is 4 + 2 = 6). This allows for if the key value is r, we increment
  // count[r + 1].
  // count[0] is always 0.
  const std::size_t count_size {static_cast<std::size_t>(MaximumKeyValue) + 2};
  int count[count_size] {};
  std::fill(count, count + count_size, 0);

  // O(N) time complexity, N is the number of elements to sort.
  for (auto& entry : a)
  {
    count[static_cast<std::size_t>(entry.key()) + 1] += 1;
  }

  // Compute cumulates.
  // See pp. 704, Ch. 5 Strings, Algorithms, 4th Ed. by Robert Sedgewick, Kevin
  // Wayne, "Transform counts to indices".
  // Next, use count[] to compute, for each key value, the starting index
  // positions in sorted order of items with that key. This explains why we had
  // placed the frequencies in the "next" index position to account for the
  // frequencies prior, in order to get the start of the items.
  //
  // O(R) time complexity for R being the maximum key value.
  for (std::size_t r {0}; r < count_size; ++r)
  {
    count[r + 1] += count[r];
  }

  // Distribute the data.
  // See pp. 704, Ch. 5 Strings, Algorithms, 4th Ed. by Robert Sedgewick, Kevin
  // Wayne, "Distribute the data."

  ContainerT auxiliary_array (a.size());

  // O(N) time complexity for N counter increments and N data moves.
  for (std::size_t i {0}; i < a.size(); ++i)
  {
    const std::size_t key_index {static_cast<std::size_t>(a[i].key())};

    // Recall, the value given from count obtains the starting index for our
    // new, sorted array, given a key.

    auxiliary_array[count[key_index]] = a[i];

    // Increment that count entry to maintain the following invariant for count:
    // for each key value r, count[r] is the index of the position in auxiliary
    // where the next item with key value r (if any) should be placed, i.e.
    // as you're adding back the elements in order, you'll increment the
    // starting index where you place the next entry, for a given key.
    count[key_index] += 1;
  }

  return std::move(auxiliary_array);
}

} // namespace Sorting
} // namespace Algorithms

#endif // ALGORITHMS_SORTING_KEY_INDEXED_COUNTING_H