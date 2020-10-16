//------------------------------------------------------------------------------
/// \file Permutations.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating permutations.
///-----------------------------------------------------------------------------
#include "Permutations.h"

#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric> // std::iota
#include <string>
#include <vector>

namespace Algorithms
{
namespace Permutations
{

namespace Details
{

void single_swap(std::string& a, std::size_t l, std::size_t r)
{
  assert(a.length() > l && a.length() > r);

  if (l == r)
  {
    return;
  }

  const char temp {a[l]};
  // https://en.cppreference.com/w/cpp/string/basic_string/replace
  // replace(size_type pos, size_type count, const basic_string& str);
  // replace(size_type pos, size_type count, const CharT* cstr);
  a.replace(l, 1, std::string{a[r]});
  a.replace(r, 1, std::string{temp});
}

// https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/
void print_permutations(std::string& a, std::size_t l, std::size_t r)
{
  // Base case
  if (l == r)
  {
    // Uncomment this out.
    //std::cout << l << " : ";
    //std::cout << a << '\n';
  }

  for (std::size_t index {l}; index <= r; ++index)
  {
    //std::cout << " indices : " << index << l << r << " ";
    // Swapping done.
    single_swap(a, l, index);
    //std::cout << " First swap : " << a << ' ';

    // Recursion called.
    print_permutations(a, l + 1, r);

    // backtrack.
    single_swap(a, l, index);

    std::cout << " Second swap : " << a << ' ';
  }
}

} // namespace Details

} // namespace Permutations
} // namespace Algorithms
