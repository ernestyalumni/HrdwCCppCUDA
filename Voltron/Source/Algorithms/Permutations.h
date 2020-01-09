//------------------------------------------------------------------------------
/// \file Permutations.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating permutations.
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_PERMUTATIONS_H
#define ALGORITHMS_PERMUTATIONS_H

#include <cstddef>
#include <string>

namespace Algorithms
{
namespace Permutations
{

namespace Details
{
// cf. https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/

void single_swap(std::string& a, std::size_t l, std::size_t r);

void print_permutations(std::string& a, std::size_t l, std::size_t r);

} // namespace Details

} // namespace Permutations
} // namespace Algorithms

#endif // ALGORITHMS_PERMUTATIONS