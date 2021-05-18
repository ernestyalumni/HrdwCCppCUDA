#ifndef ALGORITHMS_HACKERRANK_SORTING_COUNTING_SORT_COUNTING_SORT_H
#define ALGORITHMS_HACKERRANK_SORTING_COUNTING_SORT_COUNTING_SORT_H

#include <string>
#include <vector>

namespace Algorithms
{
namespace HackerRank
{

namespace Sorting
{
namespace CountingSort
{

//------------------------------------------------------------------------------
/// \ref https://www.hackerrank.com/challenges/countingsort2/problem?h_r=next-challenge&h_v=zen	
/// \details Counting sort is used if you just need to sort a list of integers.
/// Rather than using comparison, create an integer array whose index range
/// covers entire range of values in your array to sort.
//------------------------------------------------------------------------------

std::vector<int> counting_sort_frequency(std::vector<int>& arr);

std::vector<int> counting_sort(std::vector<int>& arr);

void count_sort(std::vector<std::vector<std::string>>& arr);

} // namespace CountingSort
} // namespace Sorting
} // namespace HackerRank
} // namespace Algorithms

#endif // ALGORITHMS_HACKERRANK_SORTING_COUNTING_SORT_COUNTING_SORT_H