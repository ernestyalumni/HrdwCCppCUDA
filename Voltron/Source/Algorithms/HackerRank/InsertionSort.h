#ifndef ALGORITHMS_HACKERRANK_SORTING_INSERTION_SORT_INSERTION_SORT_H
#define ALGORITHMS_HACKERRANK_SORTING_INSERTION_SORT_INSERTION_SORT_H

#include <vector>

namespace Algorithms
{
namespace HackerRank
{

namespace Sorting
{
namespace InsertionSort
{

//------------------------------------------------------------------------------
/// \ref https://www.hackerrank.com/challenges/insertionsort1/problem
//------------------------------------------------------------------------------	

void insertion_sort_1(const int n, std::vector<int>& arr);

void insertion_sort(const int N, int arr[]);

int running_time(std::vector<int>& arr);

} // namespace InsertionSort
} // namespace Sorting
} // namespace HackerRank
} // namespace Algorithms

#endif // ALGORITHMS_HACKERRANK_SORTING_INSERTION_SORT_INSERTION_SORT_H