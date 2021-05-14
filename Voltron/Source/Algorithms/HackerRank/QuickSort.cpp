#include "QuickSort.h"

#include <algorithm>
#include <iterator> // std::back_inserter
#include <vector>

using std::back_inserter;
using std::copy;
using std::vector;

namespace Algorithms
{
namespace HackerRank
{

namespace Sorting
{
namespace QuickSort
{

vector<int> quick_sort_partition(vector<int>& arr)
{
	// O(N) space complexity addition.
	vector<int> left;
	vector<int> right;

	// Value to compare against.
	const int pivot {arr[0]};

	// O(N-1) time complexity
	for (auto iter = arr.begin() + 1; iter != arr.end(); ++iter)
	{
		if (*iter < pivot)
		{
			left.emplace_back(*iter);
		}
		else
		{
			right.emplace_back(*iter);
		}
	}

	/*
	const int N {static_cast<int>(arr.size())};
	for (int i {1}; i < N; ++i)
	{
		if (arr[i] < pivot)
		{
			left.emplace_back(arr[i]);
		}
		else
		{
			right.emplace_back(arr[i]);
		}
	}*/

	left.emplace_back(pivot);

	copy(right.begin(), right.end(), back_inserter(left));

	return left;
}

} // namespace QuickSort
} // namespace Sorting
} // namespace HackerRank
} // namespace Algorithms
