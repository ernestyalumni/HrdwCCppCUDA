#include "CountingSort.h"

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
namespace CountingSort
{

vector<int> counting_sort_frequency(vector<int>& arr)
{
	// explicit vector(
	// 	size_type count,
	// 	const T& value = T(), const Allocator& alloc = Allocator())
	// https://en.cppreference.com/w/cpp/container/vector/vector
	vector<int> frequency (100, 0);

	for (int& x : arr)
	{
		frequency.at(x) += 1;
	}

	return frequency;
}

} // namespace CountingSort
} // namespace Sorting
} // namespace HackerRank
} // namespace Algorithms
