#include "CountingSort.h"

#include <algorithm>
#include <iostream>
#include <iterator> // std::back_inserter
#include <string>
#include <vector>

using std::back_inserter;
using std::copy;
using std::cout;
using std::ostream_iterator;
using std::stoi;
using std::string;
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

vector<int> counting_sort(vector<int>& arr)
{
	// O(100) in space.
	vector<int> frequency (100, 0)	;

	// O(N) in time.
	for (int& x : arr)
	{
		frequency.at(x) += 1;
	}

	vector<int> sorted;

	// O(N) overall since N elements must be added to permute original input.

	// O(100) in time for this loop.
	for (int i {0}; i < 100; ++i)
	{
		for (int j {0}; j < frequency.at(i); ++j)
		{
			sorted.emplace_back(i);
		}
	}

	return sorted;
}

void count_sort(vector<vector<string>>& arr)
{
	// Assume N is even and 1 <= N <= 1000000
	const int N {static_cast<int>(arr.size())};

	// We're given that value x for the key is an integer such that 0 <= x < 100.
	vector<vector<string>> key_bins (100, vector<string>{});

	for (int i {0}; i < N; ++i)
	{
		const int key {stoi(arr[i][0])};

		if (i < N / 2)
		{
			vector<string>& bin {key_bins[key]};

			bin.emplace_back("-");
		}
		else
		{
			key_bins.at(key).emplace_back(arr[i][1]);
		}
	}

	for (auto& bin : key_bins)
	{
		if (!bin.empty())
		{
			copy(bin.begin(), bin.end(), ostream_iterator<string>(cout, " "));
		}
	}
}

} // namespace CountingSort
} // namespace Sorting
} // namespace HackerRank
} // namespace Algorithms
