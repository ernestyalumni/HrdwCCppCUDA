#include "InsertionSort.h"

#include <algorithm> // std::copy
#include <iostream>
#include <iterator> // std::ostream_iterator
#include <vector>

using std::copy;
using std::cout;
using std::ostream_iterator;
using std::vector;

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
/// \details Stopping condition guaranteed partly because of property that
/// existing array is sorted.
///
/// Time Complexity: O(n)
//------------------------------------------------------------------------------	
void insertion_sort_1(const int n, vector<int>& arr)
{
	// Store this value for comparison to all others.
	const int pivot {arr[n - 1]};

	for (int i {n - 2}; i > -1; --i)
	{
		if (arr[i] > pivot)
		{
			// Shift value to the right.
			arr[i + 1] = arr[i];

			// https://stackoverflow.com/questions/4153110/how-do-i-use-for-each-to-output-to-cout
			// https://en.cppreference.com/w/cpp/iterator/ostream_iterator/ostream_iterator
			// ostream_iterator(ostream_type& stream, const charT* delim)
			// delim as delimiter.
			copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
			cout << "\n";

			if (i == 0)
			{
				arr[i] = pivot;

				copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
			}
		}
		else
		{
			// Insertion.
			arr[i + 1] = pivot;	

			copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));

			break;
		}
	}
}

//------------------------------------------------------------------------------
/// \ref https://www.hackerrank.com/challenges/correctness-invariant/problem
/// \details
///
/// Time Complexity: O(N^2)
//------------------------------------------------------------------------------	
void insertion_sort(const int N, int arr[])
{
	// arr[0] by itself is sorted. So i = 0 case is done.

	for (int i {1}; i < N; ++i)
	{
		// Store this value for comparison to all others.
		const int pivot {arr[i]};

		int j {i - 1};

		// Shift values to the right anytime pivot value less than array element.
		while (j > -1 && pivot < arr[j])
		{
			arr[j + 1] = arr[j];
			j--;
		}

		// Insert pivot value.
		arr[j + 1] = pivot;
	}
}

int running_time(vector<int>& arr)
{
	const int N {static_cast<int>(arr.size())};

	if (N == 0 || N == 1)
	{
		// No shifts needed.
		return 0;
	}

	int shift_counter {0};

	for (int i {1}; i < N; ++i)
	{
		const int pivot {arr[i]};

		int j {i - 1};

		// Shift values to the right anytime pivot value less than array element.
		while (j > - 1 && pivot < arr[j])
		{
			arr[j + 1] = arr[j];
			shift_counter++;
			j--;
		}

		// Insert pivot value.
		arr[j + 1] = pivot;
	}

	return shift_counter;
}

} // namespace InsertionSort
} // namespace Sorting
} // namespace HackerRank
} // namespace Algorithms
