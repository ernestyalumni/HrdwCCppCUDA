#include "BubbleSort.h"

#include <algorithm> // std::swap
#include <vector>

using std::swap;
using std::vector;

namespace Algorithms
{
namespace HackerRank
{

namespace Sorting
{

namespace BubbleSort
{

int count_swaps(vector<int>& a)
{
	int counter {0};

	int n {static_cast<int>(a.size())};

	for (int i {0}; i < n; ++i)
	{
		for (int j {0}; j < n - 1; ++j)
		{
			// Swap adjacent elements if they are in decreasing order.
			if (a[j] > a[j + 1])
			{
				swap(a[j], a[j + 1]);
				counter++;
			}
		}
	}

	return counter;
}

void print_count_swaps(std::vector<int>& a);

} // namespace BubbleSort
} // namespace Sorting
} // namespace HackerRank
} // namespace Algorithms

