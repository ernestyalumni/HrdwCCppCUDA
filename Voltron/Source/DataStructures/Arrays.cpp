//------------------------------------------------------------------------------
/// \file Arrays.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating arrays as an Abstract Data
/// Structure.
/// \ref https://www.hackerrank.com/challenges/ctci-array-left-rotation/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
//-----------------------------------------------------------------------------
#include "Arrays.h"

#include <algorithm> // std::find, std::max;
#include <cstddef> // std::size_t
#include <vector>
#include <utility> // std::swap, also in <algorithm>

using std::find;
using std::max;
using std::swap;
using std::vector;

namespace DataStructures
{
namespace Arrays
{

vector<int> rotate_left(vector<int>& a, const int d)
{
  if (d == a.size())
  {
    return a;
  }

  int l {0};

  for (int iteration {0}; iteration < a.size() / d - 1; ++iteration)
  {
    for (int index {iteration * d}; index < (iteration + 1) * d; ++index)
    {
      swap(a[index], a[index + d]);
    }
  }

  // x is number of elements left to swap with d elements. If x < d, then, we
  // can only do x swaps first.
  const size_t x {a.size() % d};

  // Do d swaps for each of the last x elements.

  for (size_t index {0}; index < x; ++index)
  {
    // Index of the element to swap of the last x elements.
    const size_t element_index {a.size() - x + index};

    // Do d swaps.
    for (int iteration {0}; iteration < d; ++ iteration)
    {
      swap(a[element_index - iteration], a[element_index - iteration - 1]);
    }
  }

  return a;
}

namespace LeetCode
{

bool check_if_double_exists(const vector<int>& arr)
{
  vector<int> possible_double_list;
  vector<int> possible_half_list;

  for (auto ele : arr)
  {
    // check if element is in the double or half list.

    if (
      find(possible_double_list.begin(), possible_double_list.end(), ele) !=
        possible_double_list.end())
    {
      return true;
    }

    if (
      find(possible_half_list.begin(), possible_half_list.end(), ele) !=
        possible_half_list.end())
    {
      return true;
    }

    possible_double_list.emplace_back(2 * ele);

    if (ele % 2 == 0)
    {
      possible_half_list.emplace_back(ele / 2);
    }
  }

  // Did not find an element fitting the criterion.
  return false;
}

CheckIfDoubleExists::CheckIfDoubleExists() = default;

bool CheckIfDoubleExists::checkIfExist(vector<int>& arr)
{
  // This was the fastest.
  // cf. https://leetcode.com/submissions/detail/409397372/?from=/explore/learn/card/fun-with-arrays/527/searching-for-items-in-an-array/3250/
  for (int i {0}; i < arr.size(); ++i)
  {
    for (int j {0}; j < arr.size(); ++j)
    {
      if (i == j)
      {
        continue;
      }
      else if (arr[i] == 2 * arr[j])
      {
        return true;
      }
    }
  }
  return false;
}

bool valid_mountain_array(const vector<int>& a)
{
  // Mountain array iff a.length >= 3.
  if (a.size() < 3)
  {
    return false;
  }

  bool strictly_increasing {true};

  for (size_t i {1}; i < a.size(); ++i)
  {
    if (i == 1 && a[i - 1] >= a[i])
    {
      return false;
    }

    if (strictly_increasing)
    {
      if (a[i - 1] == a[i])
      {
        return false;
      }

      if (a[i - 1] > a[i])
      {
        strictly_increasing = false;
      }
    }
    else
    {
      if (a[i - 1] <= a[i])
      {
        return false;
      }
    }
  }

  // Valid mountain array.
  return strictly_increasing ? false : true;
}

vector<int> replace_with_greatest_on_right(vector<int>& arr)
{
  size_t max_element_index {0};

  for (size_t i {0}; i < arr.size() - 1; ++i)
  {
    size_t j {i + 1};

    // Need to find a new greatest element among elements to its right.
    if (max_element_index < j)
    {
      int max_element_value {arr[j]};

      for (size_t k {j + 1}; k < arr.size(); ++k)
      {
        if (arr[k] > max_element_value)
        {
          max_element_value = arr[k];
          max_element_index = k;
        }
      }

      arr[i] = max_element_value;
    }
    else
    {
      arr[i] = arr[max_element_index];
    }
  }

  arr[arr.size() - 1] = -1;

  return arr;
}

// Think about working backwards for this problem.
vector<int> fastest_replace_with_greatest_on_right(vector<int>& arr)
{
  const size_t N {arr.size()};

  vector<int> maximums (N, -1);

  int current_max {arr.at(N - 1)};

  //maximums[arr.size() - 1] = -1;

  // Solution is to go backwards.
  for (size_t index {N - 2}; index > 0; --index)
  {
    maximums.at(index) = current_max;
 
    if (arr.at(index) > current_max)
    {
      current_max = arr.at(index);
    }
  }

  maximums.at(0) = current_max;

  return maximums;
}

double find_sorted_arrays_median(vector<int>& nums1, vector<int>& nums2)
{
  auto calculate_median = [](vector<int>& nums) -> double
  {
    const size_t N {nums.size()};

    if (N % 2 == 1)
    {
      return nums.at(N / 2);
    }
    else
    {
      return (nums.at(N / 2) + nums.at(N / 2 - 1)) / 2.0;
    }
  };

  if (nums1.empty() || nums2.empty())
  {
    if (nums1.empty())
    {
      return calculate_median(nums2);
    }
    else
    {
      return calculate_median(nums1);
    }
  }

  auto calculate_already_sorted = [](vector<int>& arr1, vector<int>& arr2) ->
    double
  {
    const size_t N {arr1.size() + arr2.size()};

    // N odd
    if (N % 2 == 1)
    {
      return (N / 2 < arr1.size()) ? arr1.at(N / 2) :
        arr2.at(N / 2 - arr1.size());
    }
    else
    {
      if (N / 2 < arr1.size())
      {
        return (arr1.at(N / 2) + arr1.at(N / 2 - 1)) / 2.0;
      }
      else if (arr1.size() == N / 2)
      {
        return (arr2.at(0) + arr1.back()) / 2.0;
      }
      else
      {
        return (arr2.at(N / 2 - arr1.size()) +
          arr2.at(N / 2 - 1 - arr1.size())) / 2.0;
      }
    }
  };

  // Already sorted.
  if (nums1.back() <= nums2.front() || nums2.back() <= nums1.front())
  {
    /*
    const size_t N {nums1.size() + nums2.size()};

    // If N odd,
    if (N % 2 == 1)
    {
      if (nums1.back() <= num2.front())
      { */
        /*
        if (N / 2 < nums1.size())
        {
          return nums1.at(N / 2);
        }
        else
        {
          return nums2.at(N / 2 - nums1.size());
        }
        */
        /*return (N / 2 < nums1.size()) ? nums1.at(N / 2) :
          nums2.at(N / 2 - nums1.size());
      }
      else
      {
        return (N / 2 < nums2.size()) ? nums2.at(N / 2) :
          nums1.at(N / 2 - num2.size());
      }
    }
    // N even
    else
    {
      if (nums1.back() <= num2.front())
      {

      }
    }*/
    if (nums1.back() <= nums2.front())
    {
      return calculate_already_sorted(nums1, nums2);
    }
    else
    {
      return calculate_already_sorted(nums2, nums1);
    }
  }

  // Assume arr1.back() > arr2.front() (otherwise they're already sorted) 
  auto insert_and_sort = [](vector<int>& arr1, vector<int>& arr2)
  {
    arr1.emplace_back(arr2.front());
    arr2.erase(arr2.begin());

    size_t k {arr1.size() - 1};

    while (k > 0 && arr1.at(k - 1) > arr1.at(k))
    {
      swap(arr1.at(k - 1), arr1.at(k));
      --k;
    }
  };

  while (!nums2.empty())
  {
    if (nums1.back() <= nums2.front())
    {
      return calculate_already_sorted(nums1, nums2);
    }

    insert_and_sort(nums1, nums2);
  }

  return calculate_median(nums1);
}

double fastest_find_sorted_arrays_median(vector<int>& nums1, vector<int>& nums2)
{
  const size_t N1 {nums1.size()};
  const size_t N2 {nums2.size()};

  const size_t N {N1 + N2};
  // If N even, then k is the starting index of the "right" half/partition of
  // the total N elements; if N odd, k is the starting index of the "right"
  // half/partition of the total N elements.
  const size_t K {N % 2 == 0 ? N / 2 : N / 2 + 1};        
        
  size_t low {0};
  size_t high {K};
  // midpoint is the literal midpoint if (high - low) = k is odd; otherwise, it
  // is the starting index of the "right" half/partition of the total k
  // elements.
  size_t midpoint {low + (high - low) / 2};
              
  while (low != high)
  {
    size_t i, j;

    if (midpoint > N1)
    {
      i = N1 - 1;
      j = K - N1 - 1;
    }
    else if (K - midpoint > N2)
    {
      j = N2 - 1;
      i = K - N2 - 1;
    }
    else
    {
      i = midpoint - 1;
      j = K - midpoint - 1;
    }

    if (i == -1)
    {
      // We're done.
      if (nums1.empty() || nums1.at(0) >= nums2.at(j))
      {
        break;
      }
      else
      {
        // Take less of nums2
        low = midpoint + 1;
        midpoint = low + (high - low) / 2;
      }
    }
    else if (j == -1)
    {
      // We're done
      if (nums2.empty() || nums2[0] >= nums1[i])
      {
        break;
      }
      else
      {
        // Take less of nums1
        high = midpoint - 1;
        midpoint = low + (high -low) / 2;
      }
    }
    // Which array has largest end?
    else if (nums1[i] > nums2[j])
    {
      // We're done
      if (j == N2 - 1 || nums2[j + 1] >= nums1[i])
      {
        break;
      }
      else
      {
        // Take less of nums1
        high = midpoint - 1;
        midpoint = low + (high - low) / 2;
      }
    }
    else
    {
      // We're done
      if (i = N1 - 1 || nums1[i + 1] >= nums2[j])
      {
        break;
      }
      else
      {
        // Take less of nums2
        low = midpoint + 1;
        midpoint = low + (high - low) / 2;
      }
    }
  }

  {
    int i, j;
    if (midpoint > N1)
    {
      i = N1 - 1;
      j = K - N1 - 1;
    }
    else if (K - midpoint > N2)
    {
      j = N2 - 1;
      i = K - N2 - 1;
    }
    else
    {
      i = midpoint - 1;
      j = K - midpoint - 1;
    }

    double median1 {static_cast<double>(
      max(i == - 1 ? -1 : nums1[i], j == -1 ? -1 : nums2[j]))};

    if (N % 2)
    {
      return median1;
    }

    // One extra step
    double result {0.5 * median1};

    int lastEl2;
    // One more if necessary

    if (j == N2 - 1)
    {
      result += 0.5 * nums1[i + 1];
    }
    else if (i == N1 - 1)
    {
      result += 0.5 * nums2[j + 1];
    }
    else if (nums1[i + 1] <= nums2[j + 1])
    {
      result += 0.5 * nums1[i + 1];
    }
    else
    {
      result += 0.5 * nums2[j + 1];
    }

    return result;
  }        
        
  return 0;      
}


} // namespace LeetCode

} // namespace Arrays
} // namespace DataStructures
