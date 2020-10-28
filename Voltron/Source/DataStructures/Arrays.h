//------------------------------------------------------------------------------
/// \file Arrays.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating arrays as an Abstract Data
/// Structure.
/// \ref https://youtu.be/NLAzwv4D5iI
/// \ref HackerRank Data Structures: Array vs ArrayList
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_RESIZEABLE_ARRAY_H
#define DATA_STRUCTURES_ARRAYS_RESIZEABLE_ARRAY_H

#include <algorithm> // std::copy
#include <iterator> // std::begin, std::end;
#include <stdexcept> // std::runtime_error

namespace DataStructures
{

namespace Arrays
{

//-----------------------------------------------------------------------------
/// new and delete operators allocate memory for objects from a pool called
/// free store.
//-----------------------------------------------------------------------------

/// \date 20201015 15, 17:21 complete
template <typename T>
class ResizeableArray
{
  public:

    static constexpr int default_size_ {8};

    ResizeableArray():
      items_{new T[default_size_]},
      size_{0},
      capacity_{default_size_}
    {}

    ~ResizeableArray()
    {
      // Release block of memory pointed by items_.
      // cf. https://www.softwaretestinghelp.com/new-delete-operators-in-cpp/
      // If delete items, items point to first element of array and this
      // statement will only delete first element of array. Using subscript
      // "[]", indicates variable whose memory is being freed is an array and
      // all memory allocated is to be freed.
      delete[] items_;
    }

    // Accessors

    int size() const
    {
      return size_;
    }

    int capacity() const
    {
      return capacity_;
    }

    T get(const int index)
    {
      // Out of bounds check.
      if (index < 0 || index >= size_)
      {
        throw std::runtime_error("Array Index Out of Bounds Exception");
      }

      return items_[index];
    }

    void set(const int index, T item)
    {
      // Out of bounds check.
      if (index < 0 || index >= size_)
      {
        throw std::runtime_error("Array Index Out of Bounds Exception");
      }
      items_[index] = item;
    }

    void append(T item)
    {
      ensure_extra_capacity();
      items_[size_] = item;
      size_++;
    }

  private:

    void ensure_extra_capacity()
    {
      // Then you have no actual space left.
      // Data Structures: Array vs ArrayList, HackerRank, Sep 20, 2016.
      // 7:01 for https://youtu.be/NLAzwv4D5iI
      // cf. https://stackoverflow.com/questions/37538/how-do-i-determine-the-size-of-my-array-in-c
      //
      // Wrong, No. Gets size of pointer for first, and size of element.
      //if (size_ == sizeof(items_) / sizeof(items_[0]))
      if (size_ == capacity_)
      {
        // Create a new array.
        T* new_copy {new T[size_ * 2]};

        // C++03 way
        std::copy(items_, items_ + size_, new_copy);
        //
        // C++11 way.
        //std::copy(std::begin(items_), std::end(items_), std::begin(new_copy));
        // Items should now point to new_copy.
        items_ = new_copy;

        capacity_ = size_ * 2;
      }
    }

    T* items_;
    int size_;
    int capacity_;
    //T items_[];
};

// Left Rotations
// HackerRank, Arrays: Left Rotation
// cf. https://www.hackerrank.com/challenges/ctci-array-left-rotation/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
// Constraints:
// 1 <= n <= 10^5
// 1 <= d <= n
// If d == n, then do nothing.

// Original signature.
// std::vector<int> rotate_left(std::vector<int> a, int d)
std::vector<int> rotate_left(std::vector<int>& a, const int d);

namespace LeetCode
{

// cf. https://www.geeksforgeeks.org/insertion-sort/
template <typename T>
void insertion_sort(std::vector<T>& v)
{
  // Assume v[0:1] (i.e. subarray of only v[0]) is sorted already.

  // Iterate from k = 1 to k = N - 1 over the array.
  for (std::size_t k {1}; k < v.size(); ++k)
  {
    // Since the list of size k is sorted, consider the kth element.

    // Swap adjacent values until its proper location is found.
    std::size_t j {k - 1};

    while (j >= 0 && v[j + 1] < v[j])
    {
      std::swap(v[j], v[j + 1]);
      j--;
    }
  }
}

bool check_if_double_exists(const std::vector<int>& arr);

//-----------------------------------------------------------------------------
/// \details Important point:
/// Linear search.
///
/// \ref Array 101, Searching for items in an Array.
/// Check If N and Its Double Exist. LeetCode.
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/527/searching-for-items-in-an-array/3250/
//-----------------------------------------------------------------------------
class CheckIfDoubleExists
{
  public:

    CheckIfDoubleExists();

    bool checkIfExist(std::vector<int>& arr);

  private:  

    std::vector<int> possible_double_list_;
    std::vector<int> possible_half_list_;
};

// \url https://leetcode.com/explore/learn/card/fun-with-arrays/527/searching-for-items-in-an-array/3251/
bool valid_mountain_array(const std::vector<int>& a);

/// \brief Replace Elements with Greatest Element on Right Side
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/511/in-place-operations/3259/

std::vector<int> replace_with_greatest_on_right(std::vector<int>& arr);

std::vector<int> fastest_replace_with_greatest_on_right(std::vector<int>& arr);

double find_sorted_arrays_median(
  std::vector<int>& nums1,
  std::vector<int>& nums2);

double fastest_find_sorted_arrays_median(
  std::vector<int>& nums1,
  std::vector<int>& nums2);

} // namespace LeetCode

} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_RESIZEABLE_ARRAY_H