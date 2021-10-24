//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating arrays as an Abstract Data
/// Structure.
/// \ref https://youtu.be/NLAzwv4D5iI
/// \ref HackerRank Data Structures: Array vs ArrayList
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_H
#define DATA_STRUCTURES_ARRAYS_H

#include <algorithm> // std::copy
#include <cstddef> // std::size_t
#include <cstdlib> // EXIT_FAILURE // implementation defined.
#include <initializer_list>
#include <iterator> // std::begin, std::end;
#include <stdexcept> // std::runtime_error
#include <vector>

namespace DataStructures
{

namespace Arrays
{

template <typename T>
class CStyleDynamicArray
{
  public:

    explicit CStyleDynamicArray(const std::size_t N):
      data_{new T[N]{}},
      size_{N}
    {}

    explicit CStyleDynamicArray(const std::initializer_list<T> list):
      data_{new T[list.size()]},
      size_{list.size()}
    {
      T* data_ptr {data_};

      for (auto x : list)
      {
        *data_ptr = x;
        ++data_ptr;
      }
    }

    ~CStyleDynamicArray()
    {
      delete[] data_;
    }

    T operator[](const std::size_t index)
    {
      return data_[index];
    }

    //--------------------------------------------------------------------------
    /// If an object instance is constructed as const, then this function
    /// overload must be used (otherwise, the element can be accessed and set,
    /// contradicting that it's const).
    //--------------------------------------------------------------------------
    T operator[](const std::size_t index) const
    {
      return data_[index];
    }

    std::size_t size() const
    {
      return size_;
    }

    // Returns an iterator to the beginning.
    constexpr T* begin()
    {
      return data_;
    }

    // Returns an iterator to the end.
    constexpr T* end()
    {
      return data_ + size_;
    }

  private:

    T* data_;
    std::size_t size_;
};

//-----------------------------------------------------------------------------
/// new and delete operators allocate memory for objects from a pool called
/// free store.
/// TODO: Consider obsoleting this function.
//-----------------------------------------------------------------------------
/// \date 20201015 15, 17:21 complete
template <typename T>
class ResizingArray
{
  public:

    static constexpr int default_size_ {8};

    ResizingArray():
      items_{new T[default_size_]},
      size_{0},
      capacity_{default_size_}
    {}

    ~ResizingArray()
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
        
        delete[] items_;

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

template <typename T>
class Array
{
  public:

    explicit Array(const std::size_t L, const std::size_t l = 0):
      items_{new T[L]},
      length_{l},
      capacity_{L}
    {
      if (l > L)
      {
        throw std::runtime_error("Array length Out of Bounds Exception");
      }
    }

    explicit Array(const std::size_t L, const std::size_t l, T initial_value):
      items_{new T[L]},
      length_{l},
      capacity_{L}
    {
      if (l > L)
      {
        throw std::runtime_error("Array length Out of Bounds Exception");
      }

      for (std::size_t i {0}; i < length_; ++i)
      {
        items_[i] = initial_value;
      }
    }

    ~Array()
    {
      delete[] items_;
    }

    //--------------------------------------------------------------------------
    /// \details Implementation of [] operator. This function must return a
    /// reference as array element can be put on left side.
    //--------------------------------------------------------------------------

    T& operator[](const std::size_t index)
    {
      if (index >= length_)
      {
        throw std::runtime_error("Array index out of bound");
        //return EXIT_FAILURE;
      }

      return items_[index];
    }

    T& at(const std::size_t index)
    {
      if (index >= length_)
      {
        throw std::runtime_error("Array index out of bound");
      }
      return items_[index];
    }

    void append(T item)
    {
      ensure_extra_capacity();
      items_[length_] = item;
      // Adding item to the end has grown the array by size 1.
      ++length_;
    }

    T back() const
    {
      return items_[length() - 1];
    }    

    void pop_back()
    {
      --length_;
    }

    void insert(T item, const std::size_t index)
    {
      if (index > length_)
      {
        throw std::runtime_error("Array index out of bound");
      }

      ensure_extra_capacity();

      // Shift to the right.

      // reverse_index = L, L - 1, ... 0 for index = 0, 1, ... L - 1, L.
      std::size_t reverse_index {length_ - index};

      for (std::size_t reverse_i {0}; reverse_i < reverse_index; ++reverse_i)
      {
        const std::size_t target_index {length_ - reverse_i};

        items_[target_index] = items_[target_index - 1];
      }

      items_[index] = item;

      // Inserting an item has grown the array by size 1.
      ++length_;
    }

    void delete_at_index(const std::size_t index)
    {
      if (index >= length_)
      {
        throw std::runtime_error("Array index out of bound");
      }

      // Starting at index, shift each element 1 position to the left.
      for (size_t i {index}; i < (length_ - 1); ++i)
      {
        items_[i] = items_[i + 1];
      }

      // Note that it's important to reduce the length of the array by 1.
      // Otherwise, we'll lose consistency of the size. This length variable
      // is the only thing controlling where new elements might get added.
      --length_;
    }

    std::size_t length() const
    {
      return length_;
    }

    std::size_t capacity() const
    {
      return capacity_;
    }

  private:

    void ensure_extra_capacity()
    {
      // Double the capacity once the number of elements, length_ filled up
      // capacity.
      if (length_ == capacity_)
      {
        T* new_items {new T[2 * capacity_]};

        // Possible implementation of std::copy is similar.
        for (size_t index {0}; index < length_; ++index)
        {
          new_items[index] = items_[index];
        }

        delete[] items_;

        items_ = new_items;

        capacity_ = 2 * capacity_;
      }
    }

    T* items_;
    std::size_t length_;
    std::size_t capacity_;
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

//------------------------------------------------------------------------------
/// \name Max Consecutive Ones
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3238/
/// \brief Given a binary array, find the maximum number of consecutive 1s in
/// this array.
//------------------------------------------------------------------------------
int find_max_consecutive_ones(std::vector<int>& nums);

//------------------------------------------------------------------------------
/// \name Find Numbers with Even Number of Digits
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3240/
/// \brief Given an array of integers A sorted in non-decreasing order, return
/// an array of squares of each number, also in sorted non-decreasing order.
//------------------------------------------------------------------------------
int find_even_length_numbers(std::vector<int>& nums);

//------------------------------------------------------------------------------
/// \name Squares of a Sorted Array
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3237/
/// \brief Given an array nums of integers, return how many of them contain an
/// even number of digits.
//------------------------------------------------------------------------------
std::vector<int> sorted_squares(std::vector<int>& A);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/squares-of-a-sorted-array/solution/
/// \brief Two pointer approach.
/// \details Strategy: iterate over negative part in reverse, and positive part
/// in forward direction. Then build a merged array from two sorted arrays by
/// comparing element by element.
//------------------------------------------------------------------------------
std::vector<int> sorted_squares_two_ptrs(std::vector<int>& A);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3245/
/// \brief Duplicate Zeros.
/// \details Given fixed length array arr of integers, duplicate each occurrence
/// of zero, shifting remaining elements to right.
//------------------------------------------------------------------------------
void duplicate_zeros(std::vector<int>& arr);

//------------------------------------------------------------------------------
/// \date Oct 30, 03:30
//------------------------------------------------------------------------------
void duplicate_zeros_linear_time(std::vector<int>& arr);


void duplicate_zeros_with_shift(std::vector<int>& arr);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3253/
/// \brief Merge Sorted Array.
/// \details Given 2 sorted integer arrays nums1 and nums2, merge nums2 into
/// nums2 as 1 sorted array.
/// \date Oct 30, 07:46
//------------------------------------------------------------------------------
void merge_sorted_arrays(
  std::vector<int>& nums1,
  int m,
  std::vector<int>& nums2,
  int n);

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/526/deleting-items-from-an-array/3247/
/// \brief Remove Element.
//------------------------------------------------------------------------------
int remove_element(std::vector<int>& nums, int val);

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

//------------------------------------------------------------------------------
/// \details Hint: On each step of loop, check if we've seen a double of the
/// element, or half the element.
//------------------------------------------------------------------------------

bool check_if_double_exists(const std::vector<int>& arr);

//------------------------------------------------------------------------------
/// \details Important point:
/// Linear search.
///
/// \ref Array 101, Searching for items in an Array.
/// Check If N and Its Double Exist. LeetCode.
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/527/searching-for-items-in-an-array/3250/
//------------------------------------------------------------------------------
class CheckIfDoubleExists
{
  public:

    CheckIfDoubleExists();

    //--------------------------------------------------------------------------
    /// \details Fastest - do a double for loop, for loop over i and over j;
    /// check all possible pairs for a double.
    //--------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/fun-with-arrays/526/deleting-items-from-an-array/3248/
/// \brief Remove Duplicates from Sorted Array
//------------------------------------------------------------------------------

int remove_duplicates(std::vector<int>& nums);

} // namespace LeetCode

} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_H