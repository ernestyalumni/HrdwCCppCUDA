#ifndef ALGORITHMS_PRE_EASY_EXERCISES_H
#define ALGORITHMS_PRE_EASY_EXERCISES_H

#include <concepts> // std::integral
#include <cstdint>
#include <iostream>
#include <iterator> // std::next, std::prev
#include <limits> // std::numeric_limits
#include <tuple>
#include <unordered_map>
#include <vector>

namespace Algorithms
{

//------------------------------------------------------------------------------
/// https://en.cppreference.com/w/cpp/concepts/integral.html
/// The concept integral<T> is satisfied if and only if T is an integral type.
/// O(sqrt(n)) time.
//------------------------------------------------------------------------------
template <typename T>
bool is_prime(const T n) requires std::integral<T>
{

  if (n <= 1)
  {
    return false;
  }

  if (n <= 3)
  {
    return true;
  }

  // Check divisibility by all numbers from 2 to the square root of n.
  for (T i {2}; i * i <= n; ++i)
  {
    if (n % i == 0)
    {
      return false;
    }
  }

  return true;
}

//------------------------------------------------------------------------------
/// O(sqrt(n)) time. O(n) space.
//------------------------------------------------------------------------------
template <typename T>
bool is_prime_with_memoization(const T n) requires std::integral<T>
{
  static std::unordered_map<T, bool> memo {};
  if (memo.find(n) != memo.end())
  {
    return memo[n];
  }

  if (is_prime(n))
  {
    memo[n] = true;
    return true;
  }
  else
  {
    memo[n] = false;
    return false;
  }
}

namespace PreEasyExercises
{

//------------------------------------------------------------------------------
/// https://blog.faangshui.com/p/before-leetcode
/// 1. Array Indexing
/// Understanding how to navigate arrays is essential. Here are ten exercises,
//  sorted in increasing difficulty, that build upon each other:
//------------------------------------------------------------------------------
class ArrayIndexing
{
  public:

    //--------------------------------------------------------------------------
    /// 1. Iterate Over an Array
    /// Write a function that prints each element in an array in order from the
    /// first to the last.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static auto iterate_over_array(
      const Container& array,
      const bool is_print = false)
    {
      Container result {};
      for (const auto& element : array)
      {
        result.push_back(element);
        if (is_print)
        {
          std::cout << element << ' ';
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /// 2. Iterate Over an Array in Reverse 
    /// Modify the previous function to print the elements in reverse order,
    /// from the last to the first. 
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static auto iterate_over_array_in_reverse(
      const Container& array,
      const bool is_print = false)
    {
      Container result {};
      for (
        auto it = std::ranges::rbegin(array);
        it != std::ranges::rend(array);
        ++it)
      {
        const auto& element = *it;
        result.push_back(element);
        if (is_print)
        {
          std::cout << element << ' ';
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /// 3. Fetch Every Second Element
    /// Write a function that accesses every other element in the array,
    /// starting from the first element.
    /// O(n/2) time, for visiting every second element.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static auto fetch_second_element(
      const Container& array,
      const bool is_print=false)
    {
      Container result {};

      auto iter = std::ranges::begin(array);
      const auto end = std::ranges::end(array);
      while (iter != end)
      {
        result.push_back(*iter);
        if (is_print)
        {
          std::cout << *iter << ' ';
        }

        // https://en.cppreference.com/w/cpp/iterator/ranges/next
        // constexpr I next(I i, std::iter_difference_t<I> n, S bound);
        // i - an iterator
        // n - number of elements to advance
        // bound - sentinel denoting end of range i points to
        iter = std::ranges::next(iter, 2, end);  
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /// 4. Find the Index of a Target Element
    /// Write a function that searches for a specific element in an array and
    /// returns its index. If the element is not found, return -1.
    //--------------------------------------------------------------------------
    template <std::ranges::input_range Container, typename T>
    static int find_index_of_target_element(
      const Container& array,
      const T target,
      const bool is_print=false)
    {
      std::size_t index {0};
      for (const auto& element : array)
      {
        if (element == target)
        {
          return static_cast<int>(index);
        }
        ++index;
      }
      return -1;
    }

    //--------------------------------------------------------------------------
    /// 5. Find the First Prime Number in an Array
    /// Iterate over an array and find the first prime number. Stop the
    /// iteration once you find it.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::ranges::range_value_t<Container> find_first_prime_number(
      const Container& array) requires std::integral<
        std::ranges::range_value_t<Container>>
    {
      for (const auto element : array)
      {
        if (is_prime(element))
        {
          return element;
        }
      }
      return -1;
    }

    //--------------------------------------------------------------------------
    /// 6. Traverse a Two-Dimensional Array
    ///
    /// Write a function to print all elements of a 2D array (matrix), row by
    /// row.
    //--------------------------------------------------------------------------
    template <std::ranges::input_range OuterContainer>
    static std::vector<
      std::ranges::range_value_t<
        std::ranges::range_value_t<
          OuterContainer>>> traverse_two_dimensional_array(
      const OuterContainer& array,
      const bool is_print = false)
    {
      using ElementType = std::ranges::range_value_t<
        std::ranges::range_value_t<OuterContainer>>;
      std::vector<ElementType> result {};

      for (const auto& row : array)
      {
        for (const auto element : row)
        {
          result.push_back(element);
          if (is_print)
          {
            std::cout << element << ' ';
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /// 7. Traverse the Main Diagonal of a Matrix
    /// Print the elements along the main diagonal of a square matrix, where the
    /// row and column indices are equal.
    //--------------------------------------------------------------------------
    template <std::ranges::input_range OuterContainer>
    static std::vector<
      std::ranges::range_value_t<
        std::ranges::range_value_t<OuterContainer>>> traverse_main_diagonal(
        const OuterContainer& array,
        const bool is_print = false)
    {
      using ElementType = std::ranges::range_value_t<
        std::ranges::range_value_t<OuterContainer>>;
      std::vector<ElementType> result {};

      const auto M = std::ranges::size(array);

      if (M == 0)
      {
        return result;
      }
      const auto N = std::ranges::size(array[0]);
      for (std::size_t i {0}; i < std::min(M, N); ++i)
      {
        result.push_back(array[i][i]);
        if (is_print)
        {
          std::cout << array[i][i] << ' ';
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /// 8. Traverse the Perimeter of a Matrix
    /// Print the elements along the outer edge (perimeter) of a 2D array.
    ///
    /// std::ranges::range_value_t<OuterContainer>> extracts value type of
    /// OuterContainer. So e.g. for vector<vector<int>>, it extracts vector<int>
    /// requires std::ranges::random_access_range<
    ///   std::ranges::range_value_t<OuterContainer>>
    /// ensures that each row supports random access (e.g. array[i])
    /// For vector<vector<int>>, ranges::range_value_t<OuterContainer> is
    /// vector<int> and requires then checks random_access_range<vector<int>>,
    /// which is true.
    //--------------------------------------------------------------------------

    template <std::ranges::random_access_range OuterContainer>
      requires std::ranges::random_access_range<
        std::ranges::range_value_t<OuterContainer>>
    static std::vector<std::ranges::range_value_t<std::ranges::range_value_t<
      OuterContainer>>> traverse_perimeter(
      const OuterContainer& array,
      const bool is_print = false)
    {
      using ElementType = std::ranges::range_value_t<
        std::ranges::range_value_t<OuterContainer>>;
      std::vector<ElementType> result {};

      const int M {static_cast<int>(std::ranges::size(array))};
      if (M == 0)
      {
        return result;
      }

      for (const auto& element : array[0])
      {
        result.push_back(element);
        if (is_print)
        {
          std::cout << element << ' ';
        }
      }

      if (M == 1)
      {
        return result;
      }

      const int N {static_cast<int>(std::ranges::size(array[0]))};

      for (std::size_t i {1}; i < M; ++i)
      {
        result.push_back(array[i][N - 1]);
        if (is_print)
        {
          std::cout << array[i][N - 1] << ' ';
        }
      }

      if (N == 1)
      {
        return result;
      }

      for (int j {static_cast<int>(N - 2)}; j >= 0; --j)
      {
        result.push_back(array[M - 1][j]);
        if (is_print)
        {
          std::cout << array[M - 1][j] << ' ';
        }
      }

      for (int i {static_cast<int>(M - 2)}; i >= 1; --i)
      {
        result.push_back(array[i][0]);
        if (is_print)
        {
          std::cout << array[i][0] << ' ';
        }
      }

      return result;
    }

    //--------------------------------------------------------------------------
    /// 9. Traverse Elements in Spiral Order
    /// Print elements of a 2D array in spiral order, starting from the top-left
    /// corner and moving inward.
    //--------------------------------------------------------------------------
    template <std::ranges::random_access_range OuterContainer>
      requires std::ranges::random_access_range<
        std::ranges::range_value_t<OuterContainer>>
    static std::vector<std::ranges::range_value_t<std::ranges::range_value_t<
      OuterContainer>>> traverse_spiral_clockwise(
      const OuterContainer& array,
      const bool is_print = false)
    {
      using ElementType = std::ranges::range_value_t<
        std::ranges::range_value_t<OuterContainer>>;
      std::vector<ElementType> result {};

      const auto M = std::ranges::size(array);
      if (M == 0)
      {
        return result;
      }

      const auto N = std::ranges::size(array[0]);
      if (N == 0)
      {
        return result;
      }

      std::size_t m_up {0};
      std::size_t m_down {M - 1};
      std::size_t n_left {0};
      std::size_t n_right {N - 1};

      auto is_traverse_complete = 
        [](
          const std::size_t m_up,
          const std::size_t m_down,
          const std::size_t n_left,
          const std::size_t n_right) -> bool
      {
        return m_up > m_down || n_left > n_right;
      };

      while (!is_traverse_complete(m_up, m_down, n_left, n_right))
      {
        // Initially, we'll get all of the top row. And then in subsequent
        // traversals, we'll start from the "second" element in given row or
        // column.
        // Traverse top row.
        for (std::size_t j {n_left}; j <= n_right; ++j)
        {
          result.push_back(array[m_up][j]);
          if (is_print)
          {
            std::cout << array[m_up][j] << ' ';
          }
        }
        ++m_up;

        if (is_traverse_complete(m_up, m_down, n_left, n_right))
        {
          break;
        }

        // Traverse right column.
        for (std::size_t i {m_up}; i <= m_down; ++i)
        {
          result.push_back(array[i][n_right]);
          if (is_print)
          {
            std::cout << array[i][n_right] << ' ';
          }
        }
        --n_right;

        if (is_traverse_complete(m_up, m_down, n_left, n_right))
        {
          break;
        }

        // Traverse bottom row.
        // Shift by +1 on the j index since std::size_t is unsigned.
        for (std::size_t j {n_right + 1}; j >= n_left + 1; --j)
        {
          result.push_back(array[m_down][j - 1]);
          if (is_print)
          {
            std::cout << array[m_down][j - 1] << ' ';
          }
        }
        --m_down;

        if (is_traverse_complete(m_up, m_down, n_left, n_right))
        {
          break;
        }

        // Traverse left column.
        for (std::size_t i {m_down}; i >= m_up; --i)
        {
          result.push_back(array[i][n_left]);
          if (is_print)
          {
            std::cout << array[i][n_left] << ' ';
          }
        }

        ++n_left;
      }

      return result;
    }

    //--------------------------------------------------------------------------
    /// 10. Traverse the Lower Triangle of a Matrix
    /// Print the elements below and including the main diagonal of a square matrix.
    //--------------------------------------------------------------------------
    template <std::ranges::random_access_range OuterContainer>
      requires std::ranges::random_access_range<
        std::ranges::range_value_t<OuterContainer>>
    static std::vector<std::ranges::range_value_t<std::ranges::range_value_t<
      OuterContainer>>> traverse_lower_triangle(
      const OuterContainer& array,
      const bool is_print = false)
    {
      using ElementType = std::ranges::range_value_t<
        std::ranges::range_value_t<OuterContainer>>;
      std::vector<ElementType> result {};

      const auto M = std::ranges::size(array);
      if (M == 0)
      {
        return result;
      }

      const auto N = std::ranges::size(array[0]);
      if (N == 0)
      {
        return result;
      }

      for (std::size_t i {0}; i < M; ++i)
      {
        for (std::size_t j {0}; j <= i; ++j)  
        {
          result.push_back(array[i][j]);
          if (is_print)
          {
            std::cout << array[i][j] << ' ';
          }
        }
      }
      return result;
    }
};

//------------------------------------------------------------------------------
/// https://blog.faangshui.com/p/before-leetcode
/// 2. Accumulator Variables
/// Learn how to keep track of values during iteration. 
//------------------------------------------------------------------------------
class AccumulatorVariables
{
  public:

    //--------------------------------------------------------------------------
    /// 1. Calculate the Sum of an Array
    /// Write a function that calculates the sum of all elements in an array by
    /// accumulating the total as you iterate.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static auto calculate_sum(const Container& array)
      requires std::integral<std::ranges::range_value_t<Container>>
    {
      auto sum = 0;
      for (const auto& element : array)
      {
        sum += element;
      }

      return sum;
    }

    //--------------------------------------------------------------------------
    /// 2. Find the Minimum and Maximum Elements
    /// Find the smallest and largest numbers in an array by updating minimum
    /// and maximum variables during iteration.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::tuple<
     std::ranges::range_value_t<Container>,
     std::ranges::range_value_t<Container>> find_min_max(const Container& array)
    {
      using ElementType = std::ranges::range_value_t<Container>;
      auto min_value {std::numeric_limits<ElementType>::max()};
      auto max_value {std::numeric_limits<ElementType>::min()};

      for (const auto& element : array)
      {
        if (element < min_value)
        {
          min_value = element;
        }
        if (element > max_value)
        {
          max_value = element;
        }
      }
      return std::make_tuple(min_value, max_value);
    }

    //--------------------------------------------------------------------------
    /// 3. Find the Indices of the Min and Max Elements
    /// In addition to finding the min and max values, keep track of their
    /// positions (indices) in the array.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::tuple<int, int> find_min_max_indices(const Container& array)
    {
      using ElementType = std::ranges::range_value_t<Container>;
      auto min_value {std::numeric_limits<ElementType>::max()};
      auto max_value {std::numeric_limits<ElementType>::min()};
      int min_index {-1};
      int max_index {-1};

      int index {0};
      for (const auto& element : array)
      {
        if (element < min_value)
        {
          min_value = element;
          min_index = index;
        }
        if (element > max_value)
        {
          max_value = element;
          max_index = index;
        }
        ++index;
      }
      return std::make_tuple(min_index, max_index);
    }

    //--------------------------------------------------------------------------
    /// 4. Find the Two Smallest/Largest Elements Without Sorting
    /// Modify your approach to keep track of the two smallest and two largest
    /// elements during a single pass through the array.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::tuple<int, int, int, int> find_two_min_max_indices(
      const Container& array)
    {
      using ElementType = std::ranges::range_value_t<Container>;
      auto min_value {std::numeric_limits<ElementType>::max()};
      auto max_value {std::numeric_limits<ElementType>::min()};
      auto min_value_2 {std::numeric_limits<ElementType>::max()};
      auto max_value_2 {std::numeric_limits<ElementType>::min()};
      auto min_index {-1};
      auto max_index {-1};
      auto min_index_2 {-1};
      auto max_index_2 {-1};

      int index {0};
      for (const auto& element : array)
      {
        if (element < min_value)
        {
          min_value_2 = min_value;
          min_index_2 = min_index;
          min_value = element;
          min_index = index;
        }
        else if (element < min_value_2)
        {
          min_value_2 = element;
          min_index_2 = index;
        }

        if (element > max_value)
        {
          max_value_2 = max_value;
          max_index_2 = max_index;
          max_value = element;
          max_index = index;
        }
        else if (element > max_value_2)
        {
          max_value_2 = element;
          max_index_2 = index;
        }

        ++index;
      }
      return std::make_tuple(min_index, min_index_2, max_index, max_index_2);
    }

    //--------------------------------------------------------------------------
    /// 5. Count Occurrences of a Specific Element
    /// Count how many times a given element appears in the array by
    /// incrementing a counter whenever you encounter it.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static int count_occurrences(
      const Container& array,
      const typename std::ranges::range_value_t<Container> target)
    {
      int count {0};
      for (const auto& element : array)
      {
        if (element == target)
        {
          ++count;
        }
      }
      return count;
    }

    //--------------------------------------------------------------------------
    /// 6. Count Occurrences of All Elements
    /// Use a dictionary or map to count the number of times each unique element
    /// appears in the array during a single iteration.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::unordered_map<
      std::ranges::range_value_t<
        Container>,
        int> count_occurrences_of_all(const Container& array)
    {
      std::unordered_map<std::ranges::range_value_t<Container>, int> result {};
      for (const auto& element : array)
      {
        result[element]++;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /// 7. Find the Two Most Frequent Elements
    /// Find the two elements that appear the most number of times in an array.
    /// Do 2 passes: 1 to create a frequency map, and 2 to find the two most
    /// frequent.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::tuple<
      std::ranges::range_value_t<Container>,
      std::ranges::range_value_t<Container>> find_two_most_frequent(
        const Container& array)
    {
      using ElementType = std::ranges::range_value_t<Container>;

      std::unordered_map<ElementType, int> element_to_count {};

      for (const auto& element : array)
      {
        element_to_count[element]++;
      }

      int max_count {-1};
      int second_max_count {-1};
      ElementType max_value {};
      ElementType second_max_value {};

      for (const auto& [element, count] : element_to_count)
      {
        if (count > max_count)
        {
          second_max_count = max_count;
          second_max_value = max_value;
          max_count = count;
          max_value = element;
        }
        else if (count > second_max_count)
        {
          second_max_count = count;
          second_max_value = element;
        }
      }

      // Handle case where there's only one unique element. This occurs because
      // as you iterate in the for loop through each element, count in
      // element_to_count, if there's only 1 unique element, there's only 1
      // element, count pair to iterate on and the else if statement is never
      // called.
      if (second_max_count == -1)
      {
        second_max_value = max_value;
      }

      return std::make_tuple(max_value, second_max_value);
    }

    //--------------------------------------------------------------------------
    /// 8. Compute Prefix Sums
    /// Create an array where each element at index i is the sum of all elements
    /// up to that index in the original array. We call this array prefix sums
    /// array.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::vector<
      std::ranges::range_value_t<Container>> compute_prefix_sums(
      const Container& array)
    {
      using ElementType = std::ranges::range_value_t<Container>;
      std::vector<ElementType> result {};
      ElementType running_total {0};
      for (auto element : array)
      {
        running_total += element;
        result.push_back(running_total);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /// 9. Find the Sum of Elements in a Given Range
    /// Given a range (start and end indices), write a function that calculates
    /// the sum of elements within that range by iterating from the start to the
    /// end index and accumulating the sum.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::ranges::range_value_t<Container> find_sum_in_range(
      const Container& array,
      const std::size_t start_index,
      const std::size_t end_index)
    {
      using ElementType = std::ranges::range_value_t<Container>;
      const auto N = std::ranges::size(array);
      if (start_index >= N)
      {
        return ElementType {};
      }

      ElementType running_sum {0};
      for (std::size_t i {start_index}; i <= std::min(end_index, N - 1); ++i)
      {
        running_sum += array[i];
      }
      return running_sum;
    }

    //--------------------------------------------------------------------------
    /// 10. Efficient Range Sum Queries Using Prefix Sums
    /// After computing the prefix sums array, answer multiple range sum queries
    /// efficiently:
    /// Instead of summing elements for each query, use the prefix sums array to
    /// compute the sum of elements between indices i and j in constant time.
    ///
    /// Hint: The sum from index i to j can be calculated as prefix_sum[j] -
    /// prefix_sum[i - 1]. This method requires understanding how to manipulate
    /// indices and handle edge cases when i is 0.
    //--------------------------------------------------------------------------
    template <std::ranges::range Container>
    static std::ranges::range_value_t<Container> efficient_range_sum_queries(
      const Container& array,
      const std::size_t start_index,
      const std::size_t end_index)
    {
      using ElementType = std::ranges::range_value_t<Container>;
      const auto N = std::ranges::size(array);
      if (start_index >= N)
      {
        return ElementType {};
      };

      std::vector<ElementType> prefix_sums {};
      ElementType prefix_sum_running_total {};
      for (const auto element : array)
      {
        prefix_sum_running_total += element;
        prefix_sums.push_back(prefix_sum_running_total);
      }

      if (start_index == 0)
      {
        return prefix_sums[end_index];
      }
      else
      {
        return prefix_sums[end_index] - prefix_sums[start_index - 1];
      }
    }
};

//------------------------------------------------------------------------------
/// https://blog.faangshui.com/p/before-leetcode
/// 3. Recursion
/// Get comfortable with functions that call themselves
//------------------------------------------------------------------------------

class Recursion
{
  public:

    //--------------------------------------------------------------------------
    /// 1. Calculate the nth Fibonacci Number
    /// Write a recursive function to find the nth Fibonacci number, where each
    /// number is the sum of the two preceding ones.     
    ///
    /// 12.1.6 constexpr Functions Ch. 12 Functions; Bjarne Stroustrup, The C++
    /// Programming Language, 4th Ed., Stroustrup
    ///
    /// 3 "laws" of recursion:
    /// 1. base case
    /// 2. must change its state and move toward base case
    /// 3. must call itself, recursively.
    ///
    /// Fibonacci numbers (sequence) as recurrence relation: 
    /// F_n = F_{n-1} + F_{n-2}
    /// F_1 = 1, F_0 = 1 (base cases)
    ///
    /// since constexpr function, cannot have branching (i.e. if, elses)
    //--------------------------------------------------------------------------
    template <typename T>
    static constexpr T calculate_fibonacci_number(const T n)
    {
      return (n <= static_cast<T>(2)) ?
        static_cast<T>(1) :
          (calculate_fibonacci_number<T>(n - static_cast<T>(1)) +
            calculate_fibonacci_number<T>(n - static_cast<T>(2)));
    }

    //--------------------------------------------------------------------------
    /// 2. Sum of an Array Using Recursion
    /// Find the sum of all elements in an array by recursively summing the
    /// first element and the sum of the rest of the array.
    /// Time: O(n), Space: O(n) due to recursion stack
    //--------------------------------------------------------------------------

    template <typename Iterator>
    static auto sum_recursive_helper(Iterator begin_it, Iterator end_it)
    {
      if (begin_it == end_it)
      {
        return 0;
      }
      return *begin_it + sum_recursive_helper(std::next(begin_it), end_it);
    }

    template <std::ranges::range Container>
    static auto sum_recursive(const Container& array)
    {
      return sum_recursive_helper(
        std::ranges::begin(array),
        std::ranges::end(array));
    }

    //--------------------------------------------------------------------------
    /// 3. Find the Minimum Element in an Array Using Recursion 
    /// Find the smallest element in an array without using loops by comparing
    /// the first element with the minimum of the rest of the array.
    //--------------------------------------------------------------------------

    template <typename Iterator, typename T>
    static auto min_helper(Iterator begin_it, Iterator end_it, T min_value)
    {
      if (begin_it == end_it)
      {
        return min_value;
      }

      const auto current_value = *begin_it;
      if (current_value < min_value)
      {
        min_value = current_value;
      }
      return min_helper(std::next(begin_it), end_it, min_value);
    }

    template <std::ranges::range Container>
    static auto find_min(const Container& array)
    {
      return min_helper(
        std::ranges::begin(array),
        std::ranges::end(array),
        std::numeric_limits<std::ranges::range_value_t<Container>>::max());
    }
    
    //--------------------------------------------------------------------------
    /// 4. Reverse a String Using Recursion
    /// Reverse a given string by recursively swapping characters from the ends
    /// towards the center.
    //--------------------------------------------------------------------------

    template <typename Iterator>
    static void reverse_string_helper(Iterator begin_it, Iterator end_it)
    {
      if (begin_it == end_it || std::distance(begin_it, end_it) < 0)
      {
        return;
      }
      std::swap(*begin_it, *end_it);
      reverse_string_helper(std::next(begin_it), std::prev(end_it));
    }

    template <std::ranges::range Container>
    static void reverse_string(Container& array)
    {
      reverse_string_helper(
        std::ranges::begin(array),
        std::prev(std::ranges::end(array)));
    }

    //--------------------------------------------------------------------------
    /// 5. Check if a String is a Palindrome Using Recursion
    /// Determine if a string reads the same backward as forward by comparing
    /// characters from the outside in.
    //--------------------------------------------------------------------------

    template <typename Iterator>
    static bool is_palindrome_helper(
      Iterator begin_it,
      Iterator end_it,
      bool is_palindrome = true)
    {
      if (begin_it == end_it || std::distance(begin_it, end_it) < 0)
      {
        return is_palindrome;
      }

      if (*begin_it != *end_it)
      {
        is_palindrome = false;
      }
      return is_palindrome_helper(
        std::next(begin_it),
        std::prev(end_it),
        is_palindrome);
    }

    template <std::ranges::range Container>
    static bool is_palindrome(const Container& array)
    {
      return is_palindrome_helper(
        std::ranges::begin(array),
        std::prev(std::ranges::end(array)),
        true);
    }

    //--------------------------------------------------------------------------
    /// 6. Generate All Permutations of a String
    /// Recursively generate all permutations of the characters in a string by
    /// swapping characters.
    /// Time: O(n! * n), Space: O(n) for recursion stack
    /// n! permutations, and n swaps to build each permutation.
    //--------------------------------------------------------------------------

  private:

    template <typename Iterator, typename Container>
    static void generate_all_permutations_helper(
      Iterator begin_iter,
      Iterator end_iter,
      Container& mutable_copy,
      std::vector<Container>& permutations)
    {
      // Base case, we've processed all the characters for this permutation and
      // so we add it.
      if (begin_iter == end_iter)
      {
        permutations.push_back(mutable_copy);
        return;
      }

      // Try each character at the current position.
      for (auto iter = begin_iter; iter != end_iter; ++iter)
      {
        // Fix character at current position.
        std::swap(*begin_iter, *iter);

        // Recursively generate permutations for remaining characters.
        generate_all_permutations_helper(
          std::next(begin_iter),
          end_iter,
          mutable_copy,
          permutations);

        // Backtrack: restore original order.
        std::swap(*begin_iter, *iter);
      }
    }

  public:

    template <std::ranges::range Container>
    static std::vector<Container> generate_all_permutations(
      Container& container)
    {
      std::vector<Container> permutations {};

      if (std::ranges::empty(container))
      {
        return permutations;
      }

      // Create mutable copy for swapping
      Container mutable_copy(
        std::ranges::begin(container),
        std::ranges::end(container));

      generate_all_permutations_helper(
        std::ranges::begin(mutable_copy),
        std::ranges::end(mutable_copy),
        mutable_copy,
        permutations);
      return permutations;
    }

    //--------------------------------------------------------------------------
    /// 7. Generate All Subsets of a Set
    ///
    /// Generate all possible subsets (the power set) of a set of numbers by
    /// including or excluding each element recursively.
    //--------------------------------------------------------------------------

  private:
    
    static void generate_all_subsets_helper_with_bitfield(
      std::size_t begin_index,
      std::size_t end_index,
      uint64_t subset_bitfield,
      std::vector<uint64_t>& subsets)
    {
      // base case
      if (begin_index > end_index)
      {
        subsets.push_back(subset_bitfield);
        return;
      }

      // Option 1: Don't include element at index (bit stays 0)
      generate_all_subsets_helper_with_bitfield(
        begin_index + 1,
        end_index,
        subset_bitfield,
        subsets);

      // Option 2: Include element at index (set bit to 1)
      subset_bitfield |= (1ULL << begin_index);
      generate_all_subsets_helper_with_bitfield(
        begin_index + 1,
        end_index,
        subset_bitfield,
        subsets);      

    }

  public:

    template <typename Set>
    static std::vector<Set> generate_all_subsets_with_bitfield(const Set& set)
    {
      std::vector<Set> power_set {};
      const auto N = set.size();

      if (N == 0)
      {
        power_set.push_back(Set {});
        return power_set;
      }

      if (N > 64)
      {
        // Bitfield approach limited to 64 bits.
        return power_set;
      }

      std::vector<uint64_t> power_set_bitfields {};
      uint64_t subset_bitfield {0};
      generate_all_subsets_helper_with_bitfield(
        0,
        N - 1,
        subset_bitfield,
        power_set_bitfields);

      std::vector<std::ranges::range_value_t<Set>> set_list (
        std::ranges::begin(set),
        std::ranges::end(set));

      for (const auto bitfield : power_set_bitfields)
      {
        Set subset {};
        for (std::size_t i {0}; i < N; ++i)
        {
          if (bitfield & (1 << i))
          {
            subset.insert(set_list[i]);
          }
        }
        power_set.push_back(subset);
      }
      return power_set;
    }

  private:
    // Recursive helper: build subsets by including/excluding each element
    template <typename Set, typename Iterator>
    static void generate_all_subsets_helper(
      Iterator current,
      Iterator end,
      // Current subset being built
      Set& current_subset,
      // All subsets collected so far
      std::vector<Set>& power_set)
    {
      // Base case: processed all elements, add current subset
      // Recursion depth: O(N), for N elements in the set.
      if (current == end)
      {
        power_set.push_back(current_subset);
        return;
      }

      // For each element, I havet 2 choices, include or exclude it.
      // Recursively build subsets by making these choices.

      // Option 1: Don't include current element
      generate_all_subsets_helper<Set>(
        std::next(current),
        end,
        current_subset,
        power_set);

      // Option 2: Include current element
      current_subset.insert(*current);
      generate_all_subsets_helper<Set>(
        std::next(current),
        end,
        current_subset,
        power_set);

      // Backtrack: remove current element to restore state
      current_subset.erase(*current);
    }

  public:

    //--------------------------------------------------------------------------
    /// Generate All Subsets of a Set
    /// Generate all possible subsets (the power set) of a set by including or
    /// excluding each element recursively.
    /// Time: O(2^n * n), Space: O(2^n * n) for storing all subsets
    //--------------------------------------------------------------------------
    template <typename Set>
    static std::vector<Set> generate_all_subsets(const Set& set)
    {
      std::vector<Set> power_set {};

      if (std::ranges::empty(set))
      {
        power_set.push_back(Set{});  // Empty set is a subset
        return power_set;
      }

      // Convert set to vector for indexed iteration
      using ElementType = std::ranges::range_value_t<Set>;
      std::vector<ElementType> elements(
        std::ranges::begin(set),
        std::ranges::end(set));

      // Build subsets recursively
      Set current_subset {};
      generate_all_subsets_helper<Set>(
        elements.cbegin(),  // Use const iterators
        elements.cend(),
        current_subset,
        power_set);

      return power_set;
    }

    template <typename Set>
    static std::vector<Set> generate_all_subsets_iteratively(const Set& set)
    {
      std::vector<Set> power_set {};
      // Start with empty subset
      power_set.push_back(Set{});

      // For each element, double the number of subsets by adding it to existing
      // ones
      for (const auto& element : set)
      {
        const std::size_t current_size {power_set.size()};
        for (std::size_t i {0}; i < current_size; ++i)
        {
          // Copy existing subset.
          Set new_subset {power_set[i]};  
          // Add current element.
          new_subset.insert(element);     
          power_set.push_back(new_subset);
        }
      }

      return power_set;
    }

    template <typename Set>
    static std::vector<Set> generate_all_subsets_recursively(const Set& set)
    {
      std::vector<Set> power_set {};
      using ElementType = std::ranges::range_value_t<Set>;
      if (std::ranges::empty(set))
      {
        power_set.push_back(set);
        return power_set;
      }

      auto begin_iterator = std::ranges::begin(set);
      auto end_iterator = std::ranges::end(set);
      auto next_iterator = std::next(begin_iterator);

      Set remaining_set(next_iterator, end_iterator);
      auto other_subsets = generate_all_subsets_recursively(remaining_set);
      power_set = other_subsets;
      for (const auto& subset : other_subsets)
      {
        Set new_subset {subset};
        new_subset.insert(*begin_iterator);
        power_set.push_back(new_subset);
      }

      return power_set;
    }
};

} // namespace PreEasyExercises
} // namespace Algorithms
#endif // ALGORITHMS_PRE_EASY_EXERCISES_H