#ifndef ALGORITHMS_PRE_EASY_EXERCISES_H
#define ALGORITHMS_PRE_EASY_EXERCISES_H

#include <concepts> // std::integral
#include <iostream>
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


};

} // namespace PreEasyExercises
} // namespace Algorithms
#endif // ALGORITHMS_PRE_EASY_EXERCISES_H