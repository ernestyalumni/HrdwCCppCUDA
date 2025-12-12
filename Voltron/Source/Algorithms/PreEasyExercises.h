#ifndef ALGORITHMS_PRE_EASY_EXERCISES_H
#define ALGORITHMS_PRE_EASY_EXERCISES_H

#include <concepts> // std::integral
#include <iostream>
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
    //--------------------------------------------------------------------------
    template <typename Container, typename T>
    static std::vector<T> traverse_perimeter(
      const Container& array,
      const bool is_print = false)
    {
      std::vector<T> result {};
      const int M {static_cast<int>(std::ranges::size(array))};
      if (M == 0)
      {
        return result;
      }

      const int N {static_cast<int>(std::ranges::size(array[0]))};
      int m_up {0};
      int m_down {M - 1};
      int n_left {0};
      int n_right {N - 1};

      auto is_traverse_complete =
        [](int m_up, int m_down, int n_left, int n_right) -> bool
        {
          return m_up > m_down || n_left > n_right;
        };

      while (!is_traverse_complete(m_up, m_down, n_left, n_right))
      {
        // Traverse top row, stopping before "right" column.
        for (int j {n_left}; j < n_right; ++j)
        {
          result.push_back(array[m_up][j]);
          if (is_print)
          {
            std::cout << array[m_up][j] << ' ';
          }
        }
        ++m_up;

        // Traverse right column, stopping before "down" row.
        for (int i {m_up}; i < m_down; ++i)
        {
          result.push_back(array[i][n_right]);
          if (is_print)
          {
            std::cout << array[i][n_right] << ' ';
          }
        }
      }

    }

};

} // namespace PreEasyExercises
} // namespace Algorithms

#endif // ALGORITHMS_PRE_EASY_EXERCISES_H