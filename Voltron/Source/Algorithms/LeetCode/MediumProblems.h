#ifndef ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H
#define ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// \name 3. Longest Substring Without Repeating Characters
/// Constraints:
/// 0 <= s.length <= 5 * 10^4
/// s consists of English letters, digits, symbols, and spaces
//------------------------------------------------------------------------------
class LongestSubstringWithoutRepeating
{
  public:

    //--------------------------------------------------------------------------
    /// A substring is a contiguous block of characters so a sliding window can
    /// keep track of a set of contiguous elements and can update the state of
    /// unique characters as we move along the string.
    //--------------------------------------------------------------------------
    static int length_of_longest_substring(std::string s);
};

//------------------------------------------------------------------------------
/// \name 5. Longest Palindromic Substring
/// \url https://leetcode.com/problems/longest-palindromic-substring/
/// \brief Given a string s, return the longest palindromic substring in s.
//------------------------------------------------------------------------------
class LongestPalindrome
{
  public:

    //--------------------------------------------------------------------------
    /// https://leetcode.com/problems/longest-palindromic-substring/solutions/4212564/beats-96-49-5-different-approaches-brute-force-eac-dp-ma-recursion/
    /// \details Brute force solution is to check every substring. Pick every
    /// possible starting position. For each starting position, pick every
    /// possible end position.
    /// N^2 possible substrings. Verifying each substring takes O(N) time.
    /// O(N^3) time complexity.
    //--------------------------------------------------------------------------
    static std::string brute_force(std::string s);

    // O(N) time complexity.
    static bool is_palindrome(const std::string& s);

    static std::string expand_around_center(std::string s);

    // O(N) time complexity.
    // Assumes that the letters that are included within given inputs left, and
    // right are a palindrome already.
    static std::tuple<int, int> expand_from_center(
      const std::string& s,
      const int left,
      const int right);

    static std::string with_dynamic_programming(std::string s);
    
    static std::string longest_palindrome(std::string s);

    static void find_longest_palindrome(
      int start_index,
      int current_length,
      int& maximum_length,
      int& longest_palindrome_index,
      const std::string& s,
      const int N);

    static inline bool is_outer_letters_palindrome(
      const std::size_t start_index,
      const std::size_t current_length,
      const std::string& s)
    {
      return s[start_index] == s[start_index + current_length - 1];
    }
};

//------------------------------------------------------------------------------
/// \name 11. Container With Most Water
/// \details Greedy (tag)
/// \url https://leetcode.com/problems/container-with-most-water/
/// Constraints:
/// 2 <= n <= 10^5 where n == height.length.
/// 0 <= height[i] <= 10^4
//------------------------------------------------------------------------------
class ContainerWithMostWater
{
  public:

    // O(N^2) time complexity (loop through all possible intervals).
    static int brute_force(std::vector<int>& height);

    // Greedy Algorithm.
    int maximum_area(std::vector<int>& height);
};

//------------------------------------------------------------------------------
/// \name 209. Minimum Size Subarray Sum
/// We're given an array of positive integers (that are non-zero).
/// A subarray is a contiguous non-empty sequence of elements within an array.
/// \url https://leetcode.com/problems/minimum-size-subarray-sum/
/// Constraints:
/// 1 <= target <= 10^9
/// 1 <= nums.length <= 10^5
/// 1 <= nums[i] <= 10^4
//------------------------------------------------------------------------------
class MinimumSizeSubarraySum
{
  public:

    /// @brief 
    /// @param target 
    /// @param nums 
    /// Sliding window technique.
    /// @return 
    static int minimum_subarray_length(int target, std::vector<int>& nums);
};

//------------------------------------------------------------------------------
/// \name 357. Count Numbers With Unique Digits
/// \url https://leetcode.com/problems/count-numbers-with-unique-digits/
/// Given integer n, return count of all numbers with unique digits, x, where
/// 0 <= x < 10^n.
//------------------------------------------------------------------------------
class CountNumbersWithUniqueDigits
{
  public:

    static int count_numbers_with_unique_digits(int n);
};

//------------------------------------------------------------------------------
/// \name 547. Number of Provinces
/// \url https://leetcode.com/problems/number-of-provinces/
//------------------------------------------------------------------------------
int find_number_of_provinces(std::vector<std::vector<int>>& is_connected);

//------------------------------------------------------------------------------
/// \name 2944. Minimum Number of Coins for Fruits
/// \url https://leetcode.com/problems/minimum-number-of-coins-for-fruits/
/// Dynamic programming.
/// \return minimum number of coins needed to acquire all the fruits.
/// Given a 1-indexed array prices, where prices[i] denotes number of coins
/// needed to purchase ith fruit.
/// If you purchase ith fruit at prices[i] coins, you can get next i fruits for
/// free.
/// Constraints:
/// 1 <= prices.length <= 1000
/// 1 <= prices[i] <= 10^5.
//------------------------------------------------------------------------------
class MinimumNumberOfCoinsForFruits
{
  public:

    // O(N^2) time complexity (iterate through all N fruits and for each fruit,
    // iterate through next i fruits).
    // O(N) space complexity to hold minimum_cost for each fruit of N fruits.
    static int minimum_coins(std::vector<int>& prices);
};

} // namespace LeetCode
} // namespace Algorithms

#endif // ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H