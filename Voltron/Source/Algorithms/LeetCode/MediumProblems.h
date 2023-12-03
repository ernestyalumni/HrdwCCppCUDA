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