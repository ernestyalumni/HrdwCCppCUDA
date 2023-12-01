#ifndef ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H
#define ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H

#include <cstddef>
#include <string>
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

    // O(N) time complexity.
    static bool is_palindrome(const std::string& s);
};

//------------------------------------------------------------------------------
/// \name 547. Number of Provinces
/// \url https://leetcode.com/problems/number-of-provinces/
//------------------------------------------------------------------------------
int find_number_of_provinces(std::vector<std::vector<int>>& is_connected);

} // namespace LeetCode
} // namespace Algorithms

#endif // ALGORITHMS_LEETCODE_MEDIUM_PROBLEMS_H