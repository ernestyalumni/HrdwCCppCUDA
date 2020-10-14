//------------------------------------------------------------------------------
/// \file LeetCodePlayground.cpp
/// \author
/// \brief .
/// \ref 
/// 2020/10/14 01:21 Start.
///-----------------------------------------------------------------------------

// https://leetcode.com/problems/reorder-data-in-log-files/
#include <cstddef> // std::size_t
#include <iostream>
#include <iterator> // std::distance
#include <string>
#include <unordered_set>
#include <vector>

using std::distance;
//using std::stack;
using std::string;
using std::vector;

const string example_input_1 {"hello"};
const string example_input_2 {"leetcode"};

const string example_output_1 {"holle"};
const string example_output_2 {"leotcede"};

const std::unordered_set<char> vowels {'a', 'e', 'i', "o", "u"};

vector<std::size_t> get_vowel_indices(const string& s)
{
  vector<std::size_t> vowel_indices;

  for (auto iter {s.begin()}; iter != s.end(); ++iter)
  {
    // Found letter in string to be a vowel.
    if (vowels.find(*iter) != vowels.end())
    {
      vowel_indices.emplace_back(distance(s.begin(), iter));
    }
  } 

  return vowel_indices;
}

swap_vowels(string& s, vector<size_t> vowel_indices)


class Solution
{
  public:

    string reverseVowels(string s)
    {
      return s;
    }
};

int main()
{

}