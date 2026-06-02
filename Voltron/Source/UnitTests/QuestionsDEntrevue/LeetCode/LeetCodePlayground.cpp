//------------------------------------------------------------------------------
/// \file LeetCodePlayground.cpp
/// \author
/// \brief .
/// \ref 
/// 2020/10/14 23:29 Start.
///-----------------------------------------------------------------------------

// https://leetcode.com/problems/reorder-data-in-log-files/
#include <array>
#include <cstddef> // std::size_t
#include <iostream>
#include <iterator> // std::distance
#include <numeric> // std::accumulate
#include <string>
#include <unordered_set>
#include <utility> // std::swap
#include <vector>

using std::accumulate;
using std::array;
using std::cout;
using std::distance;
//using std::stack;
using std::size_t;
using std::string;
using std::swap; // also with algorithm
using std::vector;

vector<string> split_string(string& s, string& delimiter)
{
  size_t position {0};
  vector<string> split_strings;

  // npos is a special value equal to maximum value, end of string indicator for
  // functions expecting a string index.
  // find characters in the string. Returns position of first character of found
  // substring or npos if no such substring is found.
  while ((position = s.find(delimiter)) != string::npos)
  {
    // returns a substring [pos, pos+count)
    split_strings.emplace_back(s.substr(0, position));

    // Removes count characeters starting at index, if erase(index, count).
    s.erase(0, position + delimiter.length());
  }
  split_strings.emplace_back(s);

  return split_strings;
}

int main()
{
  //cout << recursive_step_permutations(0) << "\n"; // 0

}