//------------------------------------------------------------------------------
/// \file Recursion.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating recursion.
/// \ref https://stackoverflow.com/questions/22504837/how-to-implement-quick-sort-algorithm-in-c
///-----------------------------------------------------------------------------
#include "Recursion.h"

#include <array>
#include <cstddef> // std::size_t
#include <numeric> // std::accumulate
#include <string>
#include <utility> // std::swap
#include <vector>

using std::accumulate;
using std::array;
using std::size_t;
using std::string;
using std::swap; // also with algorithm
using std::vector;


namespace Algorithms
{
namespace Recursion
{

namespace HackerRank
{

namespace DavisStaircases
{

int recursive_step_permutations(const int n)
{
  // Notice that step_permutations takes in input the number of steps in a
  // staircase. It returns the integer number of ways Davis can climb the
  // staircase, modulo 10000000007.
  // Likewise, for step_permutations(n - 1), it takes n - 1 as the number of
  // steps in a staircase and returns number of ways to climb n - 1 steps in the
  // staircase.

  switch(n)
  {
    case 0:
      return 0;
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return 4;
  }

  int number_of_ways {0};

  // step_permutation(n - 1) is the number of ways to climb n - 1 steps.
  // Imagine taking 1 step. Then you'll have n -1 steps to deal with.
  for (int initial_step {1}; initial_step < 4; ++initial_step)
  {
    number_of_ways += recursive_step_permutations(n - initial_step);  
  }

  return number_of_ways % 10000000007;
}

int cached_step_permutations(const int n)
{
  switch(n)
  {
    case 0:
      return 0;
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return 4;
  }

  array<int, 3> previous_number_of_ways {1, 2, 4};

  // Consider n steps in a staircase.
  // Only 3 ways to take steps to get to n steps: 1 step away, 2 steps away,
  // 3 steps away. -> consider ways to n - 1 steps, n - 2 steps, n - 3 steps.
  //
  // Only 3 ways to get to n - 1 steps: 1 step away, 2 steps away, etc.
  // Consider ways to n - 2, n - 3, n - 4.
  // ...
  // Only 3 ways to get to 4 steps: 1 step away, 2 steps away, 3 steps away.
  // Consider ways to 1 step, 2 steps, 3 steps.

  for (int initial_steps {3}; initial_steps < n; ++initial_steps)
  {

    // number of ways to step to
    // initial_steps - 2, initial_steps - 1, initial_steps
    // sum them all to get the number of ways to step to initial_steps + 1.
    const int number_of_ways {
      accumulate(
        previous_number_of_ways.begin(),
        previous_number_of_ways.end(),
        static_cast<int>(0))};

    // Move over number of ways to get to initial steps - 1, initial steps to
    // the "left".
    swap(previous_number_of_ways[0], previous_number_of_ways[1]);
    swap(previous_number_of_ways[1], previous_number_of_ways[2]);
    previous_number_of_ways[2] = number_of_ways;

    // We can loop again because the number of steps allowed to get to the
    // "final destination" of steps (say mth step) is still (and this is the
    // important point, it's constant) 1 step, 2 steps, or 3 steps away.
  }

  return previous_number_of_ways[2] % 10000000007;
}

} // namespace DavisStaircases

namespace CrosswordPuzzle
{

vector<string> split_string(string& s, const string& delimiter)
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

bool crosswordPuzzle(vector<string>& crossword, vector<string>& words)
{
  constexpr size_t grid_dimension {10};

  auto try_place =
    [&](auto i, auto j, auto r)
    {
      // Returns reference to last element in container.
      auto word = words.back();
      
      if ((r ? j : i) + word.length() > grid_dimension)
      {
        return false;
      }

      for (size_t k {0}; k < word.length(); ++k)
      {
        if (crossword[r ? i : i + k][r ? j + k : j] != '-' &&
          crossword[r ? i : i + k][r ? j + k:j] != word[k])
        {
          return false;
        }
      }                
      
      auto crossword_copy = crossword;
      for (size_t k {0}; k < word.length(); ++k)
      {
        crossword[r ? i : i + k][r ? j + k : j] = word[k];
      }
      
      words.pop_back();
      bool success {crosswordPuzzle(crossword, words)};

      // backtracking, recursion technique to search other directions.
      words.push_back(word);
      if (!success)
      {
        crossword = crossword_copy;
      };
      
      return success;
    };

  if (words.size() == 0)
  {
    return true;
  }

  for (size_t i {0}; i < grid_dimension; ++i)
  {
    for (size_t j {0}; j < grid_dimension; ++j)
    {
      if (try_place(i, j, 1/*by row*/))
      {
        return true;
      }
      
      if (try_place(i, j, 0/*by col*/))
      {
        return true;
      }
    }
  }

  return false;
};

vector<string> crosswordPuzzle(vector<string> crossword, string words)
{
  vector<string> word_vec;

  /*
  for (size_t i {0}, j {0}; i < words.length(); ++i)
  {
    if (words[i] == ';')
    {
            word_vec.push_back(words.substr(j, i-j));
            j = i+1;
        } else if (i == words.length()-1)
            word_vec.push_back(words.substr(j, i-j+1));
    }*/

  const string delimiter {";"};
  word_vec = split_string(words, delimiter);

  crosswordPuzzle(crossword, word_vec);
  return crossword;
}

} // namespace CrosswordPuzzle

} // namespace HackerRank

} // namespace Recursion
} // namespace Algorithms
