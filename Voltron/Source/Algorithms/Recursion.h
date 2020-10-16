//------------------------------------------------------------------------------
/// \file Recursion.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating recursion.
/// \ref 
///-----------------------------------------------------------------------------
#ifndef ALGORITHMS_RECURSION_H
#define ALGORITHMS_RECURSION_H

#include <string>
#include <vector>

namespace Algorithms
{
namespace Recursion
{

namespace Fibonacci
{

// 12.1.6 constexpr Functions Ch. 12 Functions; Bjarne Stroustrup,
// The C++ Programming Language, 4th Ed., Stroustrup.

// 3 "laws" of recursion
// 1. base case
// 2. must change its state and move toward base case
// 3. must call itself, recursively.

// Fibonacci numbers (sequence) as recurrence relation: 
// F_n = F_{n-1} + F_{n-2}
// F_1 = 1, F_0 = 1 (base cases)

//------------------------------------------------------------------------------
/// \name fib
/// \details since constexpr function, cannot have branching (i.e. if, elses)
//------------------------------------------------------------------------------
constexpr int fib_recursive(const int n)
{
  return (n < 2) ? n : (fib_recursive(n-1) + fib_recursive(n-2));
}

constexpr int ftbl[] {1, 2, 3, 4, 5, 8, 13};

constexpr int fib_with_table(int n)
{
  return (n < sizeof(ftbl) / sizeof(*ftbl)) ? ftbl[n] :
    fib_with_table(n-2) + fib_with_table(n-1);
}

} // namespace Fibonacci

// cf. https://www.hackerrank.com/challenges/ctci-recursive-staircase/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=recursion-backtracking
// Recursion: Davis' Staircase, Hackerrank.

// EY: Lesson 1: Think of the input. The input type must match the output type
// in order for the function to call itself recursively. Go from there.
//
// Lesson 2: Cacheing helps to speed up algorithm a lot.


/// 2020/10/14 17:44 Start.
/// 19:20 Started up non-recursive solution. Took break for dinner, rest ~30mins

namespace HackerRank
{

namespace DavisStaircases
{

int recursive_step_permutations(const int n);

int cached_step_permutations(const int n);

} // namespace DavisStaircases

// cf. https://www.hackerrank.com/challenges/crossword-puzzle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=recursion-backtracking

namespace CrosswordPuzzle
{

std::vector<std::string> split_string(
  std::string& s,
  const std::string& delimiter);

bool crosswordPuzzle(
  std::vector<std::string>& crossword,
  std::vector<std::string>& words);

std::vector<std::string> crosswordPuzzle(
  std::vector<std::string> crossword, std::string words);

} // namespace CrosswordPuzzle

} // namespace HackerRank

} // namespace Recursion
} // namespace Algorithms

#endif // ALGORITHMS_RECURSION_H