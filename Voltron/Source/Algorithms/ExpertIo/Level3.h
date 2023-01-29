#ifndef ALGORITHMS_EXPERT_IO_LEVEL_3_H
#define ALGORITHMS_EXPERT_IO_LEVEL_3_H

#include "DataStructures/Trees/Trie.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Algorithms
{
namespace ExpertIo
{

namespace BoggleBoard
{

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/word-search-ii/
/// \brief Given m x n board of characters and list of strings, words, return
/// all words on the board.
/// \details
///
/// Space complexity
/// Given s = maximum length of a string, w = number of words to store,
/// Space complexity due to using a trie is O(ws).
/// Given boggle board of dimensions m x n,
/// Space complexity due to auxiliary data structure with bools for marking if
/// node is visited, is O(nm).
/// Space complexity due to recursion so we have the call stack, is s, for the
/// largest length of a word as we do depth-first search.
///
/// Time complexity
//------------------------------------------------------------------------------
std::vector<std::string> boggle_board(
  std::vector<std::vector<char>> board,
  std::vector<std::string> words);

class TraverseBoggleBoard
{
  public:

    static constexpr std::size_t alphabet_size_ {128};

    static constexpr int directions[8][2] {
      {1, 0},
      {1, 1},
      {0, 1},
      {-1, 1},
      {-1, 0},
      {-1, -1},
      {0, -1},
      {1, -1}};

    void preorder_depth_first_traverse(
      const std::size_t i,
      const std::size_t j,
      std::vector<std::vector<char>>& board,
      std::vector<std::vector<bool>>& is_visited);
};

} // namespace BoggleBoard

} // namespace ExpertIo
} // namespace Algorithms

#endif // ALGORITHMS_EXPERT_IO_LEVEL_3_H