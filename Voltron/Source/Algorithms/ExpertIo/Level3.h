#ifndef ALGORITHMS_EXPERT_IO_LEVEL_3_H
#define ALGORITHMS_EXPERT_IO_LEVEL_3_H

#include "DataStructures/Trees/Trie.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace Algorithms
{
namespace ExpertIo
{

namespace BoggleBoard
{

constexpr std::size_t alphabet_size {128};

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
std::vector<std::string> solve_boggle_board(
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

using VectorOfCoordinates = std::vector<std::array<std::size_t, 2>>;

//------------------------------------------------------------------------------
/// \details Assume i, j are legitimate coordinates on the board.
/// We cannot use template parameters for the board's dimensions, because
/// template parameters need to be constexpr values for constant parameters.
//------------------------------------------------------------------------------
VectorOfCoordinates get_neighbors(
  const std::size_t i,
  const std::size_t j,
  const std::size_t board_height,
  const std::size_t board_length);

void explore(const std::size_t i, const std::size_t j, std::vector<std::vector<bool>>& is_visited);
//DataStructure::Trees::Tries::Trie<

} // namespace BoggleBoard

} // namespace ExpertIo
} // namespace Algorithms

#endif // ALGORITHMS_EXPERT_IO_LEVEL_3_H