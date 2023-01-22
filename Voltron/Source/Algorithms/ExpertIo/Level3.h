#ifndef ALGORITHMS_EXPERT_IO_LEVEL_3_H
#define ALGORITHMS_EXPERT_IO_LEVEL_3_H

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
//------------------------------------------------------------------------------
std::vector<std::string> boggle_board(
  std::vector<std::vector<char>> board,
  std::vector<std::string> words);

} // namespace BoggleBoard

} // namespace ExpertIo
} // namespace Algorithms

#endif // ALGORITHMS_EXPERT_IO_LEVEL_3_H