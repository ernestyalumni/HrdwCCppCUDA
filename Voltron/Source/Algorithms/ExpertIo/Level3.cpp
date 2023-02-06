#include "Level3.h"

//------------------------------------------------------------------------------
/// Data structure that lends itself very well to matching characters to
/// multiple strings at once.
//------------------------------------------------------------------------------
#include "DataStructures/Trees/Trie.h"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

using DataStructures::Trees::Tries::Trie;
using std::array;
using std::size_t;
using std::string;
using std::vector;

namespace Algorithms
{
namespace ExpertIo
{

namespace BoggleBoard
{

vector<string> solve_boggle_board(vector<vector<char>> board, vector<string> words)
{
  Trie<alphabet_size> t {};
  for (const string& word: words)
  {
    t.insert(word);
  }

  //----------------------------------------------------------------------------
  /// TODO: Consider doing empirical tests because branching penalty for
  /// std::unordered_map may be higher than even for std::vector.
  /// https://gist.github.com/gx578007/836e3ba0069d1570086b0a4c114dca31
  /// https://medium.com/@gx578007/searching-vector-set-and-unordered-set-6649d1aa7752#:~:text=The%20time%20complexity%20to%20find,overheads%20to%20find%20an%20element.
  //----------------------------------------------------------------------------

  std::unordered_set<string> final_words {};

  vector<vector<bool>> is_visited {};
  for (size_t i {0}; i < board.size(); ++i)
  {
    is_visited.emplace_back(vector<bool>(board[0].size(), false));
  }

  for (size_t i {0}; i < board.size(); ++i)
  {
    for (size_t j {0}; j < board[0].size(); ++j)
    {

    }
  }

  vector<string> results {};

  // See https://stackoverflow.com/questions/42519867/efficiently-moving-contents-of-stdunordered-set-to-stdvector
  // value returns a reference to container element object managed by node
  // handle.
  // See https://en.cppreference.com/w/cpp/container/node_handle
  // node_type extract(const_iterator position)
  // Unlinks node that contains element pointed by position and returns node
  // handle that owns it.
  // See https://en.cppreference.com/w/cpp/container/unordered_set/extract

  for (auto iter = final_words.begin(); iter != final_words.end(); ++iter)
  {
    results.emplace_back(std::move(final_words.extract(iter).value()));
  }

  return results;
}

void match_word_on_board(vector<vector<char>>& board, string& word)
{
  const int M {static_cast<int>(board.size())};
  const int N {static_cast<int>(board[0].size())};

}

VectorOfCoordinates get_neighbors(
  const size_t i,
  const size_t j,
  const size_t board_height,
  const size_t board_length)
{
  VectorOfCoordinates results {};

  if (i < board_height - 1)
  {
    results.emplace_back(array<size_t, 2>{i + 1, j});
  }

  if (i < board_height - 1 && j > 0)
  {
    results.emplace_back(array<size_t, 2>{i + 1, j - 1});
  }

  if (j > 0)
  {
    results.emplace_back(array<size_t, 2>{i, j - 1});
  }

  if (i > 0 && j > 0)
  {
    results.emplace_back(array<size_t, 2>{i - 1, j - 1});
  }

  if (i > 0)
  {
    results.emplace_back(array<size_t, 2>{i - 1, j});
  }

  if (i > 0 && j < board_length - 1)
  {
    results.emplace_back(array<size_t, 2>{i - 1, j + 1});
  }

  if (j < board_length - 1)
  {
    results.emplace_back(array<size_t, 2>{i, j + 1});
  }

  if (i < board_height - 1 && j < board_length - 1)
  {
    results.emplace_back(array<size_t, 2>{i + 1, j + 1});
  }

  return results;
}

} // namespace BoggleBoard

} // namespace ExpertIo
} // namespace Algorithms
