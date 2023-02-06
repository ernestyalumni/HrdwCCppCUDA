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
using std::unordered_set;
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

  unordered_set<string> found_words {};

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

  for (auto iter = found_words.begin(); iter != found_words.end(); ++iter)
  {
    results.emplace_back(std::move(found_words.extract(iter).value()));
  }

  return results;
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

void explore(
  const std::size_t i,
  const std::size_t j,
  const vector<vector<char>>& board,
  const Trie<alphabet_size>::Node* node_ptr,
  vector<vector<bool>>& is_visited,
  unordered_set<string>& found_words)
{
  if (is_visited[i][j])
  {
    return;
  }

  const char letter {board[i][j]};

  if (!node_ptr->is_in_children(letter))
  {
    return;
  }

  is_visited[i][j] = true;

  node_ptr = node_ptr->children_[static_cast<size_t>(letter)];
  if (node_ptr->is_end_of_word_)
  {
    //found_words
  }

  VectorOfCoordinates neighbors {
    get_neighbors(i, j, board.size(), board[0].size())};

  for (const auto neighbor : neighbors)
  {
    explore(neighbor[0], neighbor[1], board, node_ptr, is_visited, found_words);
  }

  // At the end, we must mark as unvisited the starting point.
  is_visited[i][j] = false;
}

} // namespace BoggleBoard

} // namespace ExpertIo
} // namespace Algorithms
