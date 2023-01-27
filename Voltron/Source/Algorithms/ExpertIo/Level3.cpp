#include "Level3.h"

//------------------------------------------------------------------------------
/// Data structure that lends itself very well to matching characters to
/// multiple strings at once.
//------------------------------------------------------------------------------
#include "DataStructures/Trees/Trie.h"

#include <cstddef>
#include <string>
#include <vector>

using DataStructures::Trees::Tries::Trie;
using std::size_t;
using std::string;
using std::vector;

namespace Algorithms
{
namespace ExpertIo
{

namespace BoggleBoard
{

vector<string> boggle_board(vector<vector<char>> board, vector<string> words)
{
  Trie<128> t {};
  for (const string& word: words)
  {
    t.insert(word);
  }

  return {};
}

void match_word_on_board(vector<vector<char>>& board, string& word)
{
  const int M {static_cast<int>(board.size())};
  const int N {static_cast<int>(board[0].size())};

}

} // namespace BoggleBoard

} // namespace ExpertIo
} // namespace Algorithms
