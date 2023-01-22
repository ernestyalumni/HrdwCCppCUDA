#include "Level2.h"

#include <cstddef>
#include <string>
#include <vector>

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
  vector<string> results {};

  for (string& word: words)
  {
    const int M {static_cast<int>(board.size())};
    const int N {static_cast<int>(board[0].size())};

    for (int i {0}; i < M; ++i)
    {
      for (int j {0}; j < N ++j)
      {
        if (word[0] != board[i][j])
        {
          continue;
        }

      }
    }
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
