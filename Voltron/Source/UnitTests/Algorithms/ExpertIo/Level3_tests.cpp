#include "Algorithms/ExpertIo/Level3.h"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using std::array;
using std::find;
using std::string;
using std::vector;

using namespace Algorithms::ExpertIo;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(ExpertIo)

BOOST_AUTO_TEST_SUITE(Level3_tests)

const vector<vector<char>> sample_board {
  vector<char>{'t','h','i','s','i','s','a'},
  vector<char>{'s','i','m','p','l','e','x'},
  vector<char>{'b','x','x','x','x','e','b'},
  vector<char>{'x','o','g','g','l','x','o'},
  vector<char>{'x','x','x','D','T','r','a'},
  vector<char>{'R','E','P','E','A','d','x'},
  vector<char>{'x','x','x','x','x','x','x'},
  vector<char>{'N','O','T','R','E','-','P'},
  vector<char>{'x','x','D','E','T','A','E'}};

const vector<string> sample_words {
  "this",
  "is",
  "not",
  "a",
  "simple",
  "boggle",
  "board",
  "test",
  "REPEATED",
  "NOTRE-PEATED"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetNeighborsGetsLegitimateNeighbors)
{
  {
    BoggleBoard::VectorOfCoordinates results {
      BoggleBoard::get_neighbors(
        0,
        0,
        sample_board.size(),
        sample_board[0].size())};

    BOOST_TEST(results.size() == 3);

    // https://en.cppreference.com/w/cpp/algorithm/find
    // template <class InputIt, class T>
    // constexpr InputIt find(InputIt first, InputIt last, const T& value);

    auto result1 = find(results.begin(), results.end(), array<size_t, 2>{1, 0});
    auto result2 = find(results.begin(), results.end(), array<size_t, 2>{1, 1});
    auto result3 = find(results.begin(), results.end(), array<size_t, 2>{0, 1});

    BOOST_TEST((result1 != results.end()));
    BOOST_TEST((result2 != results.end()));
    BOOST_TEST((result3 != results.end()));
  }
  {
    BoggleBoard::VectorOfCoordinates results {
      BoggleBoard::get_neighbors(
        8,
        0,
        sample_board.size(),
        sample_board[0].size())};

    BOOST_TEST(results.size() == 3);

    auto result1 = find(results.begin(), results.end(), array<size_t, 2>{8, 1});
    auto result2 = find(results.begin(), results.end(), array<size_t, 2>{7, 1});
    auto result3 = find(results.begin(), results.end(), array<size_t, 2>{7, 0});

    BOOST_TEST((result1 != results.end()));
    BOOST_TEST((result2 != results.end()));
    BOOST_TEST((result3 != results.end()));
  }
  {
    BoggleBoard::VectorOfCoordinates results {
      BoggleBoard::get_neighbors(
        8,
        6,
        sample_board.size(),
        sample_board[0].size())};

    BOOST_TEST(results.size() == 3);

    auto result1 = find(results.begin(), results.end(), array<size_t, 2>{8, 5});
    auto result2 = find(results.begin(), results.end(), array<size_t, 2>{7, 5});
    auto result3 = find(results.begin(), results.end(), array<size_t, 2>{7, 6});

    BOOST_TEST((result1 != results.end()));
    BOOST_TEST((result2 != results.end()));
    BOOST_TEST((result3 != results.end()));
  }
  {
    BoggleBoard::VectorOfCoordinates results {
      BoggleBoard::get_neighbors(
        0,
        6,
        sample_board.size(),
        sample_board[0].size())};

    BOOST_TEST(results.size() == 3);

    auto result1 = find(results.begin(), results.end(), array<size_t, 2>{0, 5});
    auto result2 = find(results.begin(), results.end(), array<size_t, 2>{1, 5});
    auto result3 = find(results.begin(), results.end(), array<size_t, 2>{1, 6});

    BOOST_TEST((result1 != results.end()));
    BOOST_TEST((result2 != results.end()));
    BOOST_TEST((result3 != results.end()));
  }
}

BOOST_AUTO_TEST_SUITE_END() // Level3_tests
BOOST_AUTO_TEST_SUITE_END() // ExpertIo
BOOST_AUTO_TEST_SUITE_END() // Algorithms