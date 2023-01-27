#include "Algorithms/ExpertIo/Level3.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using std::string;
using std::vector;

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
BOOST_AUTO_TEST_CASE(BoggleBoardWorks)
{
  // Sample
  {

  }

}

BOOST_AUTO_TEST_SUITE_END() // Level3_tests
BOOST_AUTO_TEST_SUITE_END() // ExpertIo
BOOST_AUTO_TEST_SUITE_END() // Algorithms