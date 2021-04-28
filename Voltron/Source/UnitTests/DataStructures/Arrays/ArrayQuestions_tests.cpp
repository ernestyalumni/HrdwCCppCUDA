#include "DataStructures/Arrays/ArrayQuestions.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using
  DataStructures::Arrays::ArrayQuestions::CrackingTheCodingInterview::
    is_unique_character_string;
using DataStructures::Arrays::ArrayQuestions::LeetCode::max_profit;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(ArrayQuestions_tests)

//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/language/ascii
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ASCIIDecimalConversionToChar)
{
  int i {33};

  BOOST_TEST(static_cast<char>(i) == '!');
}

BOOST_AUTO_TEST_CASE(CharConversionToASCIIDecimal)
{
  BOOST_TEST(static_cast<int>('#') == 35);

  const char c {'('};

  BOOST_TEST(static_cast<int>(c) == 40);
}

//------------------------------------------------------------------------------
/// \ref Gayle Laakmann McDowell, 6th Ed.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE(CrackingTheCodingInterview_tests)

//------------------------------------------------------------------------------
/// \ref pp. 90, 1.1 Is Unique, McDowell, 6th. Ed.
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsUniqueCharacterStringWorksWithCArray)
{
  const string duplicate_string {"GeeksforGeeks"};
  const string not_duplicate_string {"algorithm"};

  BOOST_TEST(!is_unique_character_string(duplicate_string));
  BOOST_TEST(is_unique_character_string(not_duplicate_string));
}

BOOST_AUTO_TEST_SUITE_END() // CrackingTheCodingInterview_tests

BOOST_AUTO_TEST_SUITE(Leetcode_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BestTimeToBuyAndSellStockIIElementaryCases)
{
  {
    vector<int> prices {42};
    BOOST_TEST(max_profit(prices) == 0);
  }
  {
    vector<int> prices {7, 2};
    BOOST_TEST(max_profit(prices) == 0);
  }
  {
    vector<int> prices {3, 5};
    BOOST_TEST(max_profit(prices) == 2);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BestTimeToBuyAndSellStockIIWithPointers)
{
  {
    const vector<int> prices {7, 1, 5, 3, 6, 4};

  }
  {
    const vector<int> prices {1, 2, 3, 4, 5};
    
  }
  {
    const vector<int> prices {7, 6, 4, 3, 1};
    
  }
}

BOOST_AUTO_TEST_SUITE_END() // Leetcode_tests

BOOST_AUTO_TEST_SUITE_END() // ArrayQuestions_tests
BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures