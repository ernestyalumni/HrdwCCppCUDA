#include "DataStructures/Arrays/ArrayQuestions.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using std::vector;
using DataStructures::Arrays::ArrayQuestions::LeetCode::max_profit;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)
BOOST_AUTO_TEST_SUITE(ArrayQuestions_tests)
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