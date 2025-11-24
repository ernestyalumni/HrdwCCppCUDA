#include "Algorithms/PreEasyExercises.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::PreEasyExercises::ArrayIndexing;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(PreEasyExercises)

BOOST_AUTO_TEST_SUITE(ArrayIndexing_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterateOverArrayIteratesOverAnArray)
{
  vector<int> array {1, 2, 3, 4, 5};
  BOOST_TEST(ArrayIndexing::iterate_over_array(array) == array);

  ArrayIndexing::iterate_over_array(array, true);
}

BOOST_AUTO_TEST_SUITE_END() // ArrayIndexing_tests

BOOST_AUTO_TEST_SUITE_END() // PreEasyExercises
BOOST_AUTO_TEST_SUITE_END() // Algorithms