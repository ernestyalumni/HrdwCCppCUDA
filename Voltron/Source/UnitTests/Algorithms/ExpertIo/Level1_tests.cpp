#include "Algorithms/ExpertIo/Level1.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::ExpertIo::two_number_sum_brute;

using std::vector;

const vector<int> example_array {3, 5, -4, 8, 11, 1, -1, 6};
const int example_target_sum {10};

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(ExpertIo)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExampleFindingAnyTwoArraySumBruteForce)
{
  const vector<int> brute_solution {
    two_number_sum_brute(example_array, example_target_sum)};


  BOOST_TEST(brute_solution[0] == 11);
  BOOST_TEST(brute_solution[1] == -1);
}

BOOST_AUTO_TEST_SUITE_END() // ExpertIo
BOOST_AUTO_TEST_SUITE_END() // Algorithms