#include "Algorithms/ExpertIo/Level1.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::ExpertIo::is_valid_subsequence;
using Algorithms::ExpertIo::two_number_sum_brute;
using Algorithms::ExpertIo::two_number_sum_with_map;

using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(ExpertIo)

BOOST_AUTO_TEST_SUITE(TwoNumberSum)

const vector<int> example_array {3, 5, -4, 8, 11, 1, -1, 6};
const int example_target_sum {10};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExampleFindingAnyTwoArraySumBruteForce)
{
  const vector<int> brute_solution {
    two_number_sum_brute(example_array, example_target_sum)};

  BOOST_TEST(brute_solution[0] == 11);
  BOOST_TEST(brute_solution[1] == -1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExampleFindingAnyTwoArraySumWithMap)
{
  const vector<int> map_solution {
    two_number_sum_with_map(example_array, example_target_sum)};

  BOOST_TEST(map_solution[0] == 11);
  BOOST_TEST(map_solution[1] == -1);
}

BOOST_AUTO_TEST_SUITE_END() // TwoNumberSum

BOOST_AUTO_TEST_SUITE(ValidateSubsequence)

const vector<int> sample_array {5, 1, 22, 25, 6, -1, 8, 10};
const vector<int> sample_sequence {1, 6, -1, 10};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExampleWithTwoPointers)
{
  BOOST_TEST(is_valid_subsequence(sample_array, sample_sequence));
}

BOOST_AUTO_TEST_SUITE_END() // ValidateSubsequence

BOOST_AUTO_TEST_SUITE_END() // ExpertIo
BOOST_AUTO_TEST_SUITE_END() // Algorithms