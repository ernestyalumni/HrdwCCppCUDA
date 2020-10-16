//------------------------------------------------------------------------------
// \file Recursion_tests.cpp
//------------------------------------------------------------------------------
#include "Algorithms/Recursion.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Algorithms::Recursion::Fibonacci::fib_recursive;
using Algorithms::Recursion::Fibonacci::fib_with_table;
using Algorithms::Recursion::HackerRank::CrosswordPuzzle::split_string;
using
  Algorithms::Recursion::HackerRank::DavisStaircases::
    recursive_step_permutations;
using
  Algorithms::Recursion::HackerRank::DavisStaircases::cached_step_permutations;
using std::string;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Recursion_tests)
BOOST_AUTO_TEST_SUITE(Fibonacci_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FibonacciFunctionsComputeFibonacciNumbers)
{ 
  BOOST_TEST(fib_recursive(1) == 1);
  BOOST_TEST(fib_recursive(2) == 1);
  BOOST_TEST(fib_recursive(13) == 233);
  BOOST_TEST(fib_recursive(14) == 377);
  BOOST_TEST(fib_with_table(1) == 2);
  BOOST_TEST(fib_with_table(2) == 3);
  BOOST_TEST(fib_with_table(13) == 377);
  BOOST_TEST(fib_with_table(14) == 610);
}

BOOST_AUTO_TEST_SUITE_END() // Fibonacci_tests

BOOST_AUTO_TEST_SUITE(HackerRank_tests)

BOOST_AUTO_TEST_SUITE(DavisStaircases_tests)

// cf. https://www.hackerrank.com/challenges/ctci-recursive-staircase/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=recursion-backtracking

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursiveStepPermutationCalculatesForLowValueIntegers)
{ 
  // Test case 0
  BOOST_TEST(recursive_step_permutations(1) == 1);
  BOOST_TEST(recursive_step_permutations(3) == 4);
  BOOST_TEST(recursive_step_permutations(7) == 44);

  // Test case 9, HackerRank
  BOOST_TEST(recursive_step_permutations(5) == 13);
  BOOST_TEST(recursive_step_permutations(8) == 81);
  // Test case 10, HackerRank
  BOOST_TEST(recursive_step_permutations(15) == 5768);
  BOOST_TEST(recursive_step_permutations(20) == 121415);
  BOOST_TEST(recursive_step_permutations(27) == 8646064);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CachingStepPermutationCalculatesManyIntegers)
{ 
  // Test case 0
  BOOST_TEST(cached_step_permutations(1) == 1);
  BOOST_TEST(cached_step_permutations(3) == 4);
  BOOST_TEST(cached_step_permutations(7) == 44);

  // Test case 9, HackerRank
  BOOST_TEST(cached_step_permutations(5) == 13);
  BOOST_TEST(cached_step_permutations(8) == 81);
  // Test case 10, HackerRank
  BOOST_TEST(cached_step_permutations(15) == 5768);
  BOOST_TEST(cached_step_permutations(20) == 121415);
  BOOST_TEST(cached_step_permutations(27) == 8646064);
}

BOOST_AUTO_TEST_SUITE_END() // DavisStaircases_tests

// cf. https://www.hackerrank.com/challenges/crossword-puzzle/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=recursion-backtracking

BOOST_AUTO_TEST_SUITE(DavisStaircases_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SplitStringSplitsSubstringsByDelimiter)
{ 
  string list_of_words {"POLAND;LHASA;SPAIN;INDIA"};
  const string delimiter {";"};
  const auto split_strings = split_string(list_of_words, delimiter);
  BOOST_TEST(split_strings[0] == "POLAND");
  BOOST_TEST(split_strings[1] == "LHASA");
  BOOST_TEST(split_strings[2] == "SPAIN");
  BOOST_TEST(split_strings[3] == "INDIA");
}

BOOST_AUTO_TEST_SUITE_END() // DavisStaircases_tests

/*

Test case 0

    +-++++++++

    +-++++++++

    +-++++++++

    +-----++++

    +-+++-++++

    +-+++-++++

    +++++-++++

    ++------++

    +++++-++++

    +++++-++++

    LONDON;DELHI;ICELAND;ANKARA

Expected Output
Download

    +L++++++++

    +O++++++++

    +N++++++++

    +DELHI++++

    +O+++C++++

    +N+++E++++

    +++++L++++

    ++ANKARA++

    +++++N++++

    +++++D++++

Test case 1

    +-++++++++

    +-++++++++

    +-------++

    +-++++++++

    +-++++++++

    +------+++

    +-+++-++++

    +++++-++++

    +++++-++++

    ++++++++++

    AGRA;NORWAY;ENGLAND;GWALIOR

Expected Output
Download

    +E++++++++

    +N++++++++

    +GWALIOR++

    +L++++++++

    +A++++++++

    +NORWAY+++

    +D+++G++++

    +++++R++++

    +++++A++++

    ++++++++++


Test case 7

    XXXXXX-XXX

    XX------XX

    XXXXXX-XXX

    XXXXXX-XXX

    XXX------X

    XXXXXX-X-X

    XXXXXX-X-X

    XXXXXXXX-X

    XXXXXXXX-X

    XXXXXXXX-X

    ICELAND;MEXICO;PANAMA;ALMATY

Expected Output
Download

    XXXXXXIXXX

    XXMEXICOXX

    XXXXXXEXXX

    XXXXXXLXXX

    XXXPANAMAX

    XXXXXXNXLX

    XXXXXXDXMX

    XXXXXXXXAX

    XXXXXXXXTX

    XXXXXXXXYX


*/

BOOST_AUTO_TEST_SUITE_END() // HackerRank_tests

BOOST_AUTO_TEST_SUITE_END() // Recursion_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms