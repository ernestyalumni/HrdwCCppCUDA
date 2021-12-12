#include "Cpp/Std/Algorithm/LeviCivitaSymbol.h"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>

using Cpp::Std::Algorithm::LeviCivitaSymbol;
using Cpp::Std::Algorithm::is_even;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeviCivitaSymbol_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LeviCivitaSymbolDefaultConstructs)
{
  LeviCivitaSymbol eps {};

  const auto result = eps();

  BOOST_TEST(result.size() == 4);
  BOOST_TEST(result[0] == 0);
  BOOST_TEST(result[1] == 1);
  BOOST_TEST(result[2] == 2);
  BOOST_TEST(result[3] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LeviCivitaSymbolConstructsWithStartingValue)
{
  LeviCivitaSymbol<4> eps {1};

  const auto result = eps();

  BOOST_TEST(result.size() == 5);
  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 2);
  BOOST_TEST(result[2] == 3);
  BOOST_TEST(result[3] == 4);
  BOOST_TEST(result[4] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LeviCivitaSymbolOperatorCalculatesAllPermutations)
{
  LeviCivitaSymbol<3> eps {1};

  auto result = eps();

  BOOST_TEST(result.size() == 4);
  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 2);
  BOOST_TEST(result[2] == 3);
  BOOST_TEST(result[3] == 1);

  result = eps();

  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 3);
  BOOST_TEST(result[2] == 2);
  BOOST_TEST(result[3] == -1);

  result = eps();

  BOOST_TEST(result[0] == 2);
  BOOST_TEST(result[1] == 1);
  BOOST_TEST(result[2] == 3);
  BOOST_TEST(result[3] == -1);

  result = eps();

  BOOST_TEST(result[0] == 2);
  BOOST_TEST(result[1] == 3);
  BOOST_TEST(result[2] == 1);
  BOOST_TEST(result[3] == 1);

  result = eps();

  BOOST_TEST(result[0] == 3);
  BOOST_TEST(result[1] == 1);
  BOOST_TEST(result[2] == 2);
  BOOST_TEST(result[3] == 1);

  result = eps();

  BOOST_TEST(result[0] == 3);
  BOOST_TEST(result[1] == 2);
  BOOST_TEST(result[2] == 1);
  BOOST_TEST(result[3] == -1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsEvenWorksOnPermutationsOf3Numbers)
{
  LeviCivitaSymbol eps {};

  std::array<unsigned int, 3> value;
  auto result = eps();
  std::copy(result.begin(), result.end() - 1, value.begin());

  BOOST_TEST(result.size() == 4);
  BOOST_TEST(result[0] == 0);
  BOOST_TEST(result[1] == 1);
  BOOST_TEST(result[2] == 2);
  BOOST_TEST(result[3] == 1);
  BOOST_TEST(is_even(value, 3));

  result = eps();
  std::copy(result.begin(), result.end() - 1, value.begin());

  BOOST_TEST(result[0] == 0);
  BOOST_TEST(result[1] == 2);
  BOOST_TEST(result[2] == 1);
  BOOST_TEST(result[3] == -1);

  BOOST_TEST(value[0] == 0);
  BOOST_TEST(value[1] == 2);
  BOOST_TEST(value[2] == 1);
  BOOST_TEST(!is_even(value, 3));

  result = eps();
  std::copy(result.begin(), result.end() - 1, value.begin());

  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 0);
  BOOST_TEST(result[2] == 2);
  BOOST_TEST(result[3] == -1);
  BOOST_TEST(!is_even(value, 3));

  result = eps();
  std::copy(result.begin(), result.end() - 1, value.begin());

  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 2);
  BOOST_TEST(result[2] == 0);
  BOOST_TEST(result[3] == 1);
  BOOST_TEST(is_even(value, 3));

  result = eps();
  std::copy(result.begin(), result.end() - 1, value.begin());

  BOOST_TEST(result[0] == 2);
  BOOST_TEST(result[1] == 0);
  BOOST_TEST(result[2] == 1);
  BOOST_TEST(result[3] == 1);
  BOOST_TEST(is_even(value, 3));

  result = eps();
  std::copy(result.begin(), result.end() - 1, value.begin());

  BOOST_TEST(result[0] == 2);
  BOOST_TEST(result[1] == 1);
  BOOST_TEST(result[2] == 0);
  BOOST_TEST(result[3] == -1);
  BOOST_TEST(!is_even(value, 3));
}

BOOST_AUTO_TEST_SUITE_END() // LeviCivitaSymbol_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms
BOOST_AUTO_TEST_SUITE_END() // Cpp
