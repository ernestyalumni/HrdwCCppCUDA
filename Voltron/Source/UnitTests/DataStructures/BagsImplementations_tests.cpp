#include "DataStructures/BagsImplementations.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Bags::BagAsVector;
using DataStructures::Bags::bag_statistics;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Bags)
BOOST_AUTO_TEST_SUITE(Bags_tests)

BOOST_AUTO_TEST_SUITE(BagAsVector_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  BagAsVector<double> bag;

  BOOST_TEST(bag.size() == 0);
  BOOST_TEST(bag.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddsToBagAsVector)
{
  BagAsVector<double> bag;

  BOOST_TEST_REQUIRE(bag.size() == 0);
  BOOST_TEST_REQUIRE(bag.is_empty());

  bag.add(100.0);
  BOOST_TEST(bag.size() == 1);
  BOOST_TEST(!bag.is_empty());

  bag.add(105.1);
  BOOST_TEST(bag.size() == 2);
  BOOST_TEST(!bag.is_empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsIterable)
{
  BagAsVector<int> bag;
  bag.add(10);
  bag.add(15);
  bag.add(20);

  int i {0};
  for (int x : bag)
  {
    BOOST_TEST(x == (10 + i * 5));
    i++;
  }
}

BOOST_AUTO_TEST_SUITE_END() // BagAsVector_tests

BOOST_AUTO_TEST_SUITE(BagStatistics_tests)


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CalculatesMeanAndStandardDeviation)
{
  const auto result = bag_statistics<BagAsVector>(
    {100, 99, 101, 120, 98, 107, 109, 81, 101, 90});

  BOOST_TEST(result.first == 100.60);
  BOOST_TEST(result.second == 10.511369505867867);
}

BOOST_AUTO_TEST_SUITE_END() // BagStatistics_tests

BOOST_AUTO_TEST_SUITE_END() // Bags_tests
BOOST_AUTO_TEST_SUITE_END() // Bags
BOOST_AUTO_TEST_SUITE_END() // DataStructures