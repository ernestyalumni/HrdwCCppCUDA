#include "DataStructures/Graphs/WeightedGraph.h"

#include <boost/test/unit_test.hpp>
#include <limits>

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Graphs)
BOOST_AUTO_TEST_SUITE(WeightedGraph_tests)

using DataStructures::Graphs::InefficientWeightedGraph;
using DataStructures::Graphs::WeightedGraph;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InefficientWeightedGraphInfinityStaticCastToBool)
{
  BOOST_TEST(static_cast<bool>(InefficientWeightedGraph<bool>::infinity_) ==
    true);

  BOOST_TEST(bool{} == false);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InefficientWeightedGraphInfinityStaticCastToInt)
{
  BOOST_TEST(static_cast<int>(InefficientWeightedGraph<int>::infinity_) ==
    std::numeric_limits<int>::max());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InefficientWeightedGraphConstructsWithDouble)
{
  InefficientWeightedGraph<double> wg3 {3};

  InefficientWeightedGraph<double> wg100 {100};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InefficientWeightedGraphConstructsWithBool)
{
  InefficientWeightedGraph<bool> wg3 {3};

  InefficientWeightedGraph<bool> wg100 {100};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WeightedGraphConstructsWithDouble)
{
  WeightedGraph<double> wg {6};
  BOOST_TEST(wg.number_of_vertices() == 6);
  BOOST_TEST(wg.number_of_edges() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WeightedGraphConstructsWithBool)
{
  WeightedGraph<bool> wg {5};
  BOOST_TEST(wg.number_of_vertices() == 5);
  BOOST_TEST(wg.number_of_edges() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WeightedGraphAddEdgeForBool)
{
  WeightedGraph<bool> wg {5};
  wg.add_edge(0, 1);
  wg.add_edge(0, 4);
  wg.add_edge(1, 0);
  wg.add_edge(1, 2);
  wg.add_edge(1, 3);
  wg.add_edge(1, 4);
  wg.add_edge(2, 1);
  wg.add_edge(2, 3);
  wg.add_edge(3, 1);
  wg.add_edge(3, 2);
  wg.add_edge(3, 4);
  wg.add_edge(4, 0);
  wg.add_edge(4, 1);
  wg.add_edge(4, 3);
  BOOST_TEST(wg.number_of_edges() == 14);
  BOOST_TEST(wg.is_edge(0, 1));
  BOOST_TEST(wg.is_edge(1, 0));
  BOOST_TEST(wg.is_edge(0, 4));
  BOOST_TEST(wg.is_edge(4, 0));
}

BOOST_AUTO_TEST_SUITE_END() // WeightedGraph_tests

BOOST_AUTO_TEST_SUITE_END() // Graphs
BOOST_AUTO_TEST_SUITE_END() // DataStructures