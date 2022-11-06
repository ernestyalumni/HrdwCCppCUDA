#include "DataStructures/Graphs/AdjacencyArrayGraph.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Graphs)
BOOST_AUTO_TEST_SUITE(Kedyk)
BOOST_AUTO_TEST_SUITE(AdjacencyArrayGraph_tests)

using DataStructures::Graphs::Kedyk::AdjacencyArrayGraph;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  AdjacencyArrayGraph<int> aag {};
  BOOST_TEST(aag.number_of_vertices() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddVertexAddVertices)
{
  AdjacencyArrayGraph<int> aag {};
  for (std::size_t i {0}; i < 100; ++i)
  {
    aag.add_vertex();
    BOOST_TEST(aag.number_of_vertices() == 1 + i);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddVertexAddVerticesWithNoEdges)
{
  AdjacencyArrayGraph<int> aag {};
  for (std::size_t i {0}; i < 3; ++i)
  {
    aag.add_vertex();
    BOOST_TEST_REQUIRE(aag.number_of_vertices() == 1 + i);
  }

  for (std::size_t i {0}; i < 3; ++i)
  {
    BOOST_TEST(aag.number_of_edges(i) == 0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddEdgeAddEdges)
{
  // Example from https://www.geeksforgeeks.org/graph-and-its-representations/
  AdjacencyArrayGraph<int> aag {};
  aag.initialize(5);
  BOOST_TEST_REQUIRE(aag.number_of_vertices() == 5);
  aag.add_edge(0, 1);
  // malloc_consolidate(): invalid chunk size
  /*
  aag.add_edge(0, 4);
  aag.add_edge(1, 2);
  aag.add_edge(1, 3);
  aag.add_edge(1, 4);
  aag.add_edge(2, 3);
  aag.add_edge(3, 4);
  BOOST_TEST(aag.number_of_edges(0) == 2);
  BOOST_TEST(aag.number_of_edges(1) == 3);
  BOOST_TEST(aag.number_of_edges(2) == 1);
  BOOST_TEST(aag.number_of_edges(3) == 1);
  */
}

BOOST_AUTO_TEST_SUITE_END() // AdjacencyArrayGraph_tests

BOOST_AUTO_TEST_SUITE_END() // Kedyk
BOOST_AUTO_TEST_SUITE_END() // Graphs
BOOST_AUTO_TEST_SUITE_END() // DataStructures