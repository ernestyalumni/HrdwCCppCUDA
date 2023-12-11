#include "DataStructures/Graphs/AdjacencyList.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Graphs::AdjacencyList;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Graphs)
BOOST_AUTO_TEST_SUITE(AdjacencyList_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  AdjacencyList gr {};

  BOOST_TEST(gr.get_size() == 0);

  BOOST_TEST(gr.get_number_of_edges() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithNumberOfVertices)
{
  {
    AdjacencyList gr {3};

    BOOST_TEST(gr.get_size() == 3);
    BOOST_TEST(gr.get_number_of_edges() == 0);
  }
  {
    AdjacencyList gr {4};

    BOOST_TEST(gr.get_size() == 4);
    BOOST_TEST(gr.get_number_of_edges() == 0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddEdgeAddsInOneDirection)
{
  AdjacencyList gr {3};
  gr.add_edge(1, 2);
  gr.add_edge(1, 0);
  gr.add_edge(2, 0);

  BOOST_TEST(gr.get_size() == 3);
  BOOST_TEST(gr.get_number_of_edges() == 3);

  BOOST_TEST(gr.is_edge(1, 0));
  BOOST_TEST(gr.is_edge(1, 2));
  BOOST_TEST(gr.is_edge(2, 0));

  BOOST_TEST(!gr.is_edge(0, 1));
  BOOST_TEST(!gr.is_edge(2, 1));
  BOOST_TEST(!gr.is_edge(0, 2));
}

BOOST_AUTO_TEST_SUITE_END() // AdjacencyList_tests

BOOST_AUTO_TEST_SUITE_END() // Graphs
BOOST_AUTO_TEST_SUITE_END() // DataStructures