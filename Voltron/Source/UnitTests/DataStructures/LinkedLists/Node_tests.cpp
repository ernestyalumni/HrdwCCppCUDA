#include "DataStructures/LinkedLists/Node.h"

#include <boost/test/unit_test.hpp>

using DataStructures::LinkedLists::Nodes::Node;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(Nodes)
BOOST_AUTO_TEST_SUITE(Node_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
	Node<int> node {};

	BOOST_TEST(node.get_value() == 0);
	BOOST_TEST(node.get_next() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithElementValueOnly)
{
	Node<int> node {42};

	BOOST_TEST(node.get_value() == 42);
	BOOST_TEST(node.get_next() == nullptr);		
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithValueAndSingleNode)
{
	Node<int> node2 {69};

	Node<int> node1 {42, &node2};

	BOOST_TEST(node1.get_value() == 42);
	BOOST_TEST(node1.get_next()->get_value() == 69);

	BOOST_TEST(node1.get_next()->get_next() == nullptr);
}

BOOST_AUTO_TEST_SUITE_END() // Node_tests
BOOST_AUTO_TEST_SUITE_END() // Nodes
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures