//------------------------------------------------------------------------------
/// \file SingleNode_tests.cpp
//------------------------------------------------------------------------------
#include "DataStructures/Lists/SingleNode.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Lists::Nodes::SingleNode;
// Alternatively,
//template <typename T>
//using SingleNode = DataStructures::Lists::SingleNode::SingleNode<T>;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Lists)
BOOST_AUTO_TEST_SUITE(Nodes)
BOOST_AUTO_TEST_SUITE(SingleNode_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
	SingleNode<int> node {};

	BOOST_TEST(node.retrieve() == 0);
	BOOST_TEST(node.next() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithElementValueOnly)
{
	SingleNode<int> node {42};

	BOOST_TEST(node.retrieve() == 42);
	BOOST_TEST(node.next() == nullptr);		
}

BOOST_AUTO_TEST_SUITE_END() // SingleNode_tests
BOOST_AUTO_TEST_SUITE_END() // Nodes
BOOST_AUTO_TEST_SUITE_END() // Lists
BOOST_AUTO_TEST_SUITE_END() // DataStructures