//------------------------------------------------------------------------------
/// \file Lists_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "DataStructures/Lists.h"

#include <boost/test/unit_test.hpp>
//#include <iostream>
#include <memory>
#include <tuple>

using namespace DataStructures::Lists;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Lists_tests)
BOOST_AUTO_TEST_SUITE(SinglyLinked)

BOOST_AUTO_TEST_SUITE(Nodes)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromValue)
{
	{
		Lists::SinglyLinked::Node node {2};
		BOOST_TEST(node.value() == 2);
	}
}

BOOST_AUTO_TEST_SUITE_END() // Nodes

BOOST_AUTO_TEST_SUITE_END() // SinglyLinked
BOOST_AUTO_TEST_SUITE_END() // Lists_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures
