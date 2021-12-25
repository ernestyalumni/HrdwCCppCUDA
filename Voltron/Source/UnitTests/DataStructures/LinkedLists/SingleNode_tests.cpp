//------------------------------------------------------------------------------
/// \file SingleNode_tests.cpp
//------------------------------------------------------------------------------
#include "DataStructures/LinkedLists/SingleNode.h"

#include <boost/test/unit_test.hpp>
#include <utility> // std::move

using DataStructures::LinkedLists::Nodes::SingleNode;
// Alternatively,
//template <typename T>
//using SingleNode = DataStructures::Lists::SingleNode::SingleNode<T>;

class SingleNodeTestsFixture
{
  public:

    SingleNodeTestsFixture():
      a_1_{42},
      a_2_{2},
      a_3_{3, &a_2_}
    {}

    virtual ~SingleNodeTestsFixture() = default;

    SingleNode<int> a_1_;
    SingleNode<int> a_2_;
    SingleNode<int> a_3_;
};

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithValueAndSingleNode)
{
	SingleNode<int> node2 {69};

	SingleNode<int> node1 {42, &node2};

	BOOST_TEST(node1.retrieve() == 42);
	BOOST_TEST(node1.next()->retrieve() == 69);

	BOOST_TEST(node1.next()->next() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CopyConstructs, SingleNodeTestsFixture)
{
  SingleNode a {a_3_};

  BOOST_TEST(a.retrieve() == 3);
  BOOST_TEST(a.next()->retrieve() == 2);

  BOOST_TEST(a_3_.retrieve() == 3);
  BOOST_TEST(a_3_.next()->retrieve() == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CopyAssignmentCopies, SingleNodeTestsFixture)
{
  SingleNode<int> a {};

  BOOST_TEST(a.retrieve() == 0);
  BOOST_TEST(a.next() == nullptr);

  a = a_3_;

  BOOST_TEST(a.retrieve() == 3);
  BOOST_TEST(a.next()->retrieve() == 2);

  BOOST_TEST(a_3_.retrieve() == 3);
  BOOST_TEST(a_3_.next()->retrieve() == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveConstructs, SingleNodeTestsFixture)
{
  SingleNode a {std::move(a_3_)};

  BOOST_TEST(a.retrieve() == 3);
  BOOST_TEST(a.next()->retrieve() == 2);

  BOOST_TEST(a_3_.retrieve() == 3);
  BOOST_TEST(a_3_.next()->retrieve() == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveAssigns, SingleNodeTestsFixture)
{
  SingleNode<int> a {};

  BOOST_TEST(a.retrieve() == 0);
  BOOST_TEST(a.next() == nullptr);

  a = std::move(a_3_);

  BOOST_TEST(a.retrieve() == 3);
  BOOST_TEST(a.next()->retrieve() == 2);

  BOOST_TEST(a_3_.retrieve() == 3);
  BOOST_TEST(a_3_.next()->retrieve() == 2);
}

BOOST_AUTO_TEST_SUITE_END() // SingleNode_tests
BOOST_AUTO_TEST_SUITE_END() // Nodes
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures