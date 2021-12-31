#include "DataStructures/LinkedLists/Node.h"

#include <boost/test/unit_test.hpp>

using DataStructures::LinkedLists::Nodes::Node;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(Nodes)
BOOST_AUTO_TEST_SUITE(Node_tests)

class NodeTestsFixture
{
  public:

    NodeTestsFixture():
      a_1_{42},
      a_2_{2},
      a_3_{3, &a_2_}
    {}

    virtual ~NodeTestsFixture() = default;

    Node<int> a_1_;
    Node<int> a_2_;
    Node<int> a_3_;    
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
	Node<int> node {};

	BOOST_TEST(node.value_ == 0);
	BOOST_TEST(node.next_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithElementValueOnly)
{
	Node<int> node {42};

	BOOST_TEST(node.value_ == 42);
	BOOST_TEST(node.next_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithValueAndNode)
{
	Node<int> node2 {69};

	Node<int> node1 {42, &node2};

	BOOST_TEST(node1.value_ == 42);
	BOOST_TEST(node1.next_->value_ == 69);

	BOOST_TEST(node1.next_->next_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CopyConstructs, NodeTestsFixture)
{
  Node a {a_3_};

  BOOST_TEST(a.value_ == 3);
  BOOST_TEST(a.next_->value_ == 2);

  BOOST_TEST(a_3_.value_ == 3);
  BOOST_TEST(a_3_.next_->value_ == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CopyAssignmentCopies, NodeTestsFixture)
{
  Node<int> a {};

  BOOST_TEST(a.value_ == 0);
  BOOST_TEST(a.next_ == nullptr);

  a = a_3_;

  BOOST_TEST(a.value_ == 3);
  BOOST_TEST(a.next_->value_ == 2);

  BOOST_TEST(a_3_.value_ == 3);
  BOOST_TEST(a_3_.next_->value_ == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveConstructs, NodeTestsFixture)
{
  Node a {std::move(a_3_)};

  BOOST_TEST(a.value_ == 3);
  BOOST_TEST(a.next_->value_ == 2);

  BOOST_TEST(a_3_.value_ == 3);
  BOOST_TEST(a_3_.next_->value_ == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveAssigns, NodeTestsFixture)
{
  Node<int> a {};

  BOOST_TEST(a.value_ == 0);
  BOOST_TEST(a.next_ == nullptr);

  a = std::move(a_3_);

  BOOST_TEST(a.value_ == 3);
  BOOST_TEST(a.next_->value_ == 2);

  BOOST_TEST(a_3_.value_ == 3);
  BOOST_TEST(a_3_.next_->value_ == 2);
}

BOOST_AUTO_TEST_SUITE_END() // Node_tests
BOOST_AUTO_TEST_SUITE_END() // Nodes
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures