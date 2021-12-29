#include "DataStructures/LinkedLists/Node.h"
#include "DataStructures/LinkedLists/ReverseList.h"

#include <boost/test/unit_test.hpp>

using DataStructures::LinkedLists::Nodes::Node;
using DataStructures::LinkedLists::reverse_list;

class NodeFixture
{
  public:

    NodeFixture():
      a_1_{1},
      a_2_{4, &a_1_},
      a_3_{9, &a_2_},
      a_4_{16, &a_3_}      
    {}

    virtual ~NodeFixture() = default;

    Node<int> a_1_;
    Node<int> a_2_;
    Node<int> a_3_;
    Node<int> a_4_;    
};

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(LinkedLists)
BOOST_AUTO_TEST_SUITE(ReverseList_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ReversesNodes, NodeFixture)
{
  Node<int>* head {&a_4_};
  BOOST_TEST_REQUIRE(head->value_ == 16);
  BOOST_TEST_REQUIRE(head->next_->value_ == 9);
  BOOST_TEST_REQUIRE(head->next_->next_->value_ == 4);
  BOOST_TEST_REQUIRE(head->next_->next_->next_->value_ == 1);

  Node<int>* new_head {reverse_list(head)};

  BOOST_TEST(new_head->value_ == 1);
  BOOST_TEST(new_head->next_->value_ == 4);
  BOOST_TEST(new_head->next_->next_->value_ == 9);
  BOOST_TEST(new_head->next_->next_->next_->value_ == 16);
}

BOOST_AUTO_TEST_SUITE_END() // ReverseList_tests
BOOST_AUTO_TEST_SUITE_END() // LinkedLists
BOOST_AUTO_TEST_SUITE_END() // DataStructures