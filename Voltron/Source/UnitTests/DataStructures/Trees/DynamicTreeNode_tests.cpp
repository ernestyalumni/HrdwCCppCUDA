#include "DataStructures/Trees/DynamicTreeNode.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using DataStructures::Trees::DynamicTreeNode;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(DynamicTreeNode_tests)

class DynamicTreeNodeFixture
{
  public:

    DynamicTreeNodeFixture():
      root_ptr_{new DynamicTreeNode<char>{'A'}}
    {
      root_ptr_->add_child('B');
      root_ptr_->add_child('C');
      root_ptr_->add_child('D');
      root_ptr_->child(0)->add_child('E');
      root_ptr_->child(0)->add_child('F');
      root_ptr_->child(1)->add_child('H');
      root_ptr_->child(2)->add_child('K');
      root_ptr_->child(2)->add_child('L');
      root_ptr_->child(2)->add_child('M');
      root_ptr_->child(0)->child(0)->add_child('G');
      root_ptr_->child(1)->child(0)->add_child('I');
      root_ptr_->child(1)->child(0)->add_child('J');
      root_ptr_->child(2)->child(2)->add_child('N');
    }

    ~DynamicTreeNodeFixture()
    {
      delete root_ptr_;
    }

    DynamicTreeNode<char>* root_ptr_;
};

class MoreDynamicTreeNodeFixture
{
  public:

    MoreDynamicTreeNodeFixture():
      root_ptr1_{new DynamicTreeNode<char>{'A'}},
      root_ptr2_{new DynamicTreeNode<char>{'A'}}
    {
      root_ptr1_->add_child('B');
      root_ptr1_->add_child('H');
      root_ptr1_->child(0)->add_child('C');
      root_ptr1_->child(0)->add_child('E');
      root_ptr1_->child(1)->add_child('I');
      root_ptr1_->child(1)->add_child('M');
      root_ptr1_->child(0)->child(0)->add_child('D');
      root_ptr1_->child(0)->child(1)->add_child('F');
      root_ptr1_->child(0)->child(1)->add_child('G');
      root_ptr1_->child(1)->child(0)->add_child('J');
      root_ptr1_->child(1)->child(0)->add_child('K');
      root_ptr1_->child(1)->child(0)->add_child('L');

      root_ptr2_->add_child('B');
      root_ptr2_->child(0)->add_child('D');
      root_ptr2_->child(0)->add_child('E');
      root_ptr2_->child(0)->child(1)->add_child('G');
      root_ptr2_->add_child('C');
      root_ptr2_->child(1)->add_child('F');
    }

    ~MoreDynamicTreeNodeFixture()
    {
      delete root_ptr1_;
      delete root_ptr2_;
    }

    DynamicTreeNode<char>* root_ptr1_;
    DynamicTreeNode<char>* root_ptr2_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsRoot)
{
  DynamicTreeNode<char> root {'A'};

  BOOST_TEST(root.value() == 'A');
  BOOST_TEST(root.degree() == 0);
  BOOST_TEST(root.parent() == nullptr);
  BOOST_TEST(root.is_root());
  BOOST_TEST(root.is_leaf());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsRootDynamically)
{
  DynamicTreeNode<char>* root_ptr {new DynamicTreeNode<char>{'A'}};

  BOOST_TEST(root_ptr->value() == 'A');
  BOOST_TEST(root_ptr->degree() == 0);
  BOOST_TEST(root_ptr->parent() == nullptr);
  BOOST_TEST(root_ptr->is_root());
  BOOST_TEST(root_ptr->is_leaf());

  delete root_ptr;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddChildDynamicallyAddsChild)
{
  {
    DynamicTreeNode<char> root {'A'};
    root.add_child('B');
    BOOST_TEST(root.degree() == 1);
    BOOST_TEST(!root.is_leaf());
    BOOST_TEST(!root.child(0)->is_root());
    BOOST_TEST(root.child(0)->is_leaf());
  }
  {
    DynamicTreeNode<char> root {'A'};
    root.add_child('B');
    root.add_child('C');
    root.child(0)->add_child('E');
    root.child(0)->add_child('F');
    BOOST_TEST(root.degree() == 2);
    BOOST_TEST(root.child(0)->degree() == 2);
    BOOST_TEST(root.child(1)->is_leaf());
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddChildAddsChildToDynamicallyConstructedRoot)
{
  {
    DynamicTreeNode<char>* root_ptr {new DynamicTreeNode<char>{'A'}};
    root_ptr->add_child('B');
    BOOST_TEST(root_ptr->degree() == 1);
    BOOST_TEST(!root_ptr->is_leaf());
    BOOST_TEST(!root_ptr->child(0)->is_root());
    BOOST_TEST(root_ptr->child(0)->is_leaf());

    delete root_ptr;
  }
  {
    DynamicTreeNode<char>* root_ptr {new DynamicTreeNode<char>{'A'}};
    root_ptr->add_child('B');
    root_ptr->add_child('C');
    root_ptr->child(0)->add_child('E');
    root_ptr->child(0)->add_child('F');
    BOOST_TEST(root_ptr->degree() == 2);
    BOOST_TEST(root_ptr->child(0)->degree() == 2);
    BOOST_TEST(root_ptr->child(1)->is_leaf());

    delete root_ptr;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SizeGetsNumberOfNodes, DynamicTreeNodeFixture)
{
  BOOST_TEST(root_ptr_->size(), 14);
  BOOST_TEST(root_ptr_->child(2)->size(), 5);
  BOOST_TEST(root_ptr_->child(1)->child(0)->size(), 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(HeightGetsHeight, DynamicTreeNodeFixture)
{
  BOOST_TEST(root_ptr_->height(), 4);
  BOOST_TEST(root_ptr_->child(2)->height(), 3);
  BOOST_TEST(root_ptr_->child(1)->child(0)->height(), 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  PreOrderTraversalRecursiveGetsNodesWhenFirstVisited,
  MoreDynamicTreeNodeFixture)
{
  {
    const auto result = preorder_traversal_recursive(root_ptr1_);
    BOOST_TEST_REQUIRE(root_ptr1_->size() == 13);
    BOOST_TEST_REQUIRE(result.size() == 13);
    BOOST_TEST(result[0] == 'A');
    BOOST_TEST(result[1] == 'B');
    BOOST_TEST(result[2] == 'C');
    BOOST_TEST(result[3] == 'D');
    BOOST_TEST(result[4] == 'E');
    BOOST_TEST(result[5] == 'F');
    BOOST_TEST(result[6] == 'G');
    BOOST_TEST(result[7] == 'H');
    BOOST_TEST(result[8] == 'I');
    BOOST_TEST(result[9] == 'J');
    BOOST_TEST(result[10] == 'K');
    BOOST_TEST(result[11] == 'L');
    BOOST_TEST(result[12] == 'M');
  }
  {
    const auto result = preorder_traversal_recursive(root_ptr2_);
    BOOST_TEST_REQUIRE(root_ptr2_->size() == 7);
    BOOST_TEST_REQUIRE(result.size() == 7);
    BOOST_TEST(result[0] == 'A');
    BOOST_TEST(result[1] == 'B');
    BOOST_TEST(result[2] == 'D');
    BOOST_TEST(result[3] == 'E');
    BOOST_TEST(result[4] == 'G');
    BOOST_TEST(result[5] == 'C');
    BOOST_TEST(result[6] == 'F');
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  PreOrderTraversalRecursiveTraversesMoreTrees,
  DynamicTreeNodeFixture)
{
  const auto result = preorder_traversal_recursive(root_ptr_);
  BOOST_TEST_REQUIRE(root_ptr_->size() == 14);
  BOOST_TEST_REQUIRE(result.size() == 14);
  BOOST_TEST(result[0] == 'A');
  BOOST_TEST(result[1] == 'B');
  BOOST_TEST(result[2] == 'E');
  BOOST_TEST(result[3] == 'G');
  BOOST_TEST(result[4] == 'F');
  BOOST_TEST(result[5] == 'C');
  BOOST_TEST(result[6] == 'H');
  BOOST_TEST(result[7] == 'I');
  BOOST_TEST(result[8] == 'J');
  BOOST_TEST(result[9] == 'D');
  BOOST_TEST(result[10] == 'K');
  BOOST_TEST(result[11] == 'L');
  BOOST_TEST(result[12] == 'M');
  BOOST_TEST(result[13] == 'N');
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  PostOrderTraversalRecursiveGetsNodesVisitedLastAfterChildrenVisited,
  MoreDynamicTreeNodeFixture)
{
  {
    const auto result = postorder_traversal_recursive(root_ptr1_);
    BOOST_TEST_REQUIRE(root_ptr1_->size() == 13);
    BOOST_TEST_REQUIRE(result.size() == 13);
    BOOST_TEST(result[0] == 'D');
    BOOST_TEST(result[1] == 'C');
    BOOST_TEST(result[2] == 'F');
    BOOST_TEST(result[3] == 'G');
    BOOST_TEST(result[4] == 'E');
    BOOST_TEST(result[5] == 'B');
    BOOST_TEST(result[6] == 'J');
    BOOST_TEST(result[7] == 'K');
    BOOST_TEST(result[8] == 'L');
    BOOST_TEST(result[9] == 'I');
    BOOST_TEST(result[10] == 'M');
    BOOST_TEST(result[11] == 'H');
    BOOST_TEST(result[12] == 'A');
  }
  {
    const auto result = postorder_traversal_recursive(root_ptr2_);
    BOOST_TEST_REQUIRE(root_ptr2_->size() == 7);
    BOOST_TEST_REQUIRE(result.size() == 7);
    BOOST_TEST(result[0] == 'D');
    BOOST_TEST(result[1] == 'G');
    BOOST_TEST(result[2] == 'E');
    BOOST_TEST(result[3] == 'B');
    BOOST_TEST(result[4] == 'F');
    BOOST_TEST(result[5] == 'C');
    BOOST_TEST(result[6] == 'A');
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  PostOrderTraversalRecursiveTraversesMoreTrees,
  DynamicTreeNodeFixture)
{
  const auto result = postorder_traversal_recursive(root_ptr_);
  BOOST_TEST_REQUIRE(root_ptr_->size() == 14);
  BOOST_TEST_REQUIRE(result.size() == 14);
  BOOST_TEST(result[0] == 'G');
  BOOST_TEST(result[1] == 'E');
  BOOST_TEST(result[2] == 'F');
  BOOST_TEST(result[3] == 'B');
  BOOST_TEST(result[4] == 'I');
  BOOST_TEST(result[5] == 'J');
  BOOST_TEST(result[6] == 'H');
  BOOST_TEST(result[7] == 'C');
  BOOST_TEST(result[8] == 'K');
  BOOST_TEST(result[9] == 'L');
  BOOST_TEST(result[10] == 'N');
  BOOST_TEST(result[11] == 'M');
  BOOST_TEST(result[12] == 'D');
  BOOST_TEST(result[13] == 'A');
}

BOOST_AUTO_TEST_SUITE_END() // DynamicTreeNode_tests

BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures
