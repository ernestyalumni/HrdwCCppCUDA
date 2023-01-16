#include "DataStructures/Trees/SimpleTree.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(SimpleTree_tests)

BOOST_AUTO_TEST_SUITE(DWHarder)

using DataStructures::Trees::DWHarder::SimpleTree;

class SimpleTreeFixture
{
  public:

    SimpleTreeFixture():
      root_{'A'},
      child0_{'B', &root_},
      child1_{'H', &root_},
      gchild0_{'C', &child0_},
      gchild1_{'E', &child0_},
      gchild2_{'I', &child1_},
      gchild3_{'M', &child1_},
      ggchild0_{'D', &gchild0_},
      ggchild1_{'F', &gchild1_},
      ggchild2_{'G', &gchild1_},
      ggchild3_{'J', &gchild2_},
      ggchild4_{'K', &gchild2_},
      ggchild5_{'L', &gchild2_}
    {}

    ~SimpleTreeFixture() = default;

    SimpleTree<char> root_;
    SimpleTree<char> child0_;
    SimpleTree<char> child1_;
    // g stands for "grand".
    SimpleTree<char> gchild0_;
    SimpleTree<char> gchild1_;
    SimpleTree<char> gchild2_;
    SimpleTree<char> gchild3_;

    // The second g stands for "great-"
    SimpleTree<char> ggchild0_;
    SimpleTree<char> ggchild1_;
    SimpleTree<char> ggchild2_;
    SimpleTree<char> ggchild3_;
    SimpleTree<char> ggchild4_;
    SimpleTree<char> ggchild5_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsRoot)
{
  SimpleTree<char> root {'A'};
  BOOST_TEST(root.retrieve() == 'A');
  BOOST_TEST(root.degree() == 0);
  BOOST_TEST(root.is_root());
  BOOST_TEST(root.is_leaf());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(DepthGetsDepth, SimpleTreeFixture)
{
  BOOST_TEST(root_.depth() == 0);

  // 'E' has depth of 2, 'L' has depth of 3.
  BOOST_TEST(gchild1_.depth() == 2);
  BOOST_TEST(ggchild5_.depth() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(IterativeDepthGetsDepth, SimpleTreeFixture)
{
  BOOST_TEST(root_.iterative_depth() == 0);

  // 'E' has depth of 2, 'L' has depth of 3.
  BOOST_TEST(gchild1_.iterative_depth() == 2);
  BOOST_TEST(ggchild5_.iterative_depth() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(RootGetsRoot, SimpleTreeFixture)
{
  {
    BOOST_TEST(root_.root()->retrieve() == 'A');
    BOOST_TEST(root_.root()->is_root());
  }
  {
    BOOST_TEST(gchild0_.root()->retrieve() == 'A');
    BOOST_TEST(gchild0_.root()->is_root());
  }
  {
    BOOST_TEST(ggchild3_.root()->retrieve() == 'A');
    BOOST_TEST(ggchild3_.root()->is_root());
  }
}

BOOST_AUTO_TEST_SUITE_END() // DWHarder

/*
class SimpleTreeNodeFixture
{
  public:

    SimpleTreeNodeFixture():
      root_{'A'},
      child0_{'B', &root_},
      child1_{'H', &root_},
      gchild0_{'C', &child0_},
      gchild1_{'E', &child0_},
      gchild2_{'I', &child1_},
      gchild3_{'M', &child1_},
      ggchild0_{'D', &gchild0_},
      ggchild1_{'F', &gchild1_},
      ggchild2_{'G', &gchild1_},
      ggchild3_{'J', &gchild2_},
      ggchild4_{'K', &gchild2_},
      ggchild5_{'L', &gchild2_}
    {}

    ~SimpleTreeNodeFixture() = default;

    SimpleTreeNode<char> root_;
    SimpleTreeNode<char> child0_;
    SimpleTreeNode<char> child1_;
    // g stands for "grand".
    SimpleTreeNode<char> gchild0_;
    SimpleTreeNode<char> gchild1_;
    SimpleTreeNode<char> gchild2_;
    SimpleTreeNode<char> gchild3_;

    // The second g stands for "great-"
    SimpleTreeNode<char> ggchild0_;
    SimpleTreeNode<char> ggchild1_;
    SimpleTreeNode<char> ggchild2_;
    SimpleTreeNode<char> ggchild3_;
    SimpleTreeNode<char> ggchild4_;
    SimpleTreeNode<char> ggchild5_;
};

BOOST_AUTO_TEST_SUITE(SimpleTreeNode_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsRoot)
{
  SimpleTreeNode<char> root {'A'};

  BOOST_TEST(root.value() == 'A');
  BOOST_TEST(root.degree() == 0);
  BOOST_TEST(root.parent() == nullptr);
  BOOST_TEST(root.is_root());
  BOOST_TEST(root.is_leaf());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddChildAddsChildren)
{
  SimpleTreeNode<char> root {'A'};
  SimpleTreeNode<char> child1 {'B', &root};
  SimpleTreeNode<char> child2 {'H', &root};

  root.add_child(&child1);
  root.add_child(&child2);

  // TODO: Determine recursive way to clean up new memory allocations.
  //root.add_child('B');
  //root.add_child('H');
  BOOST_TEST(root.degree() == 2);
  BOOST_TEST(!root.is_leaf());
  BOOST_TEST(root.child(0)->value() == 'B');
  BOOST_TEST(root.child(0)->degree() == 0);
  BOOST_TEST(root.child(0)->parent()->value() == 'A');
  BOOST_TEST(root.child(0)->is_leaf());
  BOOST_TEST(root.child(1)->is_leaf());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddChildCanAddDynamicallyAllocatedChildren)
{
  SimpleTreeNode<char>* root_ptr {new SimpleTreeNode<char>('A')};
  SimpleTreeNode<char>* child_ptr1 {new SimpleTreeNode<char>('B', root_ptr)};
  SimpleTreeNode<char>* child_ptr2 {new SimpleTreeNode<char>('H', root_ptr)};

  root_ptr->add_child(child_ptr1);
  root_ptr->add_child(child_ptr2);

  BOOST_TEST(root_ptr->degree() == 2);
  BOOST_TEST(!root_ptr->is_leaf());
  BOOST_TEST(root_ptr->child(0)->value() == 'B');
  BOOST_TEST(root_ptr->child(0)->degree() == 0);
  BOOST_TEST(root_ptr->child(0)->parent()->value() == 'A');
  BOOST_TEST(root_ptr->child(0)->is_leaf());
  BOOST_TEST(root_ptr->child(1)->is_leaf());

  delete child_ptr1;
  delete child_ptr2;
  delete root_ptr;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(DepthGetsDepth, SimpleTreeNodeFixture)
{
  BOOST_TEST(root_.depth() == 0);

  // 'E' has depth of 2, 'L' has depth of 3.
  BOOST_TEST(gchild1_.depth() == 2);
  BOOST_TEST(ggchild5_.depth() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(IterativeDepthGetsDepth, SimpleTreeNodeFixture)
{
  BOOST_TEST(root_.iterative_depth() == 0);

  // 'E' has depth of 2, 'L' has depth of 3.
  BOOST_TEST(gchild1_.iterative_depth() == 2);
  BOOST_TEST(ggchild5_.iterative_depth() == 3);
}

BOOST_AUTO_TEST_SUITE_END() // SimpleTreeNode_tests
*/

BOOST_AUTO_TEST_SUITE_END() // SimpleTree_tests
BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures
