#include "DataStructures/Trees/SimpleTreeNode.h"

#include "Tools/CaptureCout.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using DataStructures::Trees::SimpleTreeNode;
using Tools::CaptureCoutFixture;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(SimpleTreeNode_tests)

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
    {
      root_.add_child(&child0_);
      root_.add_child(&child1_);
      child0_.add_child(&gchild0_);
      child0_.add_child(&gchild1_);
      child1_.add_child(&gchild2_);
      child1_.add_child(&gchild3_);
      gchild0_.add_child(&ggchild0_);
      gchild1_.add_child(&ggchild1_);
      gchild1_.add_child(&ggchild2_);
      gchild2_.add_child(&ggchild3_);
      gchild2_.add_child(&ggchild4_);
      gchild2_.add_child(&ggchild5_);
    }

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(BreadthFirstSearchFindsValues, SimpleTreeNodeFixture)
{
  BOOST_TEST(root_.breadth_first_search('A') != nullptr);
  BOOST_TEST(root_.breadth_first_search('H') != nullptr);
  BOOST_TEST(root_.breadth_first_search('E') != nullptr);
  BOOST_TEST(root_.breadth_first_search('L') != nullptr);

  BOOST_TEST(child0_.breadth_first_search('B') != nullptr);
  BOOST_TEST(child0_.breadth_first_search('E') != nullptr);
  BOOST_TEST(child0_.breadth_first_search('D') != nullptr);
  BOOST_TEST(child0_.breadth_first_search('F') != nullptr);

  BOOST_TEST(child1_.breadth_first_search('H') != nullptr);
  BOOST_TEST(child1_.breadth_first_search('M') != nullptr);
  BOOST_TEST(child1_.breadth_first_search('K') != nullptr);
  BOOST_TEST(child1_.breadth_first_search('L') != nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  BreadthFirstSearchReturnsNullPtrForUnfoundValues,
  SimpleTreeNodeFixture)
{
  BOOST_TEST(root_.breadth_first_search('Z') == nullptr);
  BOOST_TEST(child1_.breadth_first_search('A') == nullptr);
  BOOST_TEST(gchild0_.breadth_first_search('B') == nullptr);
  BOOST_TEST(gchild3_.breadth_first_search('H') == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(BreadthFirstTraversal, CaptureCoutFixture)
{
  SimpleTreeNodeFixture fix {};

  fix.root_.breadth_first_traversal();

  BOOST_TEST(local_oss_.str() == "A, B, H, C, E, I, M, D, F, G, J, K, L, ");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(PreorderTraversalVisitsFirst, SimpleTreeNodeFixture)
{
  vector<char> result {preorder_traversal(&root_)};

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(PostorderTraversalVisitsLast, SimpleTreeNodeFixture)
{
  vector<char> result {postorder_traversal_recursive(&root_)};

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  PostorderTraversalOfNodesVisitsAll,
  SimpleTreeNodeFixture)
{
  SimpleTreeNode<char>::Queue<SimpleTreeNode<char>*> queue {};
  postorder_traversal_of_nodes_recursive_step(&root_, queue);
  BOOST_TEST(queue.dequeue()->value() == 'D');
  BOOST_TEST(queue.dequeue()->value() == 'C');
  BOOST_TEST(queue.dequeue()->value() == 'F');
  BOOST_TEST(queue.dequeue()->value() == 'G');
  BOOST_TEST(queue.dequeue()->value() == 'E');
  BOOST_TEST(queue.dequeue()->value() == 'B');
  BOOST_TEST(queue.dequeue()->value() == 'J');
  BOOST_TEST(queue.dequeue()->value() == 'K');
  BOOST_TEST(queue.dequeue()->value() == 'L');
  BOOST_TEST(queue.dequeue()->value() == 'I');
  BOOST_TEST(queue.dequeue()->value() == 'M');
  BOOST_TEST(queue.dequeue()->value() == 'H');
  BOOST_TEST(queue.dequeue()->value() == 'A');
}

BOOST_AUTO_TEST_SUITE_END() // SimpleTreeNode_tests

BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures
