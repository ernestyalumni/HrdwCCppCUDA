#include "DataStructures/Trees/BinaryTrees/BinaryNode.h"

#include <boost/test/unit_test.hpp>

template <typename T>
using DWHarderBinaryNode =
  DataStructures::Trees::BinaryTrees::DWHarder::BinaryNode<T>;

using DataStructures::Trees::BinaryTrees::Node;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(BinaryTrees)

BOOST_AUTO_TEST_SUITE(BinaryNode_tests)

class NodeFixture
{
  public:

    NodeFixture():
      x_01_{1},
      x_02_{2},
      x_03_{3},
      x_04_{4},
      x_05_{5},
      x_06_{6},
      x_07_{7},
      x_08_{8},
      x_09_{9, &x_01_},
      x_10_{10, nullptr, &x_02_},
      x_11_{11, &x_03_, &x_04_},
      x_12_{12, &x_05_},
      x_13_{13, &x_06_, &x_07_},
      x_14_{14, &x_09_, &x_08_},
      x_15_{15, &x_10_},
      x_16_{16, nullptr, &x_11_},
      x_17_{17, &x_12_, &x_13_},
      x_18_{18, &x_14_, &x_15_},
      x_19_{19, nullptr, &x_16_},
      x_20_{20, &x_18_, &x_19_},
      x_21_{21, &x_20_, &x_17_}
    {}

    virtual ~NodeFixture() = default;

    Node<int> x_01_;
    Node<int> x_02_;
    Node<int> x_03_;
    Node<int> x_04_;
    Node<int> x_05_;
    Node<int> x_06_;
    Node<int> x_07_;
    Node<int> x_08_;
    Node<int> x_09_;
    Node<int> x_10_;
    Node<int> x_11_;
    Node<int> x_12_;
    Node<int> x_13_;
    Node<int> x_14_;
    Node<int> x_15_;
    Node<int> x_16_;
    Node<int> x_17_;
    Node<int> x_18_;
    Node<int> x_19_;
    Node<int> x_20_;
    Node<int> x_21_;
};

BOOST_AUTO_TEST_SUITE(DWHarder)

//------------------------------------------------------------------------------
/// \ref Exercise 10.4-1, Ch. 10 Elementary Data Structures, Cormen, Leiserson,
/// Rivest, Stein.
//------------------------------------------------------------------------------
class BinaryNodeFixture
{
  public:

    BinaryNodeFixture():
      a_06_{new DWHarderBinaryNode<int>{18}},
      a_01_{new DWHarderBinaryNode<int>{12}},
      a_04_{new DWHarderBinaryNode<int>{10}},
      a_03_{new DWHarderBinaryNode<int>{4}},
      a_07_{new DWHarderBinaryNode<int>{7}},
      a_10_{new DWHarderBinaryNode<int>{5}},
      a_05_{new DWHarderBinaryNode<int>{2}},
      a_09_{new DWHarderBinaryNode<int>{21}}
    {
      a_06_->set_left(a_01_);
      a_06_->set_right(a_04_);
      a_01_->set_left(a_07_);
      a_01_->set_right(a_03_);
      a_03_->set_left(a_10_);
      a_04_->set_left(a_05_);
      a_04_->set_right(a_09_);
    }

    virtual ~BinaryNodeFixture()
    {
      // User defined to be empty.
    }

    DWHarderBinaryNode<int>* a_06_;
    DWHarderBinaryNode<int>* a_01_;
    DWHarderBinaryNode<int>* a_04_;
    DWHarderBinaryNode<int>* a_03_;
    DWHarderBinaryNode<int>* a_07_;
    DWHarderBinaryNode<int>* a_10_;
    DWHarderBinaryNode<int>* a_05_;
    DWHarderBinaryNode<int>* a_09_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithValue)
{
  DWHarderBinaryNode<int> a {42};
  BOOST_TEST(a.is_leaf());
  BOOST_TEST(a.value() == 42);
  BOOST_TEST(a.left() == nullptr);
  BOOST_TEST(a.right() == nullptr);
}

//------------------------------------------------------------------------------
/// \ref Exercise 10.4-1, Ch. 10 Elementary Data Structures, Cormen, Leiserson,
/// Rivest, Stein.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DynamicallyConstructBinaryNodes)
{
  DWHarderBinaryNode<int>* a_06 {new DWHarderBinaryNode<int>{18}};
  DWHarderBinaryNode<int>* a_01 {new DWHarderBinaryNode<int>{12}};
  DWHarderBinaryNode<int>* a_04 {new DWHarderBinaryNode<int>{10}};
  a_06->set_left(a_01);
  a_06->set_right(a_04);
  DWHarderBinaryNode<int>* a_03 {new DWHarderBinaryNode<int>{4}};
  DWHarderBinaryNode<int>* a_07 {new DWHarderBinaryNode<int>{7}};
  a_01->set_left(a_07);
  a_01->set_right(a_03);
  DWHarderBinaryNode<int>* a_10 {new DWHarderBinaryNode<int>{5}};
  a_03->set_left(a_10);
  DWHarderBinaryNode<int>* a_05 {new DWHarderBinaryNode<int>{2}};
  DWHarderBinaryNode<int>* a_09 {new DWHarderBinaryNode<int>{21}};
  a_04->set_left(a_05);
  a_04->set_right(a_09);

  BOOST_TEST(a_06->value() == 18);
  BOOST_TEST(a_06->left()->value() == 12);
  BOOST_TEST(a_06->right()->value() == 10);
  BOOST_TEST(a_06->left()->left()->value() == 7);
  BOOST_TEST(a_06->left()->right()->value() == 4);
  BOOST_TEST(a_06->left()->right()->left()->value() == 5);
  BOOST_TEST(a_06->right()->left()->value() == 2);
  BOOST_TEST(a_06->right()->right()->value() == 21);

  delete a_06;
  delete a_01;
  delete a_04;
  delete a_03;
  delete a_07;
  delete a_10;
  delete a_05;
  delete a_09;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DynamicallyClearBinaryNodes)
{
  DWHarderBinaryNode<int>* a_06 {new DWHarderBinaryNode<int>{18}};
  DWHarderBinaryNode<int>* a_01 {new DWHarderBinaryNode<int>{12}};
  DWHarderBinaryNode<int>* a_04 {new DWHarderBinaryNode<int>{10}};
  a_06->set_left(a_01);
  a_06->set_right(a_04);
  DWHarderBinaryNode<int>* a_03 {new DWHarderBinaryNode<int>{4}};
  DWHarderBinaryNode<int>* a_07 {new DWHarderBinaryNode<int>{7}};
  a_01->set_left(a_07);
  a_01->set_right(a_03);
  DWHarderBinaryNode<int>* a_10 {new DWHarderBinaryNode<int>{5}};
  a_03->set_left(a_10);
  DWHarderBinaryNode<int>* a_05 {new DWHarderBinaryNode<int>{2}};
  DWHarderBinaryNode<int>* a_09 {new DWHarderBinaryNode<int>{21}};
  a_04->set_left(a_05);
  a_04->set_right(a_09);

  a_06->clear();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ClearBinaryNodesFromFunction, BinaryNodeFixture)
{
  a_06_->clear();
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // DWHarder

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  Node<int> a {};
  BOOST_TEST(a.is_leaf());
  BOOST_TEST(a.value_ == 0);
  BOOST_TEST(a.left_ == nullptr);
  BOOST_TEST(a.right_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithReferences)
{
  Node<int> a_01 {1};
  BOOST_TEST(a_01.is_leaf());

  Node<int> a_03 {3};
  BOOST_TEST(a_03.is_leaf());

  Node<int> a_02 {2, &a_01};
  BOOST_TEST(!a_02.is_leaf());

  BOOST_TEST(a_02.right_ == nullptr);
  BOOST_TEST(a_02.left_->value_ == 1);

  // a_01 is still a leaf.
  BOOST_TEST(a_01.is_leaf());

  Node<int> a_04 {2, nullptr, &a_03};

  BOOST_TEST(a_04.left_ == nullptr);
  BOOST_TEST(a_04.right_->value_ == 3);

  // a_03 is still a leaf.
  BOOST_TEST(a_03.is_leaf());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithDynamicAllocation)
{
  Node<int>* a_01 {new Node<int>{1}};
  BOOST_TEST(a_01->is_leaf());

  Node<int>* a_02 {new Node<int>{2}};
  BOOST_TEST(a_02->is_leaf());

  Node<int>* a_03 {new Node<int>{3, a_01, a_02}};
  BOOST_TEST(!a_03->is_leaf());

  delete a_01;
  delete a_02;
  delete a_03;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(CanCreateTree, NodeFixture)
{
  BOOST_TEST(x_01_.is_leaf());
  BOOST_TEST(x_02_.is_leaf());
  BOOST_TEST(x_03_.is_leaf());
  BOOST_TEST(x_04_.is_leaf());
  BOOST_TEST(x_05_.is_leaf());
  BOOST_TEST(x_06_.is_leaf());
  BOOST_TEST(x_07_.is_leaf());
  BOOST_TEST(x_08_.is_leaf());
  BOOST_TEST(!x_09_.is_leaf());
  BOOST_TEST(x_09_.left_->value_ == 1);
  BOOST_TEST(x_09_.right_ == nullptr);
  BOOST_TEST(!x_10_.is_leaf());
  BOOST_TEST(x_10_.left_ == nullptr);
  BOOST_TEST(x_10_.right_->value_ == 2);
  BOOST_TEST(!x_11_.is_leaf());
  BOOST_TEST(x_11_.left_->value_ == 3);
  BOOST_TEST(x_11_.right_->value_ == 4);
  BOOST_TEST(!x_12_.is_leaf());
  BOOST_TEST(x_12_.left_->value_ == 5);
  BOOST_TEST(x_12_.right_ == nullptr);
  BOOST_TEST(!x_13_.is_leaf());
  BOOST_TEST(x_13_.left_->value_ == 6);
  BOOST_TEST(x_13_.right_->value_ == 7);
  BOOST_TEST(!x_14_.is_leaf());
  BOOST_TEST(x_14_.left_->value_ == 9);
  BOOST_TEST(x_14_.left_->left_->value_ == 1);
  BOOST_TEST(x_14_.right_->value_ == 8);
  BOOST_TEST(!x_15_.is_leaf());
  BOOST_TEST(x_15_.left_->value_ == 10);
  BOOST_TEST(x_15_.left_->right_->value_ == 2);
  BOOST_TEST(x_15_.right_ == nullptr);
  BOOST_TEST(!x_16_.is_leaf());
  BOOST_TEST(x_16_.right_->value_ == 11);
  BOOST_TEST(x_16_.right_->left_->value_ == 3);
  BOOST_TEST(x_16_.right_->right_->value_ == 4);
  BOOST_TEST(!x_17_.is_leaf());
  BOOST_TEST(x_17_.left_->value_ == 12);
  BOOST_TEST(x_17_.right_->value_ == 13);
  BOOST_TEST(x_17_.right_->left_->value_ == 6);
  BOOST_TEST(x_17_.right_->right_->value_ == 7);
  BOOST_TEST(!x_18_.is_leaf());
  BOOST_TEST(x_18_.left_->value_ == 14);
  BOOST_TEST(x_18_.right_->value_ == 15);
  BOOST_TEST(x_18_.left_->left_->value_ == 9);
  BOOST_TEST(x_18_.right_->left_->value_ == 10);
  BOOST_TEST(!x_19_.is_leaf());
  BOOST_TEST(x_19_.left_ == nullptr);
  BOOST_TEST(x_19_.right_->value_ == 16);
  BOOST_TEST(x_19_.right_->right_->value_ == 11);
  BOOST_TEST(x_19_.right_->right_->left_->value_ == 3);
  BOOST_TEST(!x_20_.is_leaf());
  BOOST_TEST(x_20_.left_->value_ == 18);
  BOOST_TEST(x_20_.right_->value_ == 19);
  BOOST_TEST(x_20_.left_->left_->value_ == 14);
  BOOST_TEST(x_20_.right_->right_->value_ == 16);
  BOOST_TEST(!x_21_.is_leaf());
  BOOST_TEST(x_21_.left_->value_ == 20);
  BOOST_TEST(x_21_.right_->value_ == 17);
  BOOST_TEST(x_21_.left_->left_->value_ == 18);
  BOOST_TEST(x_21_.right_->right_->value_ == 13);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SizeWorksOnLeaves, NodeFixture)
{
  BOOST_TEST(x_01_.size() == 1);
  BOOST_TEST(x_02_.size() == 1);
  BOOST_TEST(x_03_.size() == 1);
  BOOST_TEST(x_04_.size() == 1);
  BOOST_TEST(x_05_.size() == 1);
  BOOST_TEST(x_06_.size() == 1);
  BOOST_TEST(x_07_.size() == 1);
  BOOST_TEST(x_08_.size() == 1);
  BOOST_TEST(x_09_.size() == 2);
  BOOST_TEST(x_11_.size() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SizeWorksOnSubTrees, NodeFixture)
{
  BOOST_TEST(x_20_.size() == 14);
  BOOST_TEST(x_17_.size() == 6);
  BOOST_TEST(x_21_.size() == 21);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(HeightWorksOnSubTrees, NodeFixture)
{
  BOOST_TEST(x_20_.height() == 4);
  BOOST_TEST(x_17_.height() == 2);
  BOOST_TEST(x_21_.height() == 5);
}

BOOST_AUTO_TEST_SUITE_END() // BinaryNode_tests

BOOST_AUTO_TEST_SUITE_END() // BinaryTrees
BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures