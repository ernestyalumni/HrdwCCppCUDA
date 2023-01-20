#include "DataStructures/Trees/BinaryTrees/BinarySearchNode.h"

#include <boost/test/unit_test.hpp>
#include <bitset>
#include <cstddef>

using namespace DataStructures::Trees::BinaryTrees;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(BinaryTrees)

BOOST_AUTO_TEST_SUITE(DWHarder)

class DWHarderBinarySearchNodeFixture
{
  public:

    //--------------------------------------------------------------------------
    /// \ref 6.1.2 Binary Search Trees, 6.01.Binary_search_trees.pptx
    //--------------------------------------------------------------------------
    DWHarderBinarySearchNodeFixture():
      a_{new DWHarder::BinarySearchNode<int>{29}}
    {
      a_->insert(15);
      a_->insert(73);
      a_->insert(3);
      a_->insert(23);
      a_->insert(22);
      a_->insert(59);
      a_->insert(88);
      a_->insert(46);
      a_->insert(65);
      a_->insert(91);
      a_->insert(42);
      a_->insert(50);
      a_->insert(40);
      a_->insert(57);
    }

    ~DWHarderBinarySearchNodeFixture()
    {
      a_->clear();
    }

    DWHarder::BinarySearchNode<int>* a_;
};

BOOST_AUTO_TEST_SUITE(BinarySearchNode_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithValue)
{
  DWHarder::BinarySearchNode<char> a {'a'};
  BOOST_TEST(a.left() == nullptr); 
  BOOST_TEST(a.right() == nullptr); 
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertBuildsTree)
{
  //----------------------------------------------------------------------------
  /// \ref 6.1.2 Binary Search Trees, 6.01.Binary_sesarch_trees.pptx
  //----------------------------------------------------------------------------
  DWHarder::BinarySearchNode<int>* a {new DWHarder::BinarySearchNode<int>(42)};
  a->insert(15);
  a->insert(3);
  a->insert(22);
  a->insert(40);
  a->insert(29);
  a->insert(23);

  a->insert(50);
  a->insert(65);
  a->insert(91);
  a->insert(46);
  a->insert(88);
  a->insert(57);
  a->insert(73);
  a->insert(59);

  a->clear();

  BOOST_TEST(('O' < 'W'));

  BOOST_TEST((std::bitset<8>("10000").to_ulong() >
    std::bitset<8>("1000").to_ulong()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(LeftGetsLeft, DWHarderBinarySearchNodeFixture)
{
  BOOST_TEST(a_->value() == 29);
  BOOST_TEST(a_->left()->value() == 15);
  BOOST_TEST(a_->left()->left()->value() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(RightGetsRight, DWHarderBinarySearchNodeFixture)
{
  BOOST_TEST(a_->left()->right()->value() == 23);
  BOOST_TEST(a_->left()->right()->left()->value() == 22);
  BOOST_TEST(a_->right()->value() == 73);
  BOOST_TEST(a_->right()->left()->value() == 59);
  BOOST_TEST(a_->right()->right()->value() == 88);
  BOOST_TEST(a_->right()->right()->right()->value() == 91);
  BOOST_TEST(a_->right()->left()->right()->value() == 65);
  BOOST_TEST(a_->right()->left()->left()->value() == 46);
  BOOST_TEST(a_->right()->left()->left()->left()->value() == 42);
  BOOST_TEST(a_->right()->left()->left()->right()->value() == 50);
  BOOST_TEST(a_->right()->left()->left()->left()->left()->value() == 40);
  BOOST_TEST(a_->right()->left()->left()->right()->right()->value() == 57);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(FrontGetsSmallestValue, DWHarderBinarySearchNodeFixture)
{
  BOOST_TEST(a_->front() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(BackGetsLargestValue, DWHarderBinarySearchNodeFixture)
{
  BOOST_TEST(a_->back() == 91);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(NextFindsNext, DWHarderBinarySearchNodeFixture)
{
  BOOST_TEST(a_->next(29) == 40);
  BOOST_TEST(a_->next(30) == 40);
  BOOST_TEST(a_->next(72) == 73);
  BOOST_TEST(a_->next(14) == 15);
  BOOST_TEST(a_->next(21) == 22);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  FindReturnsFalseForMissingValue,
  DWHarderBinarySearchNodeFixture)
{
  BOOST_TEST(!a_->find(52));
  BOOST_TEST(a_->insert(52));
  BOOST_TEST(a_->find(52));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(SizeGetsSize, DWHarderBinarySearchNodeFixture)
{
  BOOST_TEST(a_->size() == 15);
  BOOST_TEST(a_->left()->size() == 4);
  BOOST_TEST(a_->right()->size() == 10);
  BOOST_TEST(a_->right()->right()->size() == 2);
}

BOOST_AUTO_TEST_SUITE_END() // BinarySearchNode_tests

BOOST_AUTO_TEST_SUITE_END() // DWHarder

BOOST_AUTO_TEST_SUITE_END() // BinaryTrees
BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures