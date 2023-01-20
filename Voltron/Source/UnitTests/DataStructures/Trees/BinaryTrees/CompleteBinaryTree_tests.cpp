#include "DataStructures/Trees/BinaryTrees/CompleteBinaryTree.h"

#include <boost/test/unit_test.hpp>
#include <cstddef>

using DataStructures::Trees::BinaryTrees::CompleteBinaryTree;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(BinaryTrees)

BOOST_AUTO_TEST_SUITE(CompleteBinaryTree_tests)

template <typename T, std::size_t N>
class TestCompleteBinaryTree : public CompleteBinaryTree<T, N>
{
  public:

    using CompleteBinaryTree<T, N>::CompleteBinaryTree;
    using CompleteBinaryTree<T, N>::find;
    using CompleteBinaryTree<T, N>::operator[];
};

class TestCompleteBinaryTreeFixture
{
  public:

    TestCompleteBinaryTreeFixture():
      t_{}
    {
      t_.push_back_no_check(3);
      t_.push_back_no_check(9);
      t_.push_back_no_check(5);
      t_.push_back_no_check(14);
      t_.push_back_no_check(10);
      t_.push_back_no_check(6);
      t_.push_back_no_check(8);
      t_.push_back_no_check(17);
      t_.push_back_no_check(15);
      t_.push_back_no_check(13);
      t_.push_back_no_check(23);
      t_.push_back_no_check(12);
    }

    ~TestCompleteBinaryTreeFixture() = default;

    TestCompleteBinaryTree<int, 12> t_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  CompleteBinaryTree<int, 69> t {};
  BOOST_TEST(t.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsDynamically)
{
  {
    CompleteBinaryTree<int, 16>* pcba {new CompleteBinaryTree<int, 16>{}};

    delete pcba;
  }
  {
    TestCompleteBinaryTree<int, 15>* pcba {
      new TestCompleteBinaryTree<int, 15>{}};
    pcba->push_back_no_check(5);
    pcba->push_back_no_check(32);
    pcba->push_back_no_check(8);
    pcba->push_back_no_check(42);
    pcba->push_back_no_check(36);
    pcba->push_back_no_check(12);
    pcba->push_back_no_check(9);
    pcba->push_back_no_check(44);
    pcba->push_back_no_check(87);
    pcba->push_back_no_check(54);
    pcba->push_back_no_check(39);
    pcba->push_back_no_check(53);
    BOOST_TEST(pcba->size() == 12);

    BOOST_TEST((*pcba)[1] == 5);
    BOOST_TEST((*pcba)[2] == 32);
    BOOST_TEST((*pcba)[3] == 8);

    delete pcba;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushBackNoCheckBuildsCompleteBinaryTree)
{
  CompleteBinaryTree<int, 12> t {};
  t.push_back_no_check(3);
  t.push_back_no_check(9);
  t.push_back_no_check(5);
  t.push_back_no_check(14);
  t.push_back_no_check(10);
  t.push_back_no_check(6);
  t.push_back_no_check(8);
  t.push_back_no_check(17);
  t.push_back_no_check(15);
  t.push_back_no_check(13);
  t.push_back_no_check(23);
  t.push_back_no_check(12);

  BOOST_TEST(t.size() == 12);
  BOOST_TEST(t.is_full());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(NodesFollowBreadthFirst, TestCompleteBinaryTreeFixture)
{
  BOOST_TEST(t_[1] == 3);
  BOOST_TEST(t_[2] == 9);
  BOOST_TEST(t_[3] == 5);
  BOOST_TEST(t_[4] == 14);
  BOOST_TEST(t_[5] == 10);
  BOOST_TEST(t_[6] == 6);
  BOOST_TEST(t_[7] == 8);
  BOOST_TEST(t_[8] == 17);
  BOOST_TEST(t_[9] == 15);
  BOOST_TEST(t_[10] == 13);
  BOOST_TEST(t_[11] == 23);
  BOOST_TEST(t_[12] == 12);
}

BOOST_AUTO_TEST_SUITE_END() // CompleteBinaryTree_tests

BOOST_AUTO_TEST_SUITE_END() // BinaryTrees
BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures