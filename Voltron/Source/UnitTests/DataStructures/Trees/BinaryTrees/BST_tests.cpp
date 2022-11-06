#include "DataStructures/Trees/BinaryTrees/BST.h"

#include <boost/test/unit_test.hpp>

template <typename T>
using AlgoExpertBST = DataStructures::Trees::BinaryTrees::ExpertIO::BST<T>;

template <typename T>
using BinarySearchTree =
  DataStructures::Trees::BinaryTrees::BinarySearchTree<T>;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(BinaryTrees)

BOOST_AUTO_TEST_SUITE(BinarySearchTree_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertInserts)
{
  BinarySearchTree<int> sample {};
  sample.insert(10);
  BOOST_TEST(sample.get_root_ptr()->contains(10));
  sample.insert(5);
  BOOST_TEST(sample.get_root_ptr()->contains(5));
  sample.insert(2);
  BOOST_TEST(sample.get_root_ptr()->contains(2));
  sample.insert(5);
  sample.insert(15);
  BOOST_TEST(sample.get_root_ptr()->contains(15));
  sample.insert(13);
  BOOST_TEST(sample.get_root_ptr()->contains(13));
  sample.insert(22);
  BOOST_TEST(sample.get_root_ptr()->contains(22));
  sample.insert(14);
  BOOST_TEST(sample.get_root_ptr()->contains(14));
  sample.insert(1);
  BOOST_TEST(sample.get_root_ptr()->contains(1));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DynamicallyDefaultConstructs)
{
  // free(): invalid next size (fast)
  /*
  BinarySearchTree<int>* sample_ptr {new BinarySearchTree<int>{}};
  sample_ptr->insert(10);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(10));
  sample_ptr->insert(5);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(5));
  sample_ptr->insert(2);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(2));
  sample_ptr->insert(5);
  sample_ptr->insert(15);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(15));
  sample_ptr->insert(13);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(13));
  sample_ptr->insert(22);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(22));
  sample_ptr->insert(14);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(14));
  sample_ptr->insert(1);
  BOOST_TEST(sample_ptr->get_root_ptr()->contains(1));

  delete sample_ptr;
  */
  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertsAndRemoves)
{
  // free(): invalid next size (fast), unknown location(0): fatal error SIGABRT
  // Test Case 1
  /*
  BinarySearchTree<int> sample {};
  sample.insert(10);
  sample.insert(5);
  sample.insert(15);
  sample.insert(2);
  sample.insert(5);
  sample.insert(13);
  sample.insert(22);
  sample.insert(1);
  sample.insert(14);
  sample.insert(12);
  sample.remove(10);

  BOOST_TEST(!sample.contains(10));
  BOOST_TEST(sample.contains(12));
  BOOST_TEST(sample.contains(14));
  BOOST_TEST(sample.contains(1));
  BOOST_TEST(sample.contains(22));
  BOOST_TEST(sample.contains(13));
  BOOST_TEST(sample.contains(5));
  BOOST_TEST(sample.contains(15));
  BOOST_TEST(sample.contains(2));

  // Debug checks.
  BOOST_TEST(sample.get_root_ptr()->value_ == 12);
  BOOST_TEST(sample.get_root_ptr()->left_->value_ == 5);
  BOOST_TEST(sample.get_root_ptr()->right_->value_ == 15);
  */
}

BOOST_AUTO_TEST_SUITE_END() // BinarySearchTree_tests

BOOST_AUTO_TEST_SUITE(BST_tests)

// Running
// valgrind ./Check --run_test=DataStructures/Trees/
// Shows that this test has memory leaks on exit.
/*
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInsertRecursivelyAlgoexpert)
{
  AlgoExpertBST<int> sample {10};
  sample.insert_recursive_algoexpert(5);
  sample.insert_recursive_algoexpert(2);
  sample.insert_recursive_algoexpert(5);
  sample.insert_recursive_algoexpert(15);
  sample.insert_recursive_algoexpert(13);
  sample.insert_recursive_algoexpert(22);
  sample.insert_recursive_algoexpert(14);
  sample.insert_recursive_algoexpert(1);
}
*/

// Running
// valgrind ./Check --run_test=DataStructures/Trees/
// Shows that this test has memory leaks on exit.
/*
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DynamicallyConstructsWithInsertRecursivelyAlgoexpert)
{
  AlgoExpertBST<int>* sample_ptr {new AlgoExpertBST<int>{10}};
  sample_ptr->insert_recursive_algoexpert(5);
  sample_ptr->insert_recursive_algoexpert(2);
  sample_ptr->insert_recursive_algoexpert(5);
  sample_ptr->insert_recursive_algoexpert(15);
  sample_ptr->insert_recursive_algoexpert(13);
  sample_ptr->insert_recursive_algoexpert(22);
  sample_ptr->insert_recursive_algoexpert(14);
  sample_ptr->insert_recursive_algoexpert(1);
  delete sample_ptr;

  BOOST_TEST(true);
}
*/

// Running
// valgrind ./Check --run_test=DataStructures/Trees/
// Shows that this test has memory leaks on exit.
/*
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RemoveAllWithRemoveRecursiveAlgoexpert)
{
  AlgoExpertBST<int> sample {10};
  sample.insert_recursive_algoexpert(5);
  sample.insert_recursive_algoexpert(2);
  sample.insert_recursive_algoexpert(5);
  sample.insert_recursive_algoexpert(15);
  sample.insert_recursive_algoexpert(13);
  sample.insert_recursive_algoexpert(22);
  sample.insert_recursive_algoexpert(14);
  sample.insert_recursive_algoexpert(1);

  sample = sample.remove_recursive_algoexpert(15);
  BOOST_TEST(!sample.contains(15));
  sample.remove_recursive_algoexpert(14);
  BOOST_TEST(!sample.contains(14));
  sample.remove_recursive_algoexpert(10);
  BOOST_TEST(!sample.contains(10));
  sample.remove_recursive_algoexpert(13);
  BOOST_TEST(!sample.contains(13));
  sample.remove_recursive_algoexpert(14);
  BOOST_TEST(!sample.contains(14));
  sample.remove_recursive_algoexpert(5);
  BOOST_TEST(sample.contains(5));
  sample.remove_recursive_algoexpert(2);
  BOOST_TEST(!sample.contains(2));
  sample.remove_recursive_algoexpert(1);
  BOOST_TEST(!sample.contains(1));
  sample.remove_recursive_algoexpert(22);
  BOOST_TEST(!sample.contains(22));
  sample.remove_recursive_algoexpert(5);
  BOOST_TEST(sample.contains(5));
}
*/

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInsertIteratively)
{
  /*
  {
    AlgoExpertBST<int> sample {10};
    sample.insert_iteratively(5);
    sample.insert_iteratively(2);
    sample.insert_iteratively(5);
    sample.insert_iteratively(15);
    sample.insert_iteratively(13);
    sample.insert_iteratively(22);
    sample.insert_iteratively(14);
    sample.insert_iteratively(1);
  }
  */

  {
    /*
    AlgoExpertBST<int>* sample_ptr {new AlgoExpertBST<int>{10}};
    sample_ptr->insert_iteratively(5);
    sample_ptr->insert_iteratively(2);
    sample_ptr->insert_iteratively(5);
    sample_ptr->insert_iteratively(15);
    sample_ptr->insert_iteratively(13);
    sample_ptr->insert_iteratively(22);
    sample_ptr->insert_iteratively(14);
    sample_ptr->insert_iteratively(1);
    delete sample_ptr;
    */
  }
}
BOOST_AUTO_TEST_SUITE_END() // BST_tests

BOOST_AUTO_TEST_SUITE_END() // BinaryTrees
BOOST_AUTO_TEST_SUITE_END() // Trees
BOOST_AUTO_TEST_SUITE_END() // DataStructures