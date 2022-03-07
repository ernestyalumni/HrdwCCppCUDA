#include "DataStructures/Trees/BinaryTrees/BST.h"

#include <boost/test/unit_test.hpp>

template <typename T>
using BST = DataStructures::Trees::BinaryTrees::ExpertIO::BST<T>;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Trees)
BOOST_AUTO_TEST_SUITE(BinaryTrees)

BOOST_AUTO_TEST_SUITE(BST_tests)

// Running
// valgrind ./Check --run_test=DataStructures/Trees/
// Shows that this test has memory leaks on exit.
/*
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInsertRecursivelyAlgoexpert)
{
  {
    BST<int> sample {10};
    sample.insert_recursive_algoexpert(5);
    sample.insert_recursive_algoexpert(2);
    sample.insert_recursive_algoexpert(5);
    sample.insert_recursive_algoexpert(15);
    sample.insert_recursive_algoexpert(13);
    sample.insert_recursive_algoexpert(22);
    sample.insert_recursive_algoexpert(14);
    sample.insert_recursive_algoexpert(1);
  }
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
  BST<int>* sample_ptr {new BST<int>{10}};
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInsertIteratively)
{
  /*
  {
    BST<int> sample {10};
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
    BST<int>* sample_ptr {new BST<int>{10}};
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