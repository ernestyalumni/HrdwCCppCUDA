//------------------------------------------------------------------------------
/// \file BinarySearchTrees_tests.cpp
/// \date 20201015 15, 17:21 complete
//------------------------------------------------------------------------------
#include "DataStructures/BinarySearchTrees.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using DataStructures::BinarySearchTrees::InorderBstIterator;
using DataStructures::BinarySearchTrees::NodeWithCounter;
using DataStructures::BinarySearchTrees::TreeNode;
using DataStructures::BinarySearchTrees::TreeWithCounter;
using DataStructures::BinarySearchTrees::inorder_traversal;
using DataStructures::BinarySearchTrees::insert_into_bst;
using DataStructures::BinarySearchTrees::is_valid_binary_search_tree;
using DataStructures::BinarySearchTrees::iterative_validate_binary_search_tree;
using DataStructures::BinarySearchTrees::search_bst;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(BinarySearchTrees_tests)

class StoreValues
{
  public:

    StoreValues() = default;

    int operator()(int value)
    {
      values_.emplace_back(value);
      return value;
    }

    vector<int> values() const
    {
      return values_;
    }

  private:

    vector<int> values_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InorderTraversalDoesDepthFirstTraversals)
{
  StoreValues value_storage;
  TreeNode node_d3_l1 {4};
  TreeNode node_d3_l2 {5};
  TreeNode node_d2_l2 {3};
  TreeNode node_d2_l1 {2, &node_d3_l1, &node_d3_l2};
  TreeNode node_d1 {1, &node_d2_l1, &node_d2_l2};

  inorder_traversal(&node_d1, value_storage);

  const auto result = value_storage.values();

  BOOST_TEST(result.size() == 5);
  BOOST_TEST(result.at(0) == 4);
  BOOST_TEST(result.at(1) == 2);
  BOOST_TEST(result.at(2) == 5);
  BOOST_TEST(result.at(3) == 1);
  BOOST_TEST(result.at(4) == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsValidBinarySearchTreeValidatesBSTs)
{
  {
    TreeNode node_d2_l1 {1};
    TreeNode node_d2_l2 {3};
    TreeNode node_d1_l1 {2, &node_d2_l1, &node_d2_l2};

    BOOST_TEST(is_valid_binary_search_tree(&node_d1_l1));
  }
  {
    TreeNode node_d3_l1 {3};
    TreeNode node_d3_l2 {6};
    TreeNode node_d2_l1 {1};
    TreeNode node_d2_l2 {4, &node_d3_l1, &node_d3_l2};
    TreeNode node_d1_l1 {5, &node_d2_l1, &node_d2_l2};

    BOOST_TEST(!is_valid_binary_search_tree(&node_d1_l1));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterativeValidateBinarySearchTreeValidatesBSTs)
{
  {
    TreeNode node_d2_l1 {1};
    TreeNode node_d2_l2 {3};
    TreeNode node_d1_l1 {2, &node_d2_l1, &node_d2_l2};

    BOOST_TEST(iterative_validate_binary_search_tree(&node_d1_l1));
  }
  {
    TreeNode node_d3_l1 {3};
    TreeNode node_d3_l2 {6};
    TreeNode node_d2_l1 {1};
    TreeNode node_d2_l2 {4, &node_d3_l1, &node_d3_l2};
    TreeNode node_d1_l1 {5, &node_d2_l1, &node_d2_l2};

    BOOST_TEST(!iterative_validate_binary_search_tree(&node_d1_l1));
  }
}

// cf. https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion/
TreeNode geek_example_node_d3_l1 {4};
TreeNode geek_example_node_d3_l2 {5};
TreeNode geek_example_node_d2_l1 {
  2,
  &geek_example_node_d3_l1,
  &geek_example_node_d3_l2};
TreeNode geek_example_node_d2_l2 {3};
TreeNode geek_example_root {
  1,
  &geek_example_node_d2_l1,
  &geek_example_node_d2_l2};

TreeNode example_node_d3_l1 {9};
TreeNode example_node_d3_l2 {20};
TreeNode example_node_d2_l1 {3};
TreeNode example_node_d2_l2 {15, &example_node_d3_l1, &example_node_d3_l2};
TreeNode example_root {7, &example_node_d2_l1, &example_node_d2_l2};

TreeNode example_root_1 {7, &example_node_d2_l1, &example_node_d3_l2};

// Binary Search Tree Iterator
/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/140/introduction-to-a-bst/1008/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InorderBstIteratorStepsThroughBST)
{
  {
    InorderBstIterator<TreeNode, int> iter {&geek_example_root};
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 4);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 2);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 5);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 1);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 3);
    BOOST_TEST(!iter.has_next());
  }
  {
    InorderBstIterator<TreeNode, int> iter {&example_root};
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 3);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 7);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 9);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 15);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 20);
    BOOST_TEST(!iter.has_next());
  }
  {
    InorderBstIterator<TreeNode, int> iter {&example_root_1};
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 3);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 7);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 20);
    BOOST_TEST(!iter.has_next());
  }

}

/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/141/basic-operations-in-a-bst/1000/
/// \brief Search in a Binary Search Tree.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SearchBstSearchesInABinarySearchTree)
{
  TreeNode example_node_d3_l1 {1};
  TreeNode example_node_d3_l2 {3};
  TreeNode example_node_d2_l1 {2, &example_node_d3_l1, &example_node_d3_l2};
  TreeNode example_node_d2_l2 {7};
  TreeNode example_root {4, &example_node_d2_l1, &example_node_d2_l2};

  {
    auto result = search_bst<TreeNode, int>(&example_root, 2);
    BOOST_TEST(result->value_ == 2);
    BOOST_TEST(result->left_->value_ == 1);
    BOOST_TEST(result->right_->value_ == 3);
  }
  {
    auto result = search_bst<TreeNode, int>(&example_root, 5);
    BOOST_TEST(result == nullptr);
  }
}

/// \url https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/141/basic-operations-in-a-bst/1003/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertIntoBst)
{
  {
    TreeNode example_node_d3_l1 {1};
    TreeNode example_node_d3_l2 {3};
    TreeNode example_node_d2_l1 {2, &example_node_d3_l1, &example_node_d3_l2};
    TreeNode example_node_d2_l2 {7};
    TreeNode example_root {4, &example_node_d2_l1, &example_node_d2_l2};

    auto result = insert_into_bst<TreeNode, int>(&example_root, 5);
    InorderBstIterator<TreeNode, int> iter {result};

    
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 1);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 2);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 3);
    BOOST_TEST(iter.has_next());
    BOOST_TEST(iter.next() == 4);
    BOOST_TEST(iter.has_next());
    //BOOST_TEST(iter.next() == 5);
    //BOOST_TEST(iter.has_next());
  }
}

BOOST_AUTO_TEST_SUITE(TreeWithCounter_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  TreeWithCounter tree {};
  BOOST_TEST(tree.root_ptr() == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertNewValueCreatesRootForEmptyTree)
{
  // Works
  TreeWithCounter* tree_ptr {new TreeWithCounter()};
  BOOST_TEST(tree_ptr->root_ptr() == nullptr);
  tree_ptr->insert_new_value(5);

  // Doesn't work.
//  BOOST_TEST(tree_ptr->root_ptr()->value_ == 5);

  // Does not work.
  //TreeWithCounter tree {};
  //BOOST_TEST(tree.root_ptr() == nullptr);
  //tree.insert_new_value(5); 

  BOOST_TEST(tree_ptr->is_counter_stack_empty());

  NodeWithCounter* check_root {tree_ptr->root_ptr()};

  BOOST_TEST(check_root->value_ == 5);
  BOOST_TEST(check_root->counter_ == 0);
  BOOST_TEST(check_root->left_ == nullptr);
  BOOST_TEST(check_root->right_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertNewValuePlacesLessThanValuesToTheLeft)
{
  TreeWithCounter* tree_ptr {new TreeWithCounter()};
  BOOST_TEST(tree_ptr->root_ptr() == nullptr);
  tree_ptr->insert_new_value(5);

  NodeWithCounter* check_root {tree_ptr->root_ptr()};
  BOOST_TEST(check_root->value_ == 5);

  tree_ptr->insert_new_value(4);

  BOOST_TEST(tree_ptr->is_counter_stack_empty());

  BOOST_TEST(check_root->value_ == 5);
  BOOST_TEST(check_root->counter_ == 1);
  BOOST_TEST(check_root->left_ != nullptr);
  BOOST_TEST(check_root->left_->value_ == 4);
  BOOST_TEST(check_root->left_->left_ == nullptr);
  BOOST_TEST(check_root->left_->right_ == nullptr);
  BOOST_TEST(check_root->left_->counter_ == 0);

  tree_ptr->insert_new_value(2);
  BOOST_TEST(tree_ptr->is_counter_stack_empty());

  BOOST_TEST(check_root->value_ == 5);
  BOOST_TEST(check_root->counter_ == 2);
  BOOST_TEST(check_root->left_ != nullptr);
  BOOST_TEST(check_root->left_->value_ == 4);
  BOOST_TEST(check_root->left_->left_ != nullptr);
  BOOST_TEST(check_root->left_->right_ == nullptr);
  BOOST_TEST(check_root->left_->counter_ == 1);
  BOOST_TEST(check_root->left_->left_->value_ == 2);
  BOOST_TEST(check_root->left_->left_->counter_ == 0);
  BOOST_TEST(check_root->left_->left_->left_ == nullptr);
  BOOST_TEST(check_root->left_->left_->right_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InsertNewValuePlacesGreaterThanValuesToTheRight)
{
  TreeWithCounter* tree_ptr {new TreeWithCounter()};
  BOOST_TEST(tree_ptr->root_ptr() == nullptr);
  tree_ptr->insert_new_value(5);

  NodeWithCounter* check_root {tree_ptr->root_ptr()};
  BOOST_TEST(check_root->value_ == 5);

  tree_ptr->insert_new_value(6);

  BOOST_TEST(tree_ptr->is_counter_stack_empty());

  BOOST_TEST(check_root->value_ == 5);
  BOOST_TEST(check_root->counter_ == 1);
  BOOST_TEST(check_root->right_ != nullptr);
  BOOST_TEST(check_root->right_->value_ == 6);
  BOOST_TEST(check_root->right_->left_ == nullptr);
  BOOST_TEST(check_root->right_->right_ == nullptr);
  BOOST_TEST(check_root->right_->counter_ == 0);

  tree_ptr->insert_new_value(8);
  BOOST_TEST(tree_ptr->is_counter_stack_empty());

  BOOST_TEST(check_root->value_ == 5);
  BOOST_TEST(check_root->counter_ == 2);
  BOOST_TEST(check_root->right_ != nullptr);
  BOOST_TEST(check_root->right_->value_ == 6);
  BOOST_TEST(check_root->right_->left_ == nullptr);
  BOOST_TEST(check_root->right_->right_ != nullptr);
  BOOST_TEST(check_root->right_->counter_ == 1);
  BOOST_TEST(check_root->right_->right_->value_ == 8);
  BOOST_TEST(check_root->right_->right_->counter_ == 0);
  BOOST_TEST(check_root->right_->right_->left_ == nullptr);
  BOOST_TEST(check_root->right_->right_->right_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithVectorOfIntegers)
{
  vector<int> input {5, 2, 6, 1, 7, 4, 3};

  TreeWithCounter* tree_ptr {new TreeWithCounter(input)};

  NodeWithCounter* check_root {tree_ptr->root_ptr()};
  BOOST_TEST(check_root->value_ == 5);
  BOOST_TEST(check_root->counter_ == 6);

  // Right subtree.
  BOOST_TEST(check_root->right_->value_ == 6);
  BOOST_TEST(check_root->right_->counter_ == 1);
  BOOST_TEST(check_root->right_->right_->value_ == 7);
  BOOST_TEST(check_root->right_->right_->counter_ == 0);

  BOOST_TEST(check_root->left_->value_ == 2);
  BOOST_TEST(check_root->left_->counter_ == 3);
  BOOST_TEST(check_root->left_->left_->value_ == 1);
  BOOST_TEST(check_root->left_->left_->counter_ == 0);

  BOOST_TEST(check_root->left_->right_->value_ == 4);
  BOOST_TEST(check_root->left_->right_->counter_ == 1);
  BOOST_TEST(check_root->left_->right_->left_->value_ == 3);
  BOOST_TEST(check_root->left_->right_->left_->counter_ == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindKthLargestElementFindsKthLargestElement)
{
  vector<int> input {4, 5, 8, 2};

  TreeWithCounter* tree_ptr {new TreeWithCounter(input)};

  NodeWithCounter* check_root {tree_ptr->root_ptr()};
  BOOST_TEST(check_root->value_ == 4);
  BOOST_TEST(check_root->counter_ == 3);
  BOOST_TEST(check_root->right_->value_ == 5);
  BOOST_TEST(check_root->right_->counter_ == 1);

  auto check_kth_element = tree_ptr->find_kth_largest_element(3);
  BOOST_TEST(check_kth_element == 4);
  //BOOST_TEST(check_kth_element != nullptr);
//  BOOST_TEST(check_kth_element->counter_ == 3);
  tree_ptr->insert_new_value(3);
  BOOST_TEST(tree_ptr->find_kth_largest_element(3) == 4);
  BOOST_TEST(check_root->counter_ == 4);

  //tree_ptr->insert_new_value(5);
  //BOOST_TEST(tree_ptr->find_kth_largest_element(3) == 5);
  //BOOST_TEST(check_root->counter_ == 5);

  tree_ptr->insert_new_value(10);
  BOOST_TEST(tree_ptr->find_kth_largest_element(3) == 5);
  BOOST_TEST(check_root->counter_ == 5);

}

BOOST_AUTO_TEST_SUITE_END() // TreeWithCounter_tests
BOOST_AUTO_TEST_SUITE_END() // BinarySearchTrees_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures