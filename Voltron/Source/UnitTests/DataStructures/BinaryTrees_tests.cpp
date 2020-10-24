//------------------------------------------------------------------------------
/// \file BinaryTrees_tests.cpp
/// \date 20201023 03:44
//------------------------------------------------------------------------------
#include "DataStructures/BinaryTrees.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using DataStructures::BinaryTrees::TreeNode;
using DataStructures::BinaryTrees::inorder_traversal_iterative;
using DataStructures::BinaryTrees::inorder_traversal_recursive;
using DataStructures::BinaryTrees::level_order_traversal;
using DataStructures::BinaryTrees::max_depth;
using DataStructures::BinaryTrees::postorder_traversal_iterative_simple;
using DataStructures::BinaryTrees::postorder_traversal_recursive;
using DataStructures::BinaryTrees::preorder_traversal;
using DataStructures::BinaryTrees::preorder_traversal_recursive;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(BinaryTrees_tests)

// cf. https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/
TreeNode example_root {6};
TreeNode d11 {2};
TreeNode d12 {7};
TreeNode d21 {1};
TreeNode d22 {4};
TreeNode d23 {9};
TreeNode d31 {3};
TreeNode d32 {5};
TreeNode d33 {8};

TreeNode example_root_A {1};
TreeNode d11_A {2};
TreeNode d12_A {3};
TreeNode d21_A {4};
TreeNode d22_A {5};
TreeNode d23_A {6};

TreeNode example_root_B {3};
TreeNode d11_B {9};
TreeNode d12_B {20};
TreeNode d21_B {15};
TreeNode d22_B {7};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PreorderTraversalTraversesFirstEncounters)
{
  example_root.left_ = &d11;
  example_root.right_ = &d12;
  d11.left_ = &d21;
  d11.right_ = &d22;
  d22.left_ = &d31;
  d22.right_ = &d32;
  d12.right_ = &d23;
  d23.right_ = &d33;

  TreeNode* example_root_ptr {&example_root};

  vector<int> result {preorder_traversal(example_root_ptr)};

  vector<int> expected {6, 2, 1, 4, 3, 5, 7, 9, 8};

  BOOST_TEST(result == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PreorderTraversalRecursiveTraversesFirstEncounters)
{
  example_root.left_ = &d11;
  example_root.right_ = &d12;
  d11.left_ = &d21;
  d11.right_ = &d22;
  d22.left_ = &d31;
  d22.right_ = &d32;
  d12.right_ = &d23;
  d23.right_ = &d33;

  TreeNode* example_root_ptr {&example_root};

  vector<int> result {preorder_traversal_recursive(example_root_ptr)};

  vector<int> expected {6, 2, 1, 4, 3, 5, 7, 9, 8};

  BOOST_TEST(result == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InorderTraversalIterativeTraversesAfterLeftSubtree)
{
  example_root_A.left_ = &d11_A;
  example_root_A.right_ = &d12_A;
  d11_A.left_ = &d21_A;
  d11_A.right_ = &d22_A;
  d12_A.left_ = &d23_A;

  TreeNode* example_root_ptr {&example_root_A};

  vector<int> result {inorder_traversal_iterative(example_root_ptr)};

  vector<int> expected {4, 2, 5, 1, 6, 3};

  BOOST_TEST(result == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InorderTraversalRecursiveTraversesAfterLeftSubtree)
{
  example_root.left_ = &d11;
  example_root.right_ = &d12;
  d11.left_ = &d21;
  d11.right_ = &d22;
  d22.left_ = &d31;
  d22.right_ = &d32;
  d12.right_ = &d23;
  d23.right_ = &d33;

  TreeNode* example_root_ptr {&example_root};

  vector<int> result {inorder_traversal_recursive(example_root_ptr)};

  vector<int> expected {1, 2, 3, 4, 5, 6, 7, 9, 8};

  BOOST_TEST(result == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PostorderTraversalIterativeTraversesAfterRightSubtree)
{
  example_root_A.left_ = &d11_A;
  example_root_A.right_ = &d12_A;
  d11_A.left_ = &d21_A;
  d11_A.right_ = &d22_A;
  d12_A.left_ = &d23_A;

  TreeNode* example_root_ptr {&example_root_A};

  vector<int> result {postorder_traversal_iterative(example_root_ptr)};

  vector<int> expected {4, 5, 2, 6, 3, 1};

  BOOST_TEST(result == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PostorderTraversalRecursiveTraversesAfterRightSubtree)
{
  example_root.left_ = &d11;
  example_root.right_ = &d12;
  d11.left_ = &d21;
  d11.right_ = &d22;
  d22.left_ = &d31;
  d22.right_ = &d32;
  d12.right_ = &d23;
  d23.right_ = &d33;

  TreeNode* example_root_ptr {&example_root};

  vector<int> result {postorder_traversal_recursive(example_root_ptr)};

  vector<int> expected {1, 3, 5, 4, 2, 8, 9, 7, 6};

  BOOST_TEST(result == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
  PostorderTraversalIterativeSimpleTraversesAfterRightSubtree)
{
  example_root_A.left_ = &d11_A;
  example_root_A.right_ = &d12_A;
  d11_A.left_ = &d21_A;
  d11_A.right_ = &d22_A;
  d12_A.left_ = &d23_A;

  TreeNode* example_root_ptr {&example_root_A};

  vector<int> result {postorder_traversal_iterative_simple(example_root_ptr)};

  vector<int> expected {4, 5, 2, 6, 3, 1};

  BOOST_TEST(result == expected);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LevelOrderTraversalReturnsNodesInByLevels)
{
  example_root_B.left_ = &d11_B;
  example_root_B.right_ = &d12_B;
  d12_B.left_ = &d21_B;
  d12_B.right_ = &d22_B;

  TreeNode* example_root_ptr {&example_root_B};

  vector<vector<int>> result {level_order_traversal(example_root_ptr)};

  BOOST_TEST(result.at(0) == vector<int>{3});
  BOOST_TEST(result.at(1) == vector<int>({9, 20}));
  BOOST_TEST(result.at(2) == vector<int>({15, 7}));
  BOOST_TEST(result.size() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MaxDepthFindsMaximumDepth)
{
  example_root_B.left_ = &d11_B;
  example_root_B.right_ = &d12_B;
  d12_B.left_ = &d21_B;
  d12_B.right_ = &d22_B;

  TreeNode* example_root_ptr {&example_root_B};

  const int result {max_depth(example_root_ptr)};

  BOOST_TEST(result == 3);
}

BOOST_AUTO_TEST_SUITE_END() // BinaryTrees_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures