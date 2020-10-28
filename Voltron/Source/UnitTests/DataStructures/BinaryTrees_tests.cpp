//------------------------------------------------------------------------------
/// \file BinaryTrees_tests.cpp
/// \date 20201023 03:44
//------------------------------------------------------------------------------
#include "DataStructures/BinaryTrees.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using DataStructures::BinaryTrees::TreeNode;
using DataStructures::BinaryTrees::balance_max_height_recursive;
using DataStructures::BinaryTrees::inorder_traversal_iterative;
using DataStructures::BinaryTrees::inorder_traversal_recursive;
using DataStructures::BinaryTrees::is_same_recursive;
using DataStructures::BinaryTrees::level_order_traversal;
using DataStructures::BinaryTrees::max_depth;
using DataStructures::BinaryTrees::postorder_traversal_iterative_simple;
using DataStructures::BinaryTrees::postorder_traversal_recursive;
using DataStructures::BinaryTrees::preorder_traversal;
using DataStructures::BinaryTrees::preorder_traversal_recursive;
using DataStructures::BinaryTrees::serialize;
using std::string;
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

TreeNode leaf {42};

TreeNode base_case_1_root {1};
TreeNode base_case_1_d11 {2};
TreeNode base_case_1_d12 {3};

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BalanceMaxHeightRecursiveWorksOnSimpleBaseCases)
{
  {
    TreeNode* example_ptr {nullptr};

    const auto result = balance_max_height_recursive(example_ptr);
    BOOST_TEST(result.first == true);
    BOOST_TEST(result.second == -1);
  }
  {
    // Check if we have the test setup initial conditions we want.
    TreeNode* example_root_ptr {&leaf};
    BOOST_TEST(example_root_ptr->left_ == nullptr);
    BOOST_TEST(example_root_ptr->right_ == nullptr);

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == true);
    BOOST_TEST(result.second == 0);
  }
  {
    base_case_1_root.left_ = &base_case_1_d11;
    base_case_1_root.right_ = &base_case_1_d12;
    TreeNode* example_root_ptr {&base_case_1_root};

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == true);
    BOOST_TEST(result.second == 1);
  }
  {
    TreeNode* example_root_ptr {&example_root_B};
    BOOST_TEST(example_root_ptr->left_ != nullptr);
    BOOST_TEST(example_root_ptr->right_ != nullptr);

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == true);
    BOOST_TEST(result.second == 2);
  }
}

TreeNode height_example_1_root {3};
TreeNode height_example_1_d11 {9};
TreeNode height_example_1_d12 {20};
TreeNode height_example_1_d21 {15};
TreeNode height_example_1_d22 {7};

TreeNode height_example_2_root {1};
TreeNode height_example_2_d11 {2};
TreeNode height_example_2_d12 {2};
TreeNode height_example_2_d21 {3};
TreeNode height_example_2_d22 {3};
TreeNode height_example_2_d31 {4};
TreeNode height_example_2_d32 {4};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BalanceMaxHeightRecursiveDeterminesBalanceProperty)
{
  {
    TreeNode* example_root_ptr {&example_root_B};
    BOOST_TEST(example_root_ptr->left_ != nullptr);
    BOOST_TEST(example_root_ptr->right_ != nullptr);

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == true);
    BOOST_TEST(result.second == 2);
  }
  {
    TreeNode* example_root_ptr {&example_root_A};
    BOOST_TEST(example_root_ptr->left_ != nullptr);
    BOOST_TEST(example_root_ptr->right_ != nullptr);

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == true);
    BOOST_TEST(result.second == 2);
  }
  {
    TreeNode* example_root_ptr {&example_root};
    BOOST_TEST(example_root_ptr->left_ != nullptr);
    BOOST_TEST(example_root_ptr->right_ != nullptr);

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == false);
    BOOST_TEST(result.second == 3);
  }
  {
    height_example_1_root.left_ = &height_example_1_d11;
    height_example_1_root.right_ = &height_example_1_d12;
    height_example_1_d12.left_ = &height_example_1_d21;
    height_example_1_d12.right_ = &height_example_1_d22;

    TreeNode* example_root_ptr {&height_example_1_root};
    BOOST_TEST(example_root_ptr->left_ != nullptr);
    BOOST_TEST(example_root_ptr->right_ != nullptr);

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == true);
    BOOST_TEST(result.second == 2);
  }
  {
    height_example_2_root.left_ = &height_example_2_d11;
    height_example_2_root.right_ = &height_example_2_d12;
    height_example_2_d11.left_ = &height_example_2_d21;
    height_example_2_d11.right_ = &height_example_2_d22;
    height_example_2_d21.left_ = &height_example_2_d31;
    height_example_2_d21.right_ = &height_example_2_d32;

    TreeNode* example_root_ptr {&height_example_2_root};
    BOOST_TEST(example_root_ptr->left_ != nullptr);
    BOOST_TEST(example_root_ptr->right_ != nullptr);

    const auto result = balance_max_height_recursive(example_root_ptr);
    BOOST_TEST(result.first == false);
    BOOST_TEST(result.second == 3);
  }
}

/// cf. https://www.techiedelight.com/check-if-two-binary-trees-are-identical-not-iterative-recursive/
/// cf. https://medium.com/techie-delight/binary-tree-interview-questions-and-practice-problems-439df7e5ea1f
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsSameRecursiveReturnsTrueForSameTrees)
{
  {
    TreeNode x {15};
    TreeNode x_d11 {10};
    TreeNode x_d12 {20};
    TreeNode x_d21 {8};
    TreeNode x_d22 {12};
    TreeNode x_d23 {16};
    TreeNode x_d24 {25};

    TreeNode* x_ptr {&x};
    x_ptr->left_ = &x_d11;
    x_ptr->right_ = &x_d12;
    x_ptr->left_->left_ = &x_d21;
    x_ptr->left_->right_ = &x_d22;
    x_ptr->right_->left_ = &x_d23;
    x_ptr->right_->left_ = &x_d24;

    TreeNode y {15};
    TreeNode y_d11 {10};
    TreeNode y_d12 {20};
    TreeNode y_d21 {8};
    TreeNode y_d22 {12};
    TreeNode y_d23 {16};
    TreeNode y_d24 {25};

    TreeNode* y_ptr {&y};
    y_ptr->left_ = &y_d11;
    y_ptr->right_ = &y_d12;
    y_ptr->left_->left_ = &y_d21;
    y_ptr->left_->right_ = &y_d22;
    y_ptr->right_->left_ = &y_d23;
    y_ptr->right_->left_ = &y_d24;

    BOOST_TEST(is_same_recursive(x_ptr, y_ptr));
  }
  {
    TreeNode* example_root_ptr {&example_root};
    TreeNode* example_root_A_ptr {&example_root_A};
    BOOST_TEST(!is_same_recursive(example_root_ptr, example_root_A_ptr));
  }
}

//------------------------------------------------------------------------------
/// cf. https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
//------------------------------------------------------------------------------

TreeNode serialize_example_1_root {1};
TreeNode serialize_example_1_d11 {2};
TreeNode serialize_example_1_d12 {3};
TreeNode serialize_example_1_d21 {4};
TreeNode serialize_example_1_d22 {5};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SerializeSerializesBinaryTree)
{
  serialize_example_1_root.left_ = &serialize_example_1_d11;
  serialize_example_1_root.right_ = &serialize_example_1_d12;
  serialize_example_1_d12.left_ = &serialize_example_1_d21;
  serialize_example_1_d12.right_ = &serialize_example_1_d22;

  TreeNode* example_root_ptr {&serialize_example_1_root};

  const string result {serialize(example_root_ptr)};

  BOOST_TEST(result == "1,2,null,null,3,4,null,null,5,null,null");
}

BOOST_AUTO_TEST_SUITE_END() // BinaryTrees_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures