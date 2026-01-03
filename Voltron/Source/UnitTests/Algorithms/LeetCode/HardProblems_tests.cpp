#include "Algorithms/LeetCode/HardProblems.h"
#include "DataStructures/BinaryTrees.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

// Ordered by Leetcode number.
// 23. Merge k Sorted Lists
using Algorithms::LeetCode::MergeKSortedLists;
using Algorithms::LeetCode::FirstMissingPositive;
using Algorithms::LeetCode::MinimumWindowSubstring;
using Algorithms::LeetCode::SlidingWindowMaximum;
using Algorithms::LeetCode::WaysToEarnPoints;
using Algorithms::LeetCode::MinimumCostToCutStick;
using Algorithms::LeetCode::NumberOfVisiblePeopleInAQueue;
using DataStructures::BinaryTrees::TreeNode;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeetCode)

//------------------------------------------------------------------------------
/// https://leetcode.com/problems/merge-k-sorted-lists/
/// 23. Merge k Sorted Lists
//------------------------------------------------------------------------------

class MergeKSortedListsTestCases
{
  public:

    MergeKSortedListsTestCases():
      test_cases_{}
    {
      // Example 1
      // First linked list
      MergeKSortedLists::ListNode root_1_1 {1};
      MergeKSortedLists::ListNode c1_1_1 {4};
      MergeKSortedLists::ListNode c2_1_1 {5};
      vector<MergeKSortedLists::ListNode> example_1_1 {
        root_1_1,
        c1_1_1,
        c2_1_1};

      MergeKSortedLists::ListNode root_1_2 {1};
      MergeKSortedLists::ListNode c1_1_2 {3};
      MergeKSortedLists::ListNode c2_1_2 {4};
      vector<MergeKSortedLists::ListNode> example_1_2 {
        root_1_2,
        c1_1_2,
        c2_1_2};

      MergeKSortedLists::ListNode root_1_3 {2};
      MergeKSortedLists::ListNode c1_1_3 {6};
      vector<MergeKSortedLists::ListNode> example_1_3 {root_1_3, c1_1_3};

      vector<vector<MergeKSortedLists::ListNode>> example_1 {
        example_1_1,
        example_1_2,
        example_1_3};

      test_cases_.push_back(example_1);

      // Link nodes in the persistent test_cases_ vectors
      {
        auto& list1 = test_cases_[0][0];
        if (list1.size() > 1)
        {
          list1[0].next_ = &list1[1];
        }
        if (list1.size() > 2)
        {
          list1[1].next_ = &list1[2];
        }
      }
      {
        auto& list2 = test_cases_[0][1];
        if (list2.size() > 1)
        {
          list2[0].next_ = &list2[1];
        }
        if (list2.size() > 2)
        {
          list2[1].next_ = &list2[2];
        }
      }
      {
        auto& list3 = test_cases_[0][2];
        if (list3.size() > 1)
        {
          list3[0].next_ = &list3[1];
        }
      }
    }

    vector<vector<vector<MergeKSortedLists::ListNode>>> test_cases_;
  };

vector<vector<int>> create_merge_k_sorted_lists_expected_output()
{
  return {{1,1,2,3,4,4,5,6}};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MergeKSortedListsBruteForceWorks)
{
  MergeKSortedListsTestCases test_cases {};

  const vector<vector<int>> expected_output {
    create_merge_k_sorted_lists_expected_output()
  };

  const size_t N {test_cases.test_cases_.size()};

  for (size_t i {0}; i < N; ++i)
  {
    vector<MergeKSortedLists::ListNode*> test_input {};
    for (const auto& nodes : test_cases.test_cases_[i])
    {
      MergeKSortedLists::ListNode* node_ptr {
        const_cast<MergeKSortedLists::ListNode*>(&(nodes[0]))};

      test_input.push_back(node_ptr);
    }

    auto output = MergeKSortedLists::merge_k_lists_brute_force(test_input);

    if (output != nullptr)
    {
      for (size_t j {0}; j < expected_output[i].size(); ++j)
      {
        BOOST_TEST(output->value_ == expected_output[i][j]);
        output = output->next_;
        if (output == nullptr && j < expected_output[i].size() - 1)
        {
          BOOST_FAIL("Output list is shorter than expected");
        }
      }
    }
    else
    {
      BOOST_FAIL("Output is nullptr");
    }
  }

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MergeKSortedListsMinHeapWorks)
{
  MergeKSortedListsTestCases test_cases {};

  const vector<vector<int>> expected_output {
    create_merge_k_sorted_lists_expected_output()
  };

  const size_t N {test_cases.test_cases_.size()};

  for (size_t i {0}; i < N; ++i)
  {
    vector<MergeKSortedLists::ListNode*> test_input {};
    for (const auto& nodes : test_cases.test_cases_[i])
    {
      MergeKSortedLists::ListNode* node_ptr {
        const_cast<MergeKSortedLists::ListNode*>(&(nodes[0]))};

      test_input.push_back(node_ptr);
    }

    auto output = MergeKSortedLists::merge_k_lists_min_heap(test_input);

    if (output != nullptr)
    {
      for (size_t j {0}; j < expected_output[i].size(); ++j)
      {
        BOOST_TEST(output->value_ == expected_output[i][j]);
        output = output->next_;
        if (output == nullptr && j < expected_output[i].size() - 1)
        {
          BOOST_FAIL("Output list is shorter than expected");
        }
      }
    }
    else
    {
      BOOST_FAIL("Output is nullptr");
    }
  }

  BOOST_TEST(true);
}

/// 41. First Missing Positive

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FirstMissingPositiveWorks)
{
  {
    vector<int> nums {1,2,0};
    const int expected {3};

    BOOST_TEST(
      FirstMissingPositive::first_missing_positive(nums) == expected);
  }
  {
    vector<int> nums {3,4,-1,1};
    const int expected {2};

    BOOST_TEST(
      FirstMissingPositive::first_missing_positive(nums) == expected);
  }
  {
    vector<int> nums {7,8,9,11,12};
    const int expected {1};

    BOOST_TEST(
      FirstMissingPositive::first_missing_positive(nums) == expected);
  }
}

// 76. Minimum Window Substring
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MinimumWindowSubstringWorksWithSlidingWindow)
{
  {
    const string s {"ADOBECODEBANC"};
    const string t {"ABC"};
    const string expected {"BANC"};

    BOOST_TEST(MinimumWindowSubstring::minimum_window(s, t) == expected);
  }
  {
    const string s {"a"};
    const string t {"a"};
    const string expected {"a"};

    BOOST_TEST(MinimumWindowSubstring::minimum_window(s, t) == expected);
  }
  {
    const string s {"a"};
    const string t {"aa"};
    const string expected {""};

    BOOST_TEST(MinimumWindowSubstring::minimum_window(s, t) == expected);
  }  
}

//------------------------------------------------------------------------------
/// 124. Binary Tree Maximum Path Sum
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BinaryTreeMaximumPathSumFindsMaxSum)
{
  {
    TreeNode example_root {1};
    TreeNode c1l2 {2};
    TreeNode c1r3 {3};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;

    TreeNode* root {&example_root};
    
    const int expected {6};

    BOOST_TEST(BinaryTreeMaximumPathSum::max_path_sum(root) == expected);
  }
  {
    TreeNode example_root {-10};
    TreeNode c1l2 {9};
    TreeNode c1r3 {20};
    TreeNode c3l4 {15};
    TreeNode c3r5 {7};

    example_root.left_ = &c1l2;
    example_root.right_ = &c1r3;
    c1r3.left_ = &c3l4;
    c1r3.right_ = &c3r5;

    TreeNode* root {&example_root};

    const int expected {42};

    BOOST_TEST(BinaryTreeMaximumPathSum::max_path_sum(root) == expected);
  }
}

//------------------------------------------------------------------------------
/// 239. Sliding Window Maximum
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SlidingWindowMaximumWorks)
{
  {
    vector<int> nums {1,3,-1,-3,5,3,6,7};
    const int k {3};
    const vector<int> expected {3,3,5,5,6,7};
    BOOST_TEST(SlidingWindowMaximum::max_sliding_window(nums, k) == expected);
  }
  {
    vector<int> nums {1};
    const int k {1};
    const vector<int> expected {1};
    BOOST_TEST(SlidingWindowMaximum::max_sliding_window(nums, k) == expected);
  }
  // Test case 10 / 51
  {
    vector<int> nums {7, 2, 4};
    const int k {2};
    const vector<int> expected {7, 4};
    BOOST_TEST(SlidingWindowMaximum::max_sliding_window(nums, k) == expected);
  }
}

//------------------------------------------------------------------------------
/// 295. Find Median from Data Stream
//------------------------------------------------------------------------------


/// 1293. Shortest Path in a Grid with Obstacles Elimination
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ShortestPathInGridWithObstaclesFindShortest)
{
  {
    vector<vector<int>> grid {
      {0, 0, 0},
      {1, 1, 0},
      {0, 0, 0},
      {0, 1, 1},
      {0, 0, 0}};
    int k {1};

    int output {6};
  }
  {
    vector<vector<int>> grid {{0, 1, 1}, {1, 1, 1}, {1, 0, 0}};
    int k {1};

    // We needed to eliminate at least 2 obstables to find such a walk.
    int output {-1};
  }
}

/// 1547. Minimum Cost to Cut a Stick
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MinimumCostToCutStickFindsMinimum)
{
  // Example 1
  {
    const int n {7};
    vector<int> cuts {1, 3, 4, 5};
    const int expected {16};

    BOOST_TEST(
      MinimumCostToCutStick::minimum_cost_to_cut_stick(n, cuts) == expected);
  }
  // Example 2
  {
    const int n {9};
    vector<int> cuts {5, 6, 1, 4, 2};
    const int expected {22};

    BOOST_TEST(
      MinimumCostToCutStick::minimum_cost_to_cut_stick(n, cuts) == expected);
  }
}

//------------------------------------------------------------------------------
/// 1799. Maximize Score After N Operations
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MaximizeScoreAfterNOperationsWorksWithDynamicProgramming)
{
  {
    vector<int> nums {1, 2};
    const int expected {1};
    BOOST_TEST(MaximumScoreAfterNOperations::max_score(nums) == expected);
  }
  {
    vector<int> nums {3,4,6,8};
    const int expected {11};
    BOOST_TEST(MaximumScoreAfterNOperations::max_score(nums) == expected);
  }
  {
    vector<int> nums {1,2,3,4,5,6};
    const int expected {14};
    BOOST_TEST(MaximumScoreAfterNOperations::max_score(nums) == expected);
  }
}

/// 1944. Number of Visible People in a Queue
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NumberOfVisiblePeopleInAQueueWorksWithStack)
{
  {
    vector<int> heights {10,6,8,5,11,9};
    vector<int> expected {3,1,2,1,1,0};
    const auto output =
      NumberOfVisiblePeopleInAQueue::can_see_persons_count(heights);

    BOOST_TEST(output == expected);
  }
  {
    vector<int> heights {5,1,2,3,10};
    vector<int> expected {4,1,1,1,0};

    const auto output =
      NumberOfVisiblePeopleInAQueue::can_see_persons_count(heights);

    BOOST_TEST(output == expected);
  }
}

//------------------------------------------------------------------------------
/// 2585. NumberOfWaysToEarnPoints
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WaysToEarnPointsReachesTarget)
{
  // Example 1
  {
    const int target {6};
    vector<vector<int>> types {{6,1}, {3,2}, {2,3}};

    const int expected {7};

    BOOST_TEST(WaysToEarnPoints::ways_to_reach_target(target, types), expected);
  }
  // Example 2
  {
    const int target {5};
    vector<vector<int>> types {{50,1}, {50,2}, {50,5}};

    const int expected {4};

    BOOST_TEST(WaysToEarnPoints::ways_to_reach_target(target, types), expected);
  }
  // Example 3
  {
    const int target {18};
    vector<vector<int>> types {{6,1},{3,2},{2,3}};

    const int expected {1};

    BOOST_TEST(WaysToEarnPoints::ways_to_reach_target(target, types), expected);
  }
}

BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms