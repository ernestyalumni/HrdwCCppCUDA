#include "EasyProblems.h"

#include "DataStructures/BinaryTrees.h"

#include <algorithm> // std::max
#include <functional>
#include <limits.h> // INT_MIN
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility> // std::pair, std::make_pair
#include <vector>

using DataStructures::BinaryTrees::TreeNode;
using std::function;
using std::make_pair;
using std::map;
using std::max;
using std::pair;
using std::queue;
using std::stack;
using std::string;
using std::unordered_set;
using std::vector;

namespace Algorithms
{
namespace LeetCode
{

//------------------------------------------------------------------------------
/// 1. Two Sum
//------------------------------------------------------------------------------

vector<int> TwoSum::brute_force(vector<int>& nums, int target)
{
  const int N {static_cast<int>(nums.size())};

  // O(N^2) time complexity.
  // Given N number of integers, for all pairs of i, j \in 0 .. N - 1, i 1= j
  // find i, j such that nums[i] + nums[j] = target.
  for (int i {0}; i < N - 1; ++i)
  {
    for (int j {i + 1}; j < N; ++j)
    {
      if (nums[i] + nums[j] == target)
      {
        return vector<int>{i, j};
      }
    }
  }

  return vector<int>{};
}

vector<int> TwoSum::two_sum(vector<int>& nums, int target)
{
  const int N {static_cast<int>(nums.size())};

  // Use another data structure to store progress as we traverse the array nums.
  std::map<int, int> value_and_indices {};

  for (int i {0}; i < N; ++i)
  {
    const int complement {target - nums[i]};

    if (value_and_indices.count(complement) > 0)
    {
      return vector<int>{i, value_and_indices[complement]};
    }
    else
    {
      value_and_indices.emplace(nums[i], i);
    }
  }

  return vector<int>{};
}

//------------------------------------------------------------------------------
/// 88. Merge Sorted Array
//------------------------------------------------------------------------------

void MergeSortedArray::merge(
  vector<int>& nums1,
  int m,
  vector<int>& nums2,
  int n)
{
  if (n == 0)
  {
    return;
  }

  if (m == 0)
  {
    nums1 = nums2;
  }

  // The key insight is to start from the end and we know from the end and
  // decrementing, we obtain the largest, and non-increasing.

  int current_index_1 {m - 1};
  int current_index_2 {n - 1};
  int tail {m + n - 1};

  while (tail >= 0)
  {
    if (current_index_1 >= 0 && current_index_2 >= 0)
    {
      if (nums1[current_index_1] > nums2[current_index_2])
      {
        nums1[tail] = nums1[current_index_1];
        --current_index_1;
      }
      else
      {
        nums1[tail] = nums2[current_index_2];
        --current_index_2;
      }

      --tail;
    }
    else if (current_index_2 >= 0)
    {
      while (current_index_2 >= 0)
      {
        nums1[tail] = nums2[current_index_2];
        --current_index_2;
        --tail;
      }
    }
    // Otherwise nums1 is already in non-decreasing order.
    else
    {
      --tail;
    }
  }
}

//------------------------------------------------------------------------------
/// 100. Same Tree
//------------------------------------------------------------------------------
bool SameTree::is_same_tree(TreeNode* p, TreeNode* q)
{
  function<bool(TreeNode*, TreeNode*)> step = [&](TreeNode* pp, TreeNode* qq)
  {
    // If we reached the leaf at any point, return true.
    if ((p == nullptr) && (q == nullptr))
    {
      return true;
    }

    // Nodes have to have the same value, and be there, in the same position.
    if ((p == nullptr) || (q == nullptr) || p->value_ != q->value_)
    {
      return false;
    }

    // Use recursion to solve the subproblem on left and right.

    return step(pp->left_, qq->right_) && step(pp->right_, qq->right_);
  };

  return step(p, q);
}

//------------------------------------------------------------------------------
/// 104. Maximum Depth of Binary Tree
//------------------------------------------------------------------------------
int MaximumDepthOfBinaryTree::max_depth_recursive(TreeNode* root)
{
  // This was a preorder traversal implementation that didn't work because it
  // overcounted.
  /*
  if (root == nullptr)
  {
    return 0;
  }

  int depth {1};
  stack<TreeNode*> unvisited_nodes {};
  unvisited_nodes.push(root);

  while (!unvisited_nodes.empty())
  {
    TreeNode* current_node {unvisited_nodes.top()};
    unvisited_nodes.pop();

    // Push right before left so that, due to the property of a stack, namely
    // that it's LIFO, left is processed before right.

    if (current_node->right_ != nullptr)
    {
      unvisited_nodes.push(current_node->right_);
    }
    if (current_node->left_ != nullptr)
    {
      unvisited_nodes.push(current_node->left_);
    }

    if (current_node->right_ != nullptr || current_node->left_ != nullptr)
    {
      depth++;
    }
  }

  return depth;
  */

  function<int(TreeNode*)> max_depth_recursive_step = [&](TreeNode* node)
  {
    if (node == nullptr)
    {
      return 0;
    }
    // TreeNode node is a leaf.
    if (node->left_ == nullptr && node->right_ == nullptr)
    {
      return 1;
    }

    const int left_depth {max_depth_recursive_step(node->left_)};
    const int right_depth {max_depth_recursive_step(node->right_)};

    // Add 1 to account for the root node itself.
    return max(left_depth, right_depth) + 1;
  };

  return max_depth_recursive_step(root);
}

int MaximumDepthOfBinaryTree::max_depth_iterative(TreeNode* root)
{
  if (root == nullptr)
  {
    return 0;    
  }

  int depth {0};

  queue<TreeNode*> unvisited_nodes {};
  unvisited_nodes.push(root);

  while (!unvisited_nodes.empty())
  {
    // Number of nodes at this current level.
    const int level_size {static_cast<int>(unvisited_nodes.size())};
    depth++;
    for (int i {0}; i < level_size; ++i)
    {
      // For each node at this level, remove it from the queue and add its
      // children to the queue.
      TreeNode* current_node {unvisited_nodes.front()};
      unvisited_nodes.pop();

      // Add all the children nodes for the next level or i.e. next depth.

      if (current_node->left_ != nullptr)
      {
        unvisited_nodes.push(current_node->left_);
      }
      if (current_node->right_ != nullptr)
      {
        unvisited_nodes.push(current_node->right_);
      }
    }
  }

  return depth;
}

//------------------------------------------------------------------------------
/// 121. Best Time to Buy and Sell Stock
/// Key idea: at each step update profit for maximum profit and minimum price in
/// that order.
//------------------------------------------------------------------------------

int BestTimeToBuyAndSellStock::max_profit(vector<int>& prices)
{
  const int N {static_cast<int>(prices.size())};
  int minimum_price {prices[0]};
  int profit {0};

  for (int i {0}; i < N; ++i)
  {
    const int current_profit {prices[i] - minimum_price};

    if (current_profit > profit)
    {
      profit = current_profit;
    }

    if (prices[i] < minimum_price)
    {
      minimum_price = prices[i];
    }
  }

  return profit;
}

//------------------------------------------------------------------------------
/// 125. Valid Palindrome
//------------------------------------------------------------------------------

bool ValidPalindrome::is_palindrome(string s)
{
  const int valid_min {static_cast<int>('a')};
  const int valid_max {static_cast<int>('z')};

  // Numbers are ok ("alphanumeric characters include letters and numbers.")
  const int valid_numbers_min {static_cast<int>('0')};
  const int valid_numbers_max {static_cast<int>('9')};

  const int valid_upper_case_min {static_cast<int>('A')};
  const int valid_upper_case_max {static_cast<int>('Z')};

  // O(|s|) space complexity.
  vector<char> stripped_s {};

  // O(|s|) time complexity.
  for (const char c : s)
  {
    const int c_value {static_cast<int>(c)};

    if ((c_value <= valid_max && c_value >= valid_min) ||
      (c_value <= valid_numbers_max && c_value >= valid_numbers_min))
    {
      stripped_s.emplace_back(c);
    }
    else if (c_value <= valid_upper_case_max && c_value >= valid_upper_case_min)
    {
      stripped_s.emplace_back(c - ('A' - 'a'));
    }
  }

  int l {0};
  int r {static_cast<int>(stripped_s.size()) - 1};
  while (l <= r)
  {
    if (stripped_s[l] != stripped_s[r])
    {
      return false;
    }

    ++l;
    --r;
  }

  return true;
}

//------------------------------------------------------------------------------
/// 169. Majority Element
//------------------------------------------------------------------------------

int MajorityElement::majority_element_with_map(vector<int>& nums)
{
  const size_t N {nums.size()};

  if (N == 1)
  {
    return nums.at(0);
  }

  const size_t floor_half_of_N {N / 2};

  std::unordered_map<int, int> element_to_count {};

  // O(N) time complexity.
  for (int num : nums)
  {
    // O(1) amoritized (each hash map operation, insertion and lookup, is O(1)
    // amoritized).
    if (element_to_count.count(num) == 0)
    {
      element_to_count[num] = 1;
    }
    else if (element_to_count[num] == floor_half_of_N)
    {
      return num;
    }
    else
    {
      element_to_count[num] += 1;
    }
  }

  return INT_MIN;
}

int MajorityElement::majority_element_with_voting(vector<int>& nums)
{
  const size_t N {nums.size()};
  const size_t floor_half_of_N {N / 2};

  int candidate {nums.at(0)};
  int count {1};

  for (size_t i {1}; i < N; ++i)
  {
    if (nums[i] == candidate)
    {
      if (++count > floor_half_of_N)
      {
        return candidate;
      }
    }
    else if (--count == 0)
    {
      candidate = nums[i];
      count = 1;
    }
  }

  return candidate;
}

//------------------------------------------------------------------------------
/// 217. Contains Duplicate
//------------------------------------------------------------------------------

bool ContainsDuplicate::contains_duplicate(vector<int>& nums)
{
  unordered_set<int> seen_numbers {};

  for (const auto num : nums)
  {
    // O(1) time complexity, amoritized.
    if (seen_numbers.count(num) == 0)
    {
      seen_numbers.emplace(num);
    }
    else
    {
      return true;
    }
  }

  return false;
}

//------------------------------------------------------------------------------
/// 226. Invert Binary Tree
//------------------------------------------------------------------------------
TreeNode* InvertBinaryTree::invert_tree_recursive(TreeNode* root)
{
  function<TreeNode*(TreeNode*)> invert_tree_step = [&](TreeNode* node)
  {
    // Base case: if node is nullptr or a leaf, there's nothing to invert.
    if (node == nullptr ||
      ((node->left_ == nullptr) && (node->right_ == nullptr)))
    {
      return node;
    }

    // These are the left and right children, respectively, with their
    // respective children inverted.
    TreeNode* left_child_inverted {invert_tree_step(node->left_)};
    TreeNode* right_child_inverted {invert_tree_step(node->right_)};

    // We do the swap after the recursion step because of 2 reasons:
    // 1. If you do the swap first, then you run recursion that literally says
    // to invert the node, they will be swapped twice, and
    // 2. Consider this test case:
    //   1
    //  / \
    // 2   3
    node->left_ = right_child_inverted;
    node->right_ = left_child_inverted;

    return node;
  };

  return invert_tree_step(root);
}

TreeNode* InvertBinaryTree::invert_tree_iterative(TreeNode* root)
{
  if (root == nullptr ||
    ((root->left_ == nullptr) && (root->right_ == nullptr)))
  {
    return root;
  }

  // Imitate the recursion stack of the recursive version.
  stack<TreeNode*> unvisited_nodes {};
  unvisited_nodes.push(root);

  while (!unvisited_nodes.empty())
  {
    TreeNode* current_node {unvisited_nodes.top()};
    unvisited_nodes.pop();

    // Push the right first after the left because stack is LIFO, so that we'll
    // pop the left child first.
    if (current_node != nullptr)
    {
      unvisited_nodes.push(current_node->right_);
      unvisited_nodes.push(current_node->left_);

      TreeNode* temp {current_node->left_};
      current_node->left_ = current_node->right_;
      current_node->right_ = temp;
    }
  }

  return root;
}

//------------------------------------------------------------------------------
/// 242. Valid Anagram
//------------------------------------------------------------------------------
bool ValidAnagram::is_anagram(string s, string t)
{
  if (s.size() != t.size())
  {
    return false;
  }
  // Use unordered_map for O(1) amoritized access.
  // For each letter, map it to the number of times it was seen in string s.
  // O(S) space complexity, where S is number of unique characters in s.
  std::unordered_map<char, int> letter_to_counts {};

  // O(|s|) time complexity.
  for (const char c : s)
  {
    letter_to_counts[c] += 1;
  }

  // O(|t|) time complexity.
  for (const char c : t)
  {
    if (letter_to_counts.count(c) != 1)
    {
      return false;
    }
    else
    {
      letter_to_counts[c] -= 1;
    }
  }

  for (const auto& [key, counts] : letter_to_counts)
  {
    if (letter_to_counts[key] != 0)
    {
      return false;
    }
  }

  return true;
}

//------------------------------------------------------------------------------
/// 704. Binary Search
//------------------------------------------------------------------------------
int BinarySearch::search(vector<int>& nums, int target)
{
  const int N {static_cast<int>(nums.size())};

  int l {0};
  int r {N - 1};

  while (l <= r)
  {
    const int mid { (r - l) / 2 + l};

    if (target == nums[mid])
    {
      return mid;
    }
    else if (target < nums[mid])
    {
      r = mid - 1;
    }
    else if (target > nums[mid])
    {
      l = mid + 1;
    }
  }

  return -1;
}

//------------------------------------------------------------------------------
/// 733. Flood Fill
//------------------------------------------------------------------------------

vector<vector<int>> FloodFill::flood_fill(
  vector<vector<int>>& image,
  int sr,
  int sc,
  int color)
{
  if (image[sr][sc] == color)
  {
    return image;
  }
  const int M {static_cast<int>(image.size())};
  const int N {static_cast<int>(image[0].size())};

  const int original_color {image[sr][sc]};

  function<void(const int, const int)> search_depth = [&](
    const int i,
    const int j)
  {
    // Base case: if it's not a pixel on the image or a pixel of the starting
    // color, do nothing.
    if (0 > i || i >= M || 0 > j || j > N || image[i][j] != original_color)
    {
      return;
    }

    image[i][j] = color;

    search_depth(i + 1, j);
    search_depth(i - 1, j);
    search_depth(i, j + 1);
    search_depth(i, j - 1);
  };

  search_depth(sr, sc);

  return image;
}

vector<vector<int>> FloodFill::flood_fill_with_queue(
  vector<vector<int>>& image,
  int sr,
  int sc,
  int color)
{
  if (image[sr][sc] == color)
  {
    return image;
  }
  const int M {static_cast<int>(image.size())};
  const int N {static_cast<int>(image[0].size())};

  const int original_color {image[sr][sc]};

  auto is_valid = [&](const int i, const int j)
  {
    return (
      0 <= i && i < M && 0 <= j && j < N && image[i][j] == original_color);
  };

  queue<pair<int, int>> unvisited_pixels {};
  unvisited_pixels.push(make_pair(sr, sc));
  // Immediately mark the target pixel as being visited by coloring it.
  image[sr][sc] = color;

  while (!unvisited_pixels.empty())
  {
    const auto ij = unvisited_pixels.front();
    unvisited_pixels.pop();

    const int I {get<0>(ij)};
    const int J {get<1>(ij)};

    if (is_valid(I + 1, J))
    {
      unvisited_pixels.push(make_pair(I + 1, J));
      // Mark pixel immediately upon pushing into the queue with color.
      image[I + 1][J] = color;
    }
    if (is_valid(I - 1, J))
    {
      unvisited_pixels.push(make_pair(I - 1, J));
      // Mark pixel immediately upon pushing into the queue with color.
      image[I - 1][J] = color;
    }
    if (is_valid(I, J + 1))
    {
      unvisited_pixels.push(make_pair(I, J + 1));
      // Mark pixel immediately upon pushing into the queue with color.
      image[I][J + 1] = color;
    }
    if (is_valid(I, J - 1))
    {
      unvisited_pixels.push(make_pair(I, J - 1));
      // Mark pixel immediately upon pushing into the queue with color.
      image[I][J - 1] = color;
    }
  }

  return image;
}

//------------------------------------------------------------------------------
/// 1646. Get Maximum in Generated Array.
//------------------------------------------------------------------------------

int GetMaximumInGeneratedArray::get_maximum_generated(int n)
{
  if (n == 0 || n == 1)
  {
    return n;
  }

  // Use -1 as a value to show that there wasn't a value before.
  std::vector<int> values (n + 1, -1);

  values[0] = 0;
  values[1] = 1;
  int maximum {1};

  // O(N) time complexity.
  for (int i {2}; i < n + 1; ++i)
  {
    if (values[i] == -1)
    {
      // i is even,
      if (i % 2 == 0)
      {
        values[i] = values[i / 2];
      }
      // i is odd
      else
      {
        values[i] = values[i / 2] + values[i / 2 + 1];
      }
    }

    if (values[i] > maximum)
    {
      maximum = values[i];
    }
  }

  return maximum;
}

} // namespace LeetCode
} // namespace Algorithms
