//------------------------------------------------------------------------------
/// \file LeetCode_tests.cpp
/// \date 20201025 12:51
//------------------------------------------------------------------------------
#include "QuestionsDEntrevue/LeetCode/LeetCodeQuestions.h"

#include <boost/test/unit_test.hpp>
#include <optional>
#include <string>
#include <vector>

using QuestionsDEntrevue::LeetCode::check_palindrome;
using QuestionsDEntrevue::LeetCode::climb_stairs_iterative;
using QuestionsDEntrevue::LeetCode::coin_change_recursive;
using QuestionsDEntrevue::LeetCode::coin_change_top_down;
using QuestionsDEntrevue::LeetCode::coin_change_top_down_step;
using QuestionsDEntrevue::LeetCode::count_palindromic_substrings;
using QuestionsDEntrevue::LeetCode::count_palindromic_substrings_simple;
using QuestionsDEntrevue::LeetCode::find_even_size_palindromes;
using QuestionsDEntrevue::LeetCode::find_odd_size_palindromes;
using QuestionsDEntrevue::LeetCode::is_valid_parentheses;
using QuestionsDEntrevue::LeetCode::longest_valid_parentheses;
using QuestionsDEntrevue::LeetCode::max_profit;
using QuestionsDEntrevue::LeetCode::min_coin_change_recursive_step;
using std::make_optional;
using std::nullopt;
using std::optional;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(LeetCode)

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/valid-parentheses/
/// \name 20. Valid Parentheses. Easy.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsValidParentheseReturnsTrueForValidParentheses)
{
  {
    string example {"()"};

    BOOST_TEST(is_valid_parentheses(example));
  }
  {
    string example {"()[]{}"};

    BOOST_TEST(is_valid_parentheses(example));
  }
  {
    string example {"([)]"};

    BOOST_TEST(!is_valid_parentheses(example));
  }
  {
    string example {"{[]}"};

    BOOST_TEST(is_valid_parentheses(example));
  }

}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/longest-valid-parentheses/
/// \name 32. Longest Valid Parentheses. Hard.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LongestValidParentheseReturnsCorrentValuesForBaseCases)
{
  {
    string example {"(()"};

    BOOST_TEST(longest_valid_parentheses(example) == 2);
  }
  {
    string example {"(())"};

    BOOST_TEST(longest_valid_parentheses(example) == 4);
  }
  {
    string example {")("};
    BOOST_TEST(longest_valid_parentheses(example) == 0);
  }
  {
    string example {")(("};
    BOOST_TEST(longest_valid_parentheses(example) == 0);
  }
  {
    string example {")()())"};

    BOOST_TEST(longest_valid_parentheses(example) == 4);
  }
  {
    string example {""};

    BOOST_TEST(longest_valid_parentheses(example) == 0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
  LongestValidParentheseReturnsLongestLengthOfValidParentheses)
{
  {
    string example {"()()()"};

    BOOST_TEST(longest_valid_parentheses(example) == 6);
  }
  {
    string example {"()(()"};
    BOOST_TEST(longest_valid_parentheses(example) == 2);
  }
}

//------------------------------------------------------------------------------
/// \brief 70. Climbing Stairs.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClimbStairsIterativeFindsNumberOfDistinctWaysForBaseCases)
{
  BOOST_TEST(climb_stairs_iterative(1) == 1);
  BOOST_TEST(climb_stairs_iterative(2) == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClimbStairsIterativeWorksForGreaterThan2)
{
  BOOST_TEST(climb_stairs_iterative(3) == 3);
  BOOST_TEST(climb_stairs_iterative(4) == 5);

  BOOST_TEST(climb_stairs_iterative(42) == 433494437);
}

//------------------------------------------------------------------------------
/// \brief 121. Best Time to Buy and Sell Stock
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MaxProfitsCalculatesMaximumProfit)
{
  {
    vector<int> input {7, 1, 5, 3, 6, 4};

    BOOST_TEST(max_profit(input) == 5);
  }
  {
    vector<int> input {7, 6, 4, 3, 1};

    BOOST_TEST(max_profit(input) == 0);
  }
}

//------------------------------------------------------------------------------
/// \url https://leetcode.com/problems/coin-change/
/// \name 322. Coin Change
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdOptionalUsageExample)
{
  {
    const int amount {11};
    vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);
    BOOST_TEST(min_coins_for_amount.size() == amount + 1);

    BOOST_TEST(!min_coins_for_amount.at(0).has_value());
    BOOST_TEST(!min_coins_for_amount.at(1).has_value());

    min_coins_for_amount.at(0) = 42;
    min_coins_for_amount.at(1) = 69;

    BOOST_TEST(min_coins_for_amount.at(0).has_value());
    BOOST_TEST(min_coins_for_amount.at(1).has_value());
    BOOST_TEST(*min_coins_for_amount.at(0) == 42);
    BOOST_TEST(*min_coins_for_amount.at(1) == 69);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CoinChangeTopDownReturnsCorrectValueForBaseCases)
{
  {
    vector<int> example_1_coins {1, 2, 5};
    const int amount {11};

    BOOST_TEST(coin_change_top_down(example_1_coins, amount) == 3);
  }
  {
    vector<int> example_2_coins {2};
    const int amount {3};

    BOOST_TEST(coin_change_top_down(example_2_coins, amount) == -1);
  }
  {
    vector<int> example_3_coins {1};
    const int amount {0};
    BOOST_TEST(coin_change_top_down(example_3_coins, amount) == 0);
  }
  {
    vector<int> example_4_coins {1};
    const int amount {1};
    BOOST_TEST(coin_change_top_down(example_4_coins, amount) == 1);
  }
  {
    vector<int> example_5_coins {1};
    const int amount {2};
    BOOST_TEST(coin_change_top_down(example_5_coins, amount) == 2);
  }
  {
    vector<int> example_6_coins {186, 419, 83, 408};
    const int amount {6249};
    BOOST_TEST(coin_change_top_down(example_6_coins, amount) == 20);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MinCoinChangeRecursiveStepReturnsCorrectValueForBaseCases)
{
  {
    vector<int> example_1_coins {1, 2, 5};
    const int amount {11};
    vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);

    BOOST_TEST(
      min_coin_change_recursive_step(
        example_1_coins,
        1,
        min_coins_for_amount,
        amount) == 3);
  }
  {
    vector<int> example_2_coins {2};
    const int amount {3};
    vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);

    BOOST_TEST(
      min_coin_change_recursive_step(
        example_2_coins,
        2,
        min_coins_for_amount,
        amount) == -1);
  }
  {
    vector<int> example_3_coins {1};
    const int amount {0};
    vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);
    min_coins_for_amount.at(0) = 0;

    BOOST_TEST(
      min_coin_change_recursive_step(
        example_3_coins,
        1,
        min_coins_for_amount,
        amount) == 0);
  }
  {
    vector<int> example_4_coins {1};
    const int amount {1};
    vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);
    min_coins_for_amount.at(0) = 0;

    BOOST_TEST(
      min_coin_change_recursive_step(
        example_4_coins,
        1,
        min_coins_for_amount,
        amount) == 1);
  }
  {
    vector<int> example_5_coins {1};
    const int amount {2};
    vector<optional<int>> min_coins_for_amount (amount + 1, nullopt);
    min_coins_for_amount.at(0) = 0;

    BOOST_TEST(
      min_coin_change_recursive_step(
        example_5_coins,
        1,
        min_coins_for_amount,
        amount) == 2);
  }
}

//------------------------------------------------------------------------------
/// \brief 647. Palindromic Substrings.
//------------------------------------------------------------------------------

string example_1 {"abc"};
string example_2 {"aaa"};

const string example_str_a {"madam"};
const string example_str_b {"tenet"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StringManipulationWithIteratorsWorks)
{
  // Use this playground if necessary.

  auto start_iter = example_1.begin();

  --start_iter;
  BOOST_TEST(distance(example_1.begin(), example_1.begin()) == 0);
  BOOST_TEST(distance(example_1.begin(), start_iter) == -1);

  // Distance between beginning and end of a string returns size of a string.
  BOOST_TEST(example_str_a.size() ==
    distance(example_str_a.begin(), example_str_a.end()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CheckPalindromeWorksOnBaseCases)
{
  {
    string example {"a"};

    auto tail_iter = example.end();
    --tail_iter;
    BOOST_TEST(check_palindrome(example.begin(), tail_iter));
  }
  {
    string example {"ab"};

    auto tail_iter = example.end();
    --tail_iter;
    BOOST_TEST(!check_palindrome(example.begin(), tail_iter));

    tail_iter = example.end();
    tail_iter = tail_iter - 2;
    BOOST_TEST(check_palindrome(example.begin(), tail_iter));

    tail_iter = example.end();
    --tail_iter;
    BOOST_TEST(check_palindrome(example.begin() + 1, tail_iter));
  }
  {
    string example {"aa"};
    auto tail_iter = example.end();
    --tail_iter;
    BOOST_TEST(check_palindrome(example.begin(), tail_iter));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindEvenSizePalindromesWorksOnBaseCases)
{
  {
    string example {"a"};

    BOOST_TEST(
      find_even_size_palindromes(
        example.begin(),
        example.end(),
        example.begin()) == 0);
  }
  {
    string example {"ab"};

    auto head_iter = example.begin();
    BOOST_TEST(
      find_even_size_palindromes(
        example.begin(),
        example.end(),
        example.begin()) == 0);
  }
  {
    string example {"aa"};
    auto head_iter = example.begin();
    BOOST_TEST(
      find_even_size_palindromes(
        example.begin(),
        example.end(),
        example.begin()) == 1);

    ++head_iter;
    BOOST_TEST(
      find_even_size_palindromes(
        example.begin(),
        example.end(),
        head_iter) == 0);
  }
  {
    auto head_iter = example_2.begin();
    BOOST_TEST(
      find_even_size_palindromes(
        example_2.begin(),
        example_2.end(),
        head_iter) == 1);
    ++head_iter;
    BOOST_TEST(
      find_even_size_palindromes(
        example_2.begin(),
        example_2.end(),
        head_iter) == 1);
    ++head_iter;
    
    BOOST_TEST((head_iter != example_2.end()));
    BOOST_TEST(
      find_even_size_palindromes(
        example_2.begin(),
        example_2.end(),
        head_iter) == 0);  
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FindOddSizePalindromesWorksOnBaseCases)
{
  {
    string example {"a"};

    BOOST_TEST(
      find_odd_size_palindromes(
        example.begin(),
        example.end(),
        example.begin()) == 1);
  }
  {
    string example {"ab"};

    auto head_iter = example.begin();
    BOOST_TEST(
      find_odd_size_palindromes(
        example.begin(),
        example.end(),
        head_iter) == 1);
    ++head_iter;
    BOOST_TEST(
      find_odd_size_palindromes(
        example.begin(),
        example.end(),
        head_iter) == 1);
  }
  {
    string example {"aa"};
    auto head_iter = example.begin();
    BOOST_TEST(
      find_odd_size_palindromes(
        example.begin(),
        example.end(),
        example.begin()) == 1);

    ++head_iter;
    BOOST_TEST(
      find_odd_size_palindromes(
        example.begin(),
        example.end(),
        head_iter) == 1);
  }
  {
    auto head_iter = example_2.begin();
    BOOST_TEST(
      find_odd_size_palindromes(
        example_2.begin(),
        example_2.end(),
        head_iter) == 1);
    ++head_iter;
    BOOST_TEST(
      find_odd_size_palindromes(
        example_2.begin(),
        example_2.end(),
        head_iter) == 2);
    ++head_iter;
    
    BOOST_TEST((head_iter != example_2.end()));
    BOOST_TEST(
      find_odd_size_palindromes(
        example_2.begin(),
        example_2.end(),
        head_iter) == 1);  
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CheckPalindromeWorksOnStrings)
{
  {
    auto tail_iter = example_1.end();
    --tail_iter;
    BOOST_TEST(!check_palindrome(example_1.begin(), tail_iter));
  }
  {
    auto tail_iter = example_2.end();
    --tail_iter;
    BOOST_TEST(check_palindrome(example_2.begin(), tail_iter));
  }
  {
    auto tail_iter = example_str_a.end();
    --tail_iter;
    BOOST_TEST(check_palindrome(example_str_a.begin(), tail_iter));
  }
  {
    auto tail_iter = example_str_b.end();
    --tail_iter;
    BOOST_TEST(check_palindrome(example_str_b.begin(), tail_iter));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CheckPalindromeWorksOnSubStrings)
{
  string example_1 {"polymorphismreferstofunctioncalldependencyontype"};
  auto tail_iter = example_1.end();
  --tail_iter;
  BOOST_TEST(!check_palindrome(example_1.begin(), tail_iter));

  auto head_iter = example_1.begin();
  head_iter = head_iter + string{"polymorphism"}.size();
  BOOST_TEST(check_palindrome(head_iter, head_iter + 4));
  BOOST_TEST(check_palindrome(head_iter + 1, head_iter + 3));
  BOOST_TEST(check_palindrome(head_iter + 2, head_iter + 2));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountPalindromicSubstringsCountsCorrectly)
{
  BOOST_TEST(count_palindromic_substrings(example_1) == 3);
  BOOST_TEST(count_palindromic_substrings(example_2) == 6);
  BOOST_TEST(count_palindromic_substrings(example_str_a) == 7);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountPalindromicSubstringsSimpleCountsCorrectly)
{
  BOOST_TEST(count_palindromic_substrings_simple(example_1) == 3);
  BOOST_TEST(count_palindromic_substrings_simple(example_2) == 6);
  BOOST_TEST(count_palindromic_substrings_simple(example_str_a) == 7);
}

BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Entrevue