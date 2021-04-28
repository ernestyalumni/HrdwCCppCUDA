//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Array type Questions.
//-----------------------------------------------------------------------------
#include "ArrayQuestions.h"

#include <cstring> // std::memset
#include <vector>

using std::vector;

namespace DataStructures
{
namespace Arrays
{
namespace ArrayQuestions
{

namespace CrackingTheCodingInterview
{

bool is_unique_character_string(const std::string& s)
{
  constexpr std::size_t N {256};

  bool first_seen[N];

  std::memset(first_seen, static_cast<int>(false), 256);

  // O(|s|) time complexity.
  for (char c : s)
  {
    const int ascii_decimal {static_cast<int>(c)};

    if (first_seen[ascii_decimal])
    {
      return false;
    }
    else
    {
      first_seen[ascii_decimal] = true;
    }
  }

  return true;
}

} // namespace CrackingTheCodingInterview

namespace LeetCode
{

void max_profit_recursive_step(
  int& current_max_profit,
  int& i,
  int& j,
  const vector<int>& prices)
{
  int current_buy_ptr {i - 1};

  for (int index {current_buy_ptr + 1}; ;)
  {
    
  }
}

int max_profit(vector<int>& prices)
{
  const int N {static_cast<int>(prices.size())};

  if (N < 2)
  {
    return 0;
  }

  if (N == 2)
  {
    return prices.at(1) > prices.at(0) ? prices.at(1) - prices.at(0) : 0;
  }

  int current_buy_ptr {N - 2};
  int current_sale_ptr {N - 1};
  int current_max_profit {
    prices.at(current_sale_ptr) > prices.at(current_buy_ptr) ? 
      prices.at(current_sale_ptr) - prices.at(current_buy_ptr) : 0};


  return 0;
}


} // namespace LeetCode

} // namespace ArrayQuestions
} // namespace Arrays
} // namespace DataStructures
