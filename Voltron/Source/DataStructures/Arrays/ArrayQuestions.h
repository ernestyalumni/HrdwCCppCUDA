//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Array type questions.
//------------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_ARRAY_QUESTIONS_H
#define DATA_STRUCTURES_ARRAYS_ARRAY_QUESTIONS_H

#include <string>
#include <vector>

namespace DataStructures
{
namespace Arrays
{
namespace ArrayQuestions
{

namespace CrackingTheCodingInterview
{

//------------------------------------------------------------------------------
/// \details Time complexity O(s) <= O(255), Size complexity
/// (256 * sizeof(bool))
/// \ref https://stackoverflow.com/a/4987875
//------------------------------------------------------------------------------
bool is_unique_character_string(const std::string& s);

} // namespace CrackingTheCodingInterview

namespace LeetCode
{

int max_profit(std::vector<int>& prices);

} // namespace LeetCode

} // namespace ArrayQuestions
} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_ARRAY_QUESTIONS_H