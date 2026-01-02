#include "PreEasyExercises.h"

#include <cstdint>
#include <string>
#include <string_view>

using std::string_view;

namespace Algorithms
{
namespace PreEasyExercises
{

uint64_t Recursion::count_number_of_occurrences_of_a_character_iteratively(
  const std::string& s,
  const char c)
{
  uint64_t count {0};
  for (const auto& ch : s)
  {
    if (ch == c)
    {
      ++count;
    }
  }
  return count;
}

uint64_t Recursion::count_number_of_occurrences_of_a_character_helper(
  const string_view& s,
  const char c,
  const uint64_t current_count)
{
  return (s.empty()) ?
    current_count : count_number_of_occurrences_of_a_character_helper(
      string_view{s.begin() + 1, s.end()},
      c, current_count + (s[0] == c ? 1 : 0));
}

uint64_t Recursion::count_number_of_occurrences_of_a_character(
  const std::string& s,
  const char c)
{
  return count_number_of_occurrences_of_a_character_helper(
    string_view{s}, c, 0);
}

} // namespace PreEasyExercises
} // namespace Algorithms